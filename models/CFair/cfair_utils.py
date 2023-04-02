from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn

import numpy as np
from models import BaseModel


_lambda = 0.00  # lambda is set to 0.01 in LNL source code

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1

def grad_reverse(x):
    return GradReverse.apply(x)

def reset_sens_fc(model: BaseModel) -> nn.Module:
    # a sens_fc for each y class
    hidden_size = len(model.feature_keys) * model.embedding_dim
    sens_size = model.get_output_size(model.sens_tokenizer)

    return nn.ModuleList( [
    nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.LeakyReLU(),
        nn.Linear(hidden_size, sens_size),
        )
    for _ in range(model.label_distri.__len__()) ]
    )

def get_reweight(cfair_model):
    """
    re-implement of code from CFair paper
    https://github.com/hanzhaoml/ICLR2020-CFair/blob/4f03012bc362ff1caf5170b983c140a258a95bb6/main_adult.py#L171
    """
    if cfair_model.reweight_target_tensor and cfair_model.reweight_attr_tensors:
        return cfair_model.reweight_target_tensor, cfair_model.reweight_attr_tensors

    train_target_attrs = np.array(cfair_model.dataset.get_all_tokens(cfair_model.sens_key, remove_duplicates=False, sort=False))
    train_target_labels = np.array(cfair_model.dataset.get_all_tokens(cfair_model.label_key, remove_duplicates=False, sort=False))
    A_keys = cfair_model.dataset.get_all_tokens(cfair_model.sens_key)
    Y_keys = cfair_model.dataset.get_all_tokens(cfair_model.label_key)
    train_idx = train_target_attrs == A_keys[0]
    train_base_0, train_base_1 = np.mean(train_target_labels[train_idx]), np.mean(train_target_labels[~train_idx])
    train_y_1 = np.mean(train_target_labels == Y_keys[1])
    # For reweighing purpose.
    if cfair_model.model_var == "cfair":
        reweight_target_tensor = torch.FloatTensor([1.0 / (1.0 - train_y_1), 1.0 / train_y_1]).to(cfair_model.device)
    elif cfair_model.model_var == "cfair-eo":
        reweight_target_tensor = torch.FloatTensor([1.0, 1.0]).to(cfair_model.device)
    else:
        raise NotImplementedError

    reweight_attr_0_tensor = torch.FloatTensor([1.0 / (1.0 - train_base_0), 1.0 / train_base_0]).to(cfair_model.device)
    reweight_attr_1_tensor = torch.FloatTensor([1.0 / (1.0 - train_base_1), 1.0 / train_base_1]).to(cfair_model.device)
    reweight_attr_tensors = [reweight_attr_0_tensor, reweight_attr_1_tensor]


    return reweight_target_tensor, reweight_attr_tensors


    #train_target_attrs = np.argmax(adult_train.A, axis=1)
    #train_target_labels = np.argmax(adult_train.Y, axis=1)
    #train_idx = train_target_attrs == 0
    #train_base_0, train_base_1 = np.mean(train_target_labels[train_idx]), np.mean(train_target_labels[~train_idx])
    #train_y_1 = np.mean(train_target_labels)
    ## For reweighing purpose.
    #if args.model == "cfair":
    #    reweight_target_tensor = torch.tensor([1.0 / (1.0 - train_y_1), 1.0 / train_y_1]).to(device)
    #elif args.model == "cfair-eo":
    #    reweight_target_tensor = torch.tensor([1.0, 1.0]).to(device)
    #reweight_attr_0_tensor = torch.tensor([1.0 / (1.0 - train_base_0), 1.0 / train_base_0]).to(device)
    #reweight_attr_1_tensor = torch.tensor([1.0 / (1.0 - train_base_1), 1.0 / train_base_1]).to(device)
    #reweight_attr_tensors = [reweight_attr_0_tensor, reweight_attr_1_tensor]

def cfair_forward(model:BaseModel, **kwargs) -> Dict[str, torch.Tensor]:
    """Forward propagation.

    The label `kwargs[self.label_key]` is a list of labels for each patient.

    Args:
        **kwargs: keyword arguments for the model. The keys must contain
            all the feature keys and the label key.

    Returns:
        A dictionary with the following keys:
            loss: a scalar tensor representing the loss.
            y_prob: a tensor representing the predicted probabilities.
            y_true: a tensor representing the true labels.
    """
    patient_emb = model.embed_forward(**kwargs)
    # (patient, label_size)
    logits    = model.fc(patient_emb)
    y_true    = model.prepare_labels(kwargs[model.label_key], model.label_tokenizer) # shape bs for multiclass
    sens_true = model.prepare_sens_labels(kwargs[model.sens_key], model.sens_tokenizer)
    y_prob    = model.prepare_y_prob(logits)
    reweight_target_tensor, reweight_attr_tensors = get_reweight(model)

    # loss for label predict
    #print('reweight_target_tensor: ', reweight_target_tensor)
    loss = model.get_loss_function()(logits, y_true, weight=reweight_target_tensor)
    
    
    # loss for sens label predict
    reverse_patient_emb = grad_reverse(patient_emb)

    #sens_logits = model.sens_fc(reverse_patient_emb)
    
    sens_loss_list = []
    for j in range(model.label_distri.__len__()):
        idx = y_true == j # choose j class sampels
        #print('idx', idx)
        sens_logits = model.sens_fc[j](reverse_patient_emb[idx])
        sens_loss_list += [model.get_sens_loss_function()(sens_logits, sens_true[idx], weight=reweight_attr_tensors[j])]

    sens_loss = torch.mean(torch.stack(sens_loss_list))


    total_loss = loss + model.mu * sens_loss
    return {
        "patient_emb": patient_emb,
        "logits": logits,
        #"sens_logits":sens_logits,
        "y_true": y_true,
        "y_prob": y_prob,
        "sens_true": sens_true,
        'loss': loss,
        "sens_loss": sens_loss,
        "total_loss": total_loss
    }
