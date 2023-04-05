from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn

import numpy as np
from models import BaseModel


def reset_sens_fc(model: BaseModel) -> nn.Module:
    # a sens_fc for each y class
    hidden_size = model.label_distri.__len__() # logits as input of sens fc
    if model.model_var == 'eo':
        hidden_size += model.get_output_size(model.label_tokenizer) # logit and Y as input
    sens_size = model.get_output_size(model.sens_tokenizer)

    return nn.Sequential(
        nn.Linear(hidden_size, 100), # "A has a single 100-unit ReLU hidden layer" 
        nn.LeakyReLU(),
        nn.Linear(100, sens_size),
        )
    


def afal_forward(model:BaseModel, **kwargs) -> Dict[str, torch.Tensor]:
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
    y_true    = model.prepare_labels(kwargs[model.label_key], model.label_tokenizer) #- binary: a tensor of shape (batch_size, 1)- multiclass: a tensor of shape (batch_size,) - multilabel: a tensor of shape (batch_size, num_labels)
    sens_true = model.prepare_sens_labels(kwargs[model.sens_key], model.sens_tokenizer)
    y_prob    = model.prepare_y_prob(logits)


    # loss for label predict
    #print('reweight_target_tensor: ', reweight_target_tensor)
    loss = model.get_loss_function()(logits, y_true)
    
    
    # loss for sens label predict
    sens_fc_input = logits if model.model_var == 'dp' else torch.concat( (logits, y_true.view(y_true.shape[0], -1)), dim=1 )
    sens_logits = model.sens_fc(sens_fc_input)
    sens_loss = model.get_sens_loss_function()(sens_logits, sens_true)


    total_loss = loss + model.alpha * sens_loss
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
