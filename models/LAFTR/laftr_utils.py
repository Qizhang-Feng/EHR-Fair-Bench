from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn

import numpy as np

from models import BaseModel, BaseTransformer
from pyhealth.datasets import SampleDataset
from pyhealth.models import TransformerLayer




def reset_sens_fc(laftr_model) -> nn.Module:
    #self.sens_fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, self.get_output_size(self.sens_tokenizer))
    hidden_size = len(laftr_model.feature_keys) * laftr_model.embedding_dim
    if laftr_model.model_var != 'laftr-dp':
        hidden_size += 1 
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.LeakyReLU(),
        nn.Linear(hidden_size, 1),
        )

def get_AY_proportions(laftr_model):
    if laftr_model.AY_proportion:
        return laftr_model.AY_proportion
    
    #A_num_class = 2
    #Y_num_class = 2
    #A_label = self.A
    #Y_label = self.Y
    
    #A = self.A.tolist()
    #Y = self.Y.tolist()
    #ttl = len(A)

    A = laftr_model.dataset.get_all_tokens(laftr_model.sens_key, remove_duplicates=False, sort=False)
    Y = laftr_model.dataset.get_all_tokens(laftr_model.label_key, remove_duplicates=False, sort=False)
    ttl = laftr_model.dataset.__len__()
        
    #len_A0Y0 = len([ay for ay in zip(A, Y) if ay == (0, 0)])
    #len_A0Y1 = len([ay for ay in zip(A, Y) if ay == (0, 1)])
    #len_A1Y0 = len([ay for ay in zip(A, Y) if ay == (1, 0)])
    #len_A1Y1 = len([ay for ay in zip(A, Y) if ay == (1, 1)])

    A0_key, A1_key = laftr_model.dataset.get_all_tokens(laftr_model.sens_key)
    Y0_key, Y1_key = laftr_model.dataset.get_all_tokens(laftr_model.label_key) 

    len_A0Y0 = len([ay for ay in zip(A, Y) if ay == (A0_key, Y0_key)])
    len_A0Y1 = len([ay for ay in zip(A, Y) if ay == (A0_key, Y1_key)])
    len_A1Y0 = len([ay for ay in zip(A, Y) if ay == (A1_key, Y0_key)])
    len_A1Y1 = len([ay for ay in zip(A, Y) if ay == (A1_key, Y1_key)])

    assert (
        len_A0Y0 + len_A0Y1 + len_A1Y0 + len_A1Y1
    ) == ttl, "Problem computing train set AY proportion."
    A0Y0 = len_A0Y0 / ttl
    A0Y1 = len_A0Y1 / ttl
    A1Y0 = len_A1Y0 / ttl
    A1Y1 = len_A1Y1 / ttl
    
    laftr_model.AY_proportion = [[A0Y0, A0Y1], [A1Y0, A1Y1]]
    
    return laftr_model.AY_proportion

def get_A_proportions(laftr_model):
    AY = get_AY_proportions(laftr_model)
    ret = [AY[0][0] + AY[0][1], AY[1][0] + AY[1][1]]
    np.testing.assert_almost_equal(np.sum(ret), 1.0)
    return ret

def get_Y_proportions(laftr_model):
    AY = get_AY_proportions(laftr_model)
    ret = [AY[0][0] + AY[1][0], AY[0][1] + AY[1][1]]
    np.testing.assert_almost_equal(np.sum(ret), 1.0)
    return ret

def get_AYweights(laftr_model):
    A_weights, Y_weights, AY_weights = (
        get_A_proportions(laftr_model),
        get_Y_proportions(laftr_model),
        get_AY_proportions(laftr_model),
    )
    return A_weights, Y_weights, AY_weights

def l1_loss(y, y_logits):
    """Returns l1 loss"""
    #y_hat = torch.sigmoid(y_logits)
    y_hat = torch.squeeze(torch.sigmoid(y_logits))
    return torch.squeeze(torch.abs(y - y_hat))

def get_weighted_aud_loss(laftr_model, L, Y, A, A_wts, AY_wts):
    """Returns weighted discriminator loss"""
    #Y = Y[:, 0] y_true shape  (batchsize, 1)
    Y = torch.squeeze(Y)
    if laftr_model.model_var == "laftr-dp":
        A0_wt = A_wts[0]
        A1_wt = A_wts[1]
        wts = A0_wt * (1 - A) + A1_wt * A
        #print('wts.shape', wts.shape)
        wtd_L = L * torch.squeeze(wts)
    elif (
            laftr_model.model_var == "laftr-eqodd"
            or laftr_model.model_var == "laftr-eqopp0"
            or laftr_model.model_var == "laftr-eqopp1"
    ):
        A0_Y0_wt = AY_wts[0][0]
        A0_Y1_wt = AY_wts[0][1]
        A1_Y0_wt = AY_wts[1][0]
        A1_Y1_wt = AY_wts[1][1]

        if laftr_model.model_var == "laftr-eqodd":
            wts = (
                    A0_Y0_wt * (1 - A) * (1 - Y)
                    + A0_Y1_wt * (1 - A) * (Y)
                    + A1_Y0_wt * (A) * (1 - Y)
                    + A1_Y1_wt * (A) * (Y)
            )
        elif laftr_model.model_var == "laftr-eqopp0":
            wts = A0_Y0_wt * (1 - A) * (1 - Y) + A1_Y0_wt * (A) * (1 - Y)
        elif laftr_model.model_var == "laftr-eqopp1":
            wts = A0_Y1_wt * (1 - A) * (Y) + A1_Y1_wt * (A) * (Y)
        else:
            raise Exception("Wrong model_var")
        wtd_L = L * torch.squeeze(wts)
    else:
        raise Exception("Wrong model name")
        exit(0)
    #print('wtd_L.shape', wtd_L.shape)
    return wtd_L

def laftr_forward(laftr_model, **kwargs) -> Dict[str, torch.Tensor]:
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
    patient_emb = laftr_model.embed_forward(**kwargs)
    # (patient, label_size)
    logits = laftr_model.fc(patient_emb)
    y_true =laftr_model.prepare_labels(kwargs[laftr_model.label_key], laftr_model.label_tokenizer)
    sens_true = laftr_model.prepare_sens_labels(kwargs[laftr_model.sens_key], laftr_model.sens_tokenizer)
    
    #print('y_true[:, 0]', y_true[:, 0])
    if laftr_model.model_var != 'laftr-dp':
        sens_logits = laftr_model.sens_fc( torch.cat([patient_emb, y_true.to(laftr_model.device)], dim=1)  )
    else:
        sens_logits = laftr_model.sens_fc(patient_emb)


    
    loss = laftr_model.get_loss_function()(logits, y_true)
    
    #print('sens_true.shape', sens_true)
    y_prob = laftr_model.prepare_y_prob(logits)
    
    class_loss = laftr_model.class_coeff * loss
    sens_loss = l1_loss(sens_true, sens_logits)
    aud_loss = -laftr_model.fair_coeff * sens_loss

    A_weights, Y_weights, AY_weights = get_AYweights(laftr_model)#self.get_AYweights(self.train_data)
    weighted_aud_loss = get_weighted_aud_loss(laftr_model, aud_loss, y_true, sens_true, A_weights,
                                                    AY_weights)
    weighted_aud_loss = torch.mean(weighted_aud_loss)
    total_loss = class_loss + weighted_aud_loss

    return {
        "patient_emb": patient_emb,
        "logits": logits,
        "sens_logits":sens_logits,
        "y_true": y_true,
        "y_prob": y_prob,
        "sens_true": sens_true,
        'loss': loss,
        'class_loss': class_loss,
        'weighted_aud_loss': weighted_aud_loss,
        'total_loss': total_loss,
        "sens_loss": sens_loss,
    }