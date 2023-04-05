from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn

import numpy as np



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

def reset_sens_fc(laftr_model) -> nn.Module:
    #self.sens_fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, self.get_output_size(self.sens_tokenizer))
    hidden_size = len(laftr_model.feature_keys) * laftr_model.embedding_dim
    sens_size = laftr_model.get_output_size(laftr_model.sens_tokenizer)

    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.LeakyReLU(),
        nn.Linear(hidden_size, sens_size),
        )


def lnl_forward(lnl_model, **kwargs) -> Dict[str, torch.Tensor]:
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
    patient_emb = lnl_model.embed_forward(**kwargs)
    # (patient, label_size)
    logits    = lnl_model.fc(patient_emb)
    y_true    = lnl_model.prepare_labels(kwargs[lnl_model.label_key], lnl_model.label_tokenizer)
    sens_true = lnl_model.prepare_sens_labels(kwargs[lnl_model.sens_key], lnl_model.sens_tokenizer)
    y_prob    = lnl_model.prepare_y_prob(logits)
    
    # loss for label predict, cross entropy, first term in original paper
    loss = lnl_model.get_loss_function()(logits, y_true)

    # loss for distribution Q, neglog likelihood loss, second term in original paper
    # first get prob via softmax
    sens_logits = lnl_model.sens_fc(patient_emb)

    sens_prob = nn.functional.softmax(sens_logits, dim=-1)
    auxiliary_loss = torch.mean(torch.sum(sens_prob*torch.log(sens_prob),1))

    # loss for sens label predict
    sens_logits_reverse = lnl_model.sens_fc(grad_reverse(patient_emb))
    sens_loss = lnl_model.get_sens_loss_function()(sens_logits_reverse, sens_true)

    return {
        "patient_emb": patient_emb,
        "logits": logits,
        "sens_logits":sens_logits,
        "y_true": y_true,
        "y_prob": y_prob,
        "sens_true": sens_true,
        'loss': loss,
        "auxiliary_loss": auxiliary_loss,
        "sens_loss": sens_loss,
    }
