import numpy as np
from typing import List, Optional, Dict
import torch
from tqdm import tqdm
from pyhealth.trainer import get_metrics_fn


def evaluate(trainer, dataloader) -> Dict[str, float]:
    """Evaluates the model.

    Args:
        dataloader: Dataloader for evaluation.

    Returns:
        scores: a dictionary of scores.
    """
    y_true_all, y_prob_all, loss_mean = trainer.inference(dataloader)

    mode = trainer.model.mode
    metrics_fn = get_metrics_fn(mode)
    if y_prob_all.shape[1] ==2:
        y_prob_all = y_prob_all[:, 1]
    scores = metrics_fn(y_true_all, y_prob_all, metrics=trainer.metrics)
    scores["loss"] = loss_mean
    return scores

def fair_check(trainer, dataloader) -> Dict[str, float]:
    """Evaluates fairness metric for the model.

    Args:
        dataloader: Dataloader for evaluation.

    Returns:
        scores: a dictionary of scores.
    """
    loss_all = []
    y_true_all = []
    y_prob_all = []
    sens_true_all = []
    for data in tqdm(dataloader, desc="Evaluation"):
        trainer.model.eval()
        with torch.no_grad():
            output = trainer.model(**data)
            loss = output["loss"]
            y_true = output["y_true"].cpu().numpy()
            y_prob = output["y_prob"].cpu().numpy()
            sens_true = output["sens_true"].cpu().numpy()
            loss_all.append(loss.item())
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
            sens_true_all.append(sens_true)
    loss_mean = sum(loss_all) / len(loss_all)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    sens_true_all = np.concatenate(sens_true_all, axis=0)

    # return y_true_all, y_prob_all, sens_true_all, loss_mean
    # calculate fairness metric, depend on mode (task) and sens_mode (sens)
    # only support binary sens now


    mode = trainer.model.mode
    metrics_fn = get_metrics_fn(mode)

    # get score for different group
    #print('y_prob_all', y_prob_all)
    #print('y_true_all', y_true_all)
    #print('sens_true_all', sens_true_all)
    def get_scores(idx):
        scores = metrics_fn(y_true_all[idx], y_prob_all[idx], metrics=trainer.metrics)
        return scores

    scores_0 = get_scores(sens_true_all==0)
    scores_1 = get_scores(sens_true_all==1)

    fair_scores = {}

    for metric in scores_0.keys():
        fair_scores[metric] = np.abs(scores_0[metric] - scores_1[metric])
    return fair_scores