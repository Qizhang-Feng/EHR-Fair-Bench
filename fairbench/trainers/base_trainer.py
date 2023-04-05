import logging
import os
from datetime import datetime
from typing import Dict, List, Type, Callable
from typing import Optional, Union

import numpy as np
import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.autonotebook import trange

from pyhealth.trainer import set_logger, logger, is_best, get_metrics_fn


from models import BaseModel

class Base_Trainer:
    """Trainer for PyTorch models.

    Args:
        model: PyTorch model.
        checkpoint_path: Path to the checkpoint. Default is None, which means
            the model will be randomly initialized.
        metrics: List of metric names to be calculated. Default is None, which
            means the default metrics in each metrics_fn will be used.
        device: Device to be used for training. Default is None, which means
            the device will be GPU if available, otherwise CPU.
        enable_logging: Whether to enable logging. Default is True.
        output_path: Path to save the output. Default is "./output".
        exp_name: Name of the experiment. Default is current datetime.
    """

    def __init__(
        self,
        model,
        checkpoint_path: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        device: Optional[str] = None,
        enable_logging: bool = True,
        output_path: Optional[str] = None,
        exp_name: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.metrics = metrics
        self.device = device


        # set logger
        if enable_logging:
            if output_path is None:
                output_path = os.path.join(os.getcwd(), "output")
            if exp_name is None:
                exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.exp_path = os.path.join(output_path, exp_name)
            set_logger(self.exp_path)
        else:
            self.exp_path = None

        # set device
        self.model.to(self.device)

        # logging
        logger.info(self.model)
        logger.info(f"Metrics: {self.metrics}")
        logger.info(f"Device: {self.device}")

        # load checkpoint
        if checkpoint_path is not None:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.load_ckpt(checkpoint_path)

        logger.info("")
        return


    def train(self):
        raise NotImplementedError

    def inference(self, dataloader):
        """Model inference.

        Args:
            dataloader: Dataloader for evaluation.

        Returns:
            y_true_all: List of true labels.
            y_prob_all: List of predicted probabilities.
            loss_mean: Mean loss over batches.
        """
        loss_all = []
        y_true_all = []
        y_prob_all = []
        for data in tqdm(dataloader, desc="Evaluation"):
            self.model.eval()
            with torch.no_grad():
                output = self.model(**data)
                loss = output["loss"]
                y_true = output["y_true"].cpu().numpy()
                y_prob = output["y_prob"].cpu().numpy()
                loss_all.append(loss.item())
                y_true_all.append(y_true)
                y_prob_all.append(y_prob)
        loss_mean = sum(loss_all) / len(loss_all)
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)
        return y_true_all, y_prob_all, loss_mean

    def evaluate(self, dataloader) -> Dict[str, float]:
        from utils import evaluate
        return evaluate(self, dataloader)

        
    def save_ckpt(self, ckpt_path: str) -> None:
        """Saves the model checkpoint."""
        state_dict = self.model.state_dict()
        torch.save(state_dict, ckpt_path)
        return

    def load_ckpt(self, ckpt_path: str) -> None:
        """Saves the model checkpoint."""
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        return