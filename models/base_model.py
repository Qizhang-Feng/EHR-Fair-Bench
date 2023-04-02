from abc import ABC
from typing import List, Dict, Union, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyhealth.models
from pyhealth.datasets import SampleDataset
from pyhealth.models.utils import batch_to_multihot
from pyhealth.tokenizer import Tokenizer

class BaseModel(pyhealth.models.BaseModel):
    """BaseModel.
    This BaseModel wrap the pyhealth.models.BaseModel to specify the sens_key and sens_mode
    """
    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: List[str],
        label_key: str,
        sens_key: str, 
        mode: str,
        sens_mode: str,
        embedding_dim: int = 128,
        **kwargs
    ):

        super(BaseModel, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.sens_key = sens_key
        self.sens_mode = sens_mode
        self.sens_fc: Union[nn.Module, nn.ModuleDict, nn.ModuleList]
        self.fc: nn.Module
        self.label_tokenizer: Tokenizer
        self.sens_tokenizer: Tokenizer

        self.label_distri: Dict[str, int]
        self.sens_distri: Dict[str, int]
        
    def get_sens_tokenizer(self, special_tokens=None) -> Tokenizer:
        """Gets the default sens tokenizers using `self.sens_key`.

        Args:
            special_tokens: a list of special tokens to add to the tokenizer.
                Default is empty list.

        Returns:
            sens_tokenizer: the sens tokenizer.
        """
        if special_tokens is None:
            special_tokens = []
        sens_tokenizer = Tokenizer(
            self.dataset.get_all_tokens(key=self.sens_key),
            special_tokens=special_tokens,
        )
        return sens_tokenizer

    def get_sens_loss_function(self) -> Callable:
        """Gets the default loss function using `self.mode`.

        The default loss functions are:
            - binary: `F.binary_cross_entropy_with_logits`
            - multiclass: `F.cross_entropy`
            - multilabel: `F.binary_cross_entropy_with_logits`

        Returns:
            The default loss function.
        """
        if self.sens_mode == "binary":
            return F.binary_cross_entropy_with_logits
        elif self.sens_mode == "multiclass":
            return F.cross_entropy
        elif self.sens_mode == "multilabel":
            return F.binary_cross_entropy_with_logits
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def prepare_sens_labels(
        self,
        labels: Union[List[str], List[List[str]]],
        label_tokenizer: Tokenizer,
    ) -> torch.Tensor:
        """Prepares the labels for model training and evaluation.

        This function converts the labels to different formats depending on the
        mode. The default formats are:
            - binary: a tensor of shape (batch_size, 1)
            - multiclass: a tensor of shape (batch_size,)
            - multilabel: a tensor of shape (batch_size, num_labels)

        Args:
            labels: the raw labels from the samples. It should be
                - a list of str for binary and multiclass classificationa
                - a list of list of str for multilabel classification
            label_tokenizer: the label tokenizer.

        Returns:
            labels: the processed labels.
        """
        if self.sens_mode in ["binary"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.FloatTensor(labels).unsqueeze(-1)
        elif self.sens_mode in ["multiclass"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.LongTensor(labels)
        elif self.sens_mode in ["multilabel"]:
            # convert to indices
            labels_index = label_tokenizer.batch_encode_2d(
                labels, padding=False, truncation=False
            )
            # convert to multihot
            num_labels = label_tokenizer.get_vocabulary_size()
            labels = batch_to_multihot(labels_index, num_labels)
        else:
            raise NotImplementedError
        labels = labels.to(self.device)
        return labels

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


    def embed_forward(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError