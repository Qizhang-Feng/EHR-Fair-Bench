from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn

import numpy as np

from models import BaseModel, BaseTransformer
from pyhealth.datasets import SampleDataset
from pyhealth.models import TransformerLayer

from .laftr_utils import *

class Laftr_Transformer(BaseTransformer):
    """Transformer model.

    This model applies a separate Transformer layer for each feature, and then
    concatenates the final hidden states of each Transformer layer. The concatenated
    hidden states are then fed into a fully connected layer to make predictions.

    Note:
        We use separate Transformer layers for different feature_keys.
        Currentluy, we automatically support different input formats:
            - code based input (need to use the embedding table later)
            - float/int based value input
        We follow the current convention for the transformer model:
            - case 1. [code1, code2, code3, ...]
                - we will assume the code follows the order; our model will encode
                each code into a vector and apply transformer on the code level
            - case 2. [[code1, code2]] or [[code1, code2], [code3, code4, code5], ...]
                - we will assume the inner bracket follows the order; our model first
                use the embedding table to encode each code into a vector and then use
                average/mean pooling to get one vector for one inner bracket; then use
                transformer one the braket level
            - case 3. [[1.5, 2.0, 0.0]] or [[1.5, 2.0, 0.0], [8, 1.2, 4.5], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run transformer directly
                on the inner bracket level, similar to case 1 after embedding table
            - case 4. [[[1.5, 2.0, 0.0]]] or [[[1.5, 2.0, 0.0], [8, 1.2, 4.5]], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run transformer directly
                on the inner bracket level, similar to case 2 after embedding table

        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        embedding_dim: the embedding dimension. Default is 128.
        **kwargs: other parameters for the Transformer layer.

    Examples:
        >>> from pyhealth.datasets import SampleDataset
        >>> samples = [
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-0",
        ...             "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
        ...             "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
        ...             "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
        ...             "list_list_vectors": [
        ...                 [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
        ...                 [[7.7, 8.5, 9.4]],
        ...             ],
        ...             "label": 1,
        ...         },
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-1",
        ...             "list_codes": [
        ...                 "55154191800",
        ...                 "551541928",
        ...                 "55154192800",
        ...                 "705182798",
        ...                 "70518279800",
        ...             ],
        ...             "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
        ...             "list_list_codes": [["A04A", "B035", "C129"]],
        ...             "list_list_vectors": [
        ...                 [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
        ...             ],
        ...             "label": 0,
        ...         },
        ...     ]
        >>> dataset = SampleDataset(samples=samples, dataset_name="test")
        >>>
        >>> from pyhealth.models import Transformer
        >>> model = Transformer(
        ...         dataset=dataset,
        ...         feature_keys=[
        ...             "list_codes",
        ...             "list_vectors",
        ...             "list_list_codes",
        ...             "list_list_vectors",
        ...         ],
        ...         label_key="label",
        ...         mode="binary",
        ...     )
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {'loss': tensor(0.4234, grad_fn=<NllLossBackward0>), 'y_prob': tensor([[9.9998e-01, 2.2920e-05],
                [5.7120e-01, 4.2880e-01]], grad_fn=<SoftmaxBackward0>), 'y_true': tensor([0, 1])}
        >>>

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
        aud_steps:int = 1,
        class_coeff:float = 1.0,
        fair_coeff:float = 1.0,
        model_var:str = 'laftr-dp',
        **kwargs
    ):
        super(Laftr_Transformer, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            sens_key=sens_key, 
            mode=mode,
            sens_mode = sens_mode,
        )

        self.label_distri = self.dataset.get_distribution_tokens(label_key)
        self.sens_distri  = self.dataset.get_distribution_tokens(sens_key)

        assert (self.label_distri.__len__() == 2 and self.sens_distri.__len__() == 2) , "LAFTER only support binary label and sensitive attribute!"

        self.AY_proportion:Optional[List] = None

        self.aud_steps = aud_steps
        self.class_coeff = class_coeff
        self.fair_coeff = fair_coeff

        assert model_var in ['laftr-dp', 'laftr-eqodd', "laftr-eqopp0", "laftr-eqopp1"]
        self.model_var = model_var

        # redefine the self.sens_fc with LeakyReLU, sens output size is 1 for l1 loss
        self.sens_fc = reset_sens_fc(self)


    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        output = laftr_forward(self, **kwargs)
        return output