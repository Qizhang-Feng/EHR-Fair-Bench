# EHR Fair Bench Doc
## Environment
conda env create -f environment.yml

## General pipeline
The code is built base on pyhealth package. The trainning logic is:
1. create base dataset(MIMIC3Dataset etc.) 
2. create sample dataseet via set_task func, extract data attribute and label ( mortality_prediction_mimic3_fn etc.)
3. add sensitive attribute information via add_feature func (add sens attribute)
4. split train/test/val
5. Build model (BaseTransformer etc.)
6. Build trainer, specify model, device and metrics
7. trainer.train
8. evaluate

See subsample.ipynb as example.


## Code Structure
### datasets
raw data
### models
base model and debias methods realization inherit from base model, for example
base_model->base_transformer->LAFTR/laftr_transformer
#### base_model.py
base model inherit from pyhealth.models.BaseModel, two not implement functions **embed_forward (forward with embed representation) and forward**. Other func doc is in the code file.
#### base_transformer.py
Base transformer model inherit from base_model. Layers are defined in this level. Base forward and embed_forward function is defined here.

#### debias methods
Debias methods are realized in this level. It could contains three file: a model file, a trainer file and a utils file. A model file is needed when debias model has special layers that are different from base model. A trainer file is needed when trainning process is different from base_trainer. A utils file is needed for some special function.

For example, LAFTR contains three files. laftr_transformer.py inherit from base_transformer with a sensitive attribute prediction head. laftr_trainer.py rewrite the train function from base_trainer for adversarial training. utils.py contains the special functions for laftr method.


### output
trainning log file

### tasks
tasks function for set_task function.

### trainers
base_trainer.py

### utils.py
utils function such as evaluation and fairness check function.

## Debias method
debias methods are listed in this link: https://docs.google.com/presentation/d/1WTpIGyBGG_BjkKxRXKS2t_c_PvWQn0omcF8qIn7otlU/edit?usp=sharing

1. Month 1: Dataset collection \& processing: The first month will be dedicated to collecting and processing the EHR data that will be used for the benchmark. This will involve identifying a suitable dataset, obtaining necessary permissions and approvals, and preparing the data for use in the benchmark. We simultaneously look for all possible suitable mitigation algorithms in this phase.
2. Month 2: Code base implementation: In the second month, we will focus on implementing the code base for the benchmark. This will involve designing and implementing the ML models and any necessary dataset preprocessing, any required infrastructure for conducting the experiments, as well as all the implementation of mitigation methods. 
3. Month 3: Experiments conduction: During the third month, we will conduct the experiments to evaluate the fair ML models on the EHR data. This will involve training and evaluating the models using the various metrics and techniques proposed in the benchmark, and collecting and analyzing the results.
4. Month 4: Result analysis \& paper writing: In the final month, we will analyze the results of the experiments and write the paper detailing the findings and recommendations of the benchmark work. This will involve preparing the manuscript, reviewing and revising the manuscript based on feedback, and finalizing the paper for submission.
