# %%
#%load_ext autoreload
#%autoreload 2
from resample import subsample_subdataset
import numpy as np
from utils import evaluate, fair_check
import torch


from models import Adv_Transformer, Laftr_Transformer
from models.LAFTR import Laftr_Trainer

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.tasks import drug_recommendation_mimic3_fn, mortality_prediction_mimic3_fn, readmission_prediction_mimic3_fn
from tasks import add_feature

SENS_KEY = 'gender'


root = '/data/qf31/FBen/mimic-iii-clinical-database-1.4'
mimic3base = MIMIC3Dataset(
    root=root,
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"], # PRESCRIPTIONS
    # map all NDC codes to ATC 3-rd level codes in these tables
    code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    dev=True,
)

# %%
#from pyhealth.tasks import drug_recommendation_mimic3_fn, mortality_prediction_mimic3_fn, readmission_prediction_mimic3_fn
mimic3sample = mimic3base.set_task(task_fn=mortality_prediction_mimic3_fn) # use default task
mimic3sample = add_feature(mimic3sample, mimic3base, [SENS_KEY])


# %%
final_result = {}
for hyper_param in [0.0, 0.25, 0.5, 0.75, 1.0]:
    final_result[hyper_param] = {'auc':[], 'auc_diff': []}
    for _ in range(5):

        train_ds, val_ds, test_ds = split_by_patient(mimic3sample, [0.6, 0.2, 0.2])
        # create dataloaders (torch.data.DataLoader)
        train_loader = get_dataloader(train_ds, batch_size=3200, shuffle=True)
        val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
        test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

        model = Laftr_Transformer(
            dataset=mimic3sample,# the dataset should provide sens feat
            feature_keys=["conditions", "procedures", 'drugs'], # the model should specify the sens feat
            label_key="label",
            sens_key=SENS_KEY,
            mode="binary",
            sens_mode = 'binary',
            fair_coeff= hyper_param,
            model_var='laftr-dp',
        )
        trainer = Laftr_Trainer(model=model,device='cuda:0', metrics=['roc_auc'])

        #trainer = Trainer(model=model,device='cuda:0')
        trainer.train(
            train_dataloader= train_loader,
            val_dataloader=val_loader,
            epochs=400,
            monitor="roc_auc",
            #adv_rate = subsample_rate,
        )
        try:
            final_result[hyper_param]['auc'] += [evaluate(trainer, test_loader)['roc_auc']]
            final_result[hyper_param]['auc_diff'] += [fair_check(trainer, test_loader)['roc_auc']]
        except:
            pass
    final_result[hyper_param]['auc'] = {'mean': np.mean(final_result[hyper_param]['auc']), 'std': np.std(final_result[hyper_param]['auc'])}
    final_result[hyper_param]['auc_diff'] = {'mean': np.mean(final_result[hyper_param]['auc_diff']), 'std': np.std(final_result[hyper_param]['auc_diff'])}



# %%
torch.save(final_result, 'laftr_test_result')
