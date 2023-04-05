# %%
#%load_ext autoreload
#%autoreload 2
from resample import subsample_subdataset
import numpy as np
from utils import evaluate, fair_check
import torch

from pyhealth.datasets import MIMIC3Dataset
from models import Adv_Transformer
from pyhealth.datasets import split_by_patient, get_dataloader
from tasks import drug_recommendation_mimic3_fn

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
mimic3sample = mimic3base.set_task(task_fn=drug_recommendation_mimic3_fn) # use default task
train_ds, val_ds, test_ds = split_by_patient(mimic3sample, [0.8, 0.1, 0.1])
#train_ds, val_ds, test_ds = split_by_patient(mimic3sample, [1.0, 0.0, 0.0])

# create dataloaders (torch.data.DataLoader)
train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

# %%
model = Adv_Transformer(
    dataset=mimic3sample,# the dataset should provide sens feat
    feature_keys=["conditions", "procedures"], # the model should specify the sens feat
    label_key="drugs",
    sens_key=SENS_KEY,
    mode="multilabel",
    sens_mode = 'multiclass',
    # the model should provide sensitive attribute
)

# %%
from pyhealth.trainer import Trainer

trainer = Trainer(model=model,device='cuda:0')

#trainer = Trainer(model=model,device='cuda:0')
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=1,
    monitor="pr_auc_samples",
    #adv_rate = 10.0,
)

# %%
from utils import fair_check, evaluate
evaluate(trainer, test_loader)

# %%
from utils import fair_check
fair_check(trainer, test_loader)

# %%
final_result = {}
for subsample_rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
    final_result[subsample_rate] = {'auc':[], 'auc_diff': []}
    for _ in range(5):
        model = Adv_Transformer(
            dataset=mimic3sample,# the dataset should provide sens feat
            feature_keys=["conditions", "procedures"], # the model should specify the sens feat
            label_key="drugs",
            sens_key=SENS_KEY,
            mode="multilabel",
            sens_mode = 'multiclass',
            # the model should provide sensitive attribute   
        )
        trainer = Trainer(model=model,device='cuda:0')

        #trainer = Trainer(model=model,device='cuda:0')
        trainer.train(
            train_dataloader= get_dataloader(subsample_subdataset(train_ds, SENS_KEY, subsample_rate), batch_size=32, shuffle=True),
            val_dataloader=val_loader,
            epochs=500,
            monitor="pr_auc_samples",
            #adv_rate = subsample_rate,
        )
        
        final_result[subsample_rate]['auc'] += [evaluate(trainer, test_loader)['pr_auc_samples']]
        final_result[subsample_rate]['auc_diff'] += [fair_check(trainer, test_loader)['pr_auc_samples']]

    final_result[subsample_rate]['auc'] = {'mean': np.mean(final_result[subsample_rate]['auc']), 'std': np.std(final_result[subsample_rate]['auc'])}
    final_result[subsample_rate]['auc_diff'] = {'mean': np.mean(final_result[subsample_rate]['auc_diff']), 'std': np.std(final_result[subsample_rate]['auc_diff'])}



# %%
torch.save(final_result, 'subsample_test_result')
