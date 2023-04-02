# %%
#%load_ext autoreload
#%autoreload 2
from resample import subsample_subdataset
import numpy as np
from utils import evaluate, fair_check

from pyhealth.datasets import MIMIC3Dataset

SENS_KEY = 'gender'
root = '/data/qf31/FBen/mimic-iii-clinical-database-1.4'
mimic3base = MIMIC3Dataset(
    root=root,
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"], # PRESCRIPTIONS
    # map all NDC codes to ATC 3-rd level codes in these tables
    code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    dev=True,
    refresh_cache=False,
)

# %%
from pyhealth.tasks import drug_recommendation_mimic3_fn, mortality_prediction_mimic3_fn, readmission_prediction_mimic3_fn
from pyhealth.datasets import split_by_patient, get_dataloader, split_by_visit
from tasks import add_feature
#from tasks import drug_recommendation_mimic3_fn
mimic3sample = mimic3base.set_task(task_fn=mortality_prediction_mimic3_fn) # use default task
mimic3sample = add_feature(mimic3sample, mimic3base, [SENS_KEY])
train_ds, val_ds, test_ds = split_by_visit(mimic3sample, [0.6, 0.2, 0.2])

# create dataloaders (torch.data.DataLoader)
train_loader = get_dataloader(train_ds, batch_size=3200, shuffle=True)
val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

# %%
from models import Adv_Transformer,Laftr_Transformer, LNL_Transformer
from models import Laftr_Transformer
model = LNL_Transformer(
    dataset=mimic3sample,# the dataset should provide sens feat
    feature_keys=["conditions", "procedures", "drugs"], # the model should specify the sens feat
    label_key="label",
    sens_key='gender',
    mode="binary",
    sens_mode = 'binary',
    #fair_coeff=0.0,
    #model_var='laftr-dp'
    # the model should provide sensitive attribute
    num_layers=2,
)

# %%
from pyhealth.trainer import Trainer
from models.LAFTR import Laftr_Trainer
from models.LNL import LNL_Trainer

trainer = LNL_Trainer(model=model,device='cuda:0', metrics=['roc_auc'])

#trainer = Trainer(model=model,device='cuda:0')
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=800,
    monitor='roc_auc',#"pr_auc_samples",
    #adv_rate = 10.0,
    #optimizer_params = {'lr':1e-4}
)

# %%
from utils import fair_check, evaluate
print(evaluate(trainer, test_loader))

# %%
from utils import fair_check
print(fair_check(trainer, test_loader))

# %%
'''
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
'''