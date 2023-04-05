import numpy as np
import copy

def subsample_subdataset(ds, sens_key, rate = 1.0):
    ds = copy.deepcopy(ds)
    sens_list = []
    assert rate >= 0.0 and rate <= 1.0
    for idx in ds.indices:
        #print(train_ds.dataset[idx])
        sens_list += [ds.dataset[idx][sens_key]]
        #break  
    sens_list = np.array(sens_list)

    # statistic inform extract
    sens_num_dict = {}
    for sens_ in set(sens_list):
        sens_num_dict[sens_] = np.sum(sens_list == sens_)
    min_num = min(sens_num_dict.values())

    # balance indices
    balance_indices = []
    for sens_ in set(sens_list):
        # find indices for sens_ group
        sens_idx = np.where(sens_list == sens_)[0]
        if len(sens_idx) == min_num:
            # minority class do not need sub sample
            balance_indices += [sens_idx]
        else:
            sample_num = int(len(sens_idx) - rate * (len(sens_idx) - min_num))
            sub_sample_idx = np.random.choice(sens_idx, sample_num, replace=False)
            balance_indices += [sub_sample_idx]
            #print(sens_, sub_sample_idx)
    balance_indices = np.concatenate(balance_indices)
    balance_indices = list((np.array(ds.indices)[balance_indices]))
    ds.indices = balance_indices
    return ds