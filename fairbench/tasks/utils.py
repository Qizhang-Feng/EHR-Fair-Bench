def add_feature(sample_dataset, base_dataset, feat_list):
    for s_idx in range(len(sample_dataset.samples)):
        #print(s_idx)
        #print(mimic3sample.samples[s_idx].keys())
        # get patient from base dataset
        patient = base_dataset.patients[sample_dataset.samples[s_idx]['patient_id']]
        #print(patient.gender, patient.ethnicity)
        for f in feat_list:
            sample_dataset.samples[s_idx][f] = getattr(patient, f)
    sample_dataset.input_info = sample_dataset._validate()
    return sample_dataset