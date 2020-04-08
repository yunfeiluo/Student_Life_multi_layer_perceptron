import numpy as np

from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from src.utils import data_conversion_utils as conversions


SPLITTER_RANDOM_STATE = 100

# ---------------------------------------------------------------- 
# Independent Study 496, Student Stree Prediction
#
# File_name: cross_val
# Functionality: 
#   a) split (stratified split) the dataset for cross validation
#   b) leave one student out split
# Author: Yunfei Luo
# Start date: EST Feb.22th.2020
# Last update: EST Apr.8th.2020
# ----------------------------------------------------------------

def leave_one_subject_out_split(data: dict, groups: dict, ids: list, subject='students'):
    """
    @param data: data for which the splits are needed to be generated.
    @param groups: dictionary, map: student_id -> group_id
    @param ids: the list of id that want to be leaved out
    @param subject: determine leave what subject out
    @return: Return list of dictionary (map: train_ids, val_ids -> data_keys)
    """
    print('########## leave one out split, subject: ' + subject + '##########')
    splits = list()

    data_keys = data['data'].keys()

    student_key = dict() # map: id -> keys
    for key in data_keys:
        if subject == 'students':
            try:
                student_key[key.split('_')[0]].append(key)
            except:
                student_key[key.split('_')[0]] = [key]
        elif subject == 'groups':
            try:
                student_key[groups['student_' + key.split('_')[0]]].append(key)
            except:
                student_key[groups['student_' + key.split('_')[0]]] = [key]
        else:
            print('No such subject: ' + subject)
            exit()
        
    #for student in student_key:
    for student in ids:
        splitting_dict = dict()
        splitting_dict['train_ids'] = list()
        for rest_student in student_key:
            if rest_student != student:
                for key in student_key[rest_student]:
                    splitting_dict['train_ids'].append(key)
        splitting_dict['val_ids'] = student_key[student]
        splits.append(splitting_dict)

    return splits

def get_k_fod_cross_val_splits_stratified_by_students(data: dict, groups:dict, n_splits=5,
                                                      stratification_type="students"):
    """
    @param data: data for which the splits are needed to be generated.
    @param groups: map: student_ids -> groups ids
    @param n_splits: number of split
    @param stratification_type: deterimine the criteria for stratified split
    @return: Return list of dictionary (map: train_ids, val_ids -> data_keys)
    """
    
    print('########## k_fold stratification split, stratified by: ' + stratification_type + '############')
    print('split n: ' + str(n_splits))
    splits = list()

    data_keys = data['data'].keys()

    # determine values in stratified column
    stratification_column = list()
    pos = 0 if stratification_type == "students" else -1 if stratification_type == 'labels' else None
    if pos != None:
        for key in data_keys:
            stratification_column.append(int(key.split('_')[pos]))
    elif stratification_type == 'groups':
        for key in data_keys:
            stratification_column.append(int(groups['student_' + key.split('_')[0]].split('_')[-1]))
    elif stratification_type == 'student_label':
        keys, labels = conversions.extract_keys_and_labels_from_dict(data)
        student_ids = conversions.extract_student_ids_from_keys(keys)
        for i in range(len(student_ids)):
            stratification_column.append(str(student_ids[i]) + "_" + str(labels[i]))
    elif stratification_type == 'group_label':        
        for key in data_keys:
            stratification_column.append(groups['student_' + key.split('_')[0]].split('_')[-1] + '_' + str(data['data'][key][-1]))
    else:
        print('No such kind of criteria for splitting!!!')
        exit()

    # splitting
    data_keys = np.array(list(data_keys))
    stratification_column = np.array(list(stratification_column))
    splitter = StratifiedKFold(n_splits=n_splits, random_state=SPLITTER_RANDOM_STATE)
    for train_index, val_index in splitter.split(X=data_keys, y=stratification_column):

        splitting_dict = dict()
        splitting_dict['train_ids'] = data_keys[train_index].tolist()
        splitting_dict['val_ids'] = data_keys[val_index].tolist()
        splits.append(splitting_dict)

    return splits
