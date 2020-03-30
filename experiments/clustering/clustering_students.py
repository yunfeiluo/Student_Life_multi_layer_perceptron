# ---------------------------------------------------------------- 
# Independent Study 496, Student Stree Prediction
#
# file_name: clustering_students.py
# Functionality: clustering students, 
#    return dict: map: group_ids(str) -> list_of_student_id(list(str))
# Author: Yunfei Luo
# Start date: EST Feb.22th.2020
# Last update: EST Feb.27th.2020
# ----------------------------------------------------------------

import random
import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import model_selection
import src.utils.data_conversion_utils as conversions
from src.utils.read_utils import read_pickle
from src.utils import write_utils
from src.experiments.clustering.dtw_ import DTW_clusters

import matplotlib.pyplot as plt

def get_features(student_list, features): # helper function for kmeans
    '''
    @param student_list: list of student_id
    @param features: list of (string)feature
    '''
    # read survey data
    survey = 'Data/data/surveys_and_covariates/high_lelvel_aggregated_data.csv'
    df = pd.read_csv(survey) 
    N = len(student_list)

    # get the index of student in student_list
    ids = [int(student.split('_')[1]) for student in student_list]
    ind_pos = dict() # map: student_id -> position
    for i in range(len(df['student_id'])):
        ind_pos[int(df['student_id'][i])] = i
    ind = [ind_pos[id_] for id_ in ids]

    # cleansing, by remove NaN to avg
    isna = df.isnull()
    for feature in features:
        have_data = [df[feature][i] for i in ind if not isna[feature][i]]
        avg = sum(have_data) / len(have_data)
        for i in [j for j in ind if isna[feature][j]]:
            df[feature][i] = avg

    # extract data of features
    student_features = dict()

    for i in ind:
        row = np.array([])
        for feature in features:
            row = np.append(row, [df[feature][i]])
        student_features[int(df['student_id'][i])] = row
    
    return df, student_features

def kmeans_features(student_list, features, k): # build kmeans model
    '''
    @param student_list: list of student id, in the form (string)student_id
    @param features: list of (string)feature
    @param k: number of centers for kmeans clustering
    '''
    # read survey data
    df, student_features = get_features(student_list, features)
    A = np.array([student_features[i] for i in student_features])

    # plot
    x_ = list()
    y_ = list()
    for i in range(len(student_list)):
        x_.append(df[features[2]][i])
        y_.append(df[features[0]][i])
    
    fig, ax = plt.subplots()
    ax.scatter(x_, y_)
    for i in range(len(student_list)):
        ax.annotate(str(df[features[1]][i])[:5], (x_[i], y_[i]))

    #plt.scatter(x_, y_)
    plt.xlabel('avg_ddl_per_week')
    plt.ylabel('avg_hours_slept')
    plt.show()
    
    # kmeans clustering
    kmeans = KMeans(n_clusters = k, random_state=0).fit(A)
    centers = kmeans.cluster_centers_
    print('centers are: ')
    print(centers)

    # build group dictionary
    groups = dict()
    for student in student_list:
        quality = student_features[int(student.split('_')[1])]
        belongs = kmeans.predict([quality])[0]
        groups[student] = 'group_'+str(belongs)
        
    return groups

# original model
def one_for_each(student_list):
    groups = dict()
    for i in range(len(student_list)):
        groups[student_list[i]] = 'group_' + str(i)
    return groups

# clustering based on average stress
def avg_stress_cluster(student_list, data, k):
    '''
    @param student_list: list of student id, in the form (string)student_id
    @param data: actual data, dict: (string)keys -> data
    @param k: number of centers for kmeans clustering
    '''
    # compute averages
    stress = dict()
    for i in data:
        keys = i.split('_')
        try:
            stress['student_'+keys[0]].append(int(keys[-1]))
        except:
            stress['student_'+keys[0]] = [int(keys[-1])]
    max_stress = -1
    for i in stress:
        stress[i] = sum(stress[i]) / len(stress[i])
        max_stress = max(max_stress, stress[i])
    avgs = list()
    if k >= 5:
        avgs = [[stress[i]] for i in stress] # include all the students
    else:
        avgs = [[stress[i]] for i in stress if stress[i] > 0.0] # remove the lowest students
        avgs = [i for i in avgs if i[0] < max_stress] # remove the highest students
        
    # kmeans clustering
    kmeans = KMeans(n_clusters = k, random_state=0).fit(avgs)
    centers = kmeans.cluster_centers_
    print('(average stress) centers are: ')
    print(centers)

    # build group dictionary
    groups = dict()
    for student in student_list:
        belongs = kmeans.predict([[stress[student]]])[0]
        groups[student] = 'group_'+str(belongs)
    return groups

# time warping clustering
def time_warping(student_list, data, feature, k):
    '''
    @param student_list: list of student id, in the form (string)student_id
    @param data: actual data, dict: (string)keys -> data
    @param feature: {-1: stress label, 0-5: corresponding feature}
    @param k: number of centers for kmeans clustering
    '''
    month_days = {0: 0, 1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31} # map: month -> # days
    # TODO (yunfeiluo)
    student_key = dict() # map: student -> [(key, time)]
    for key in data:
        curr = key.split('_')
        time = sum([month_days[i] for i in range(int(curr[1]))]) + int(curr[2]) # month plus day
        try:
            student_key[curr[0]].append((key, time))
        except:
            student_key[curr[0]] = [(key, time)]
    for student in student_key:
        student_key[student] = sorted(student_key[student], key=lambda x:x[1])
    
    # formulate data
    pts = list()
    if feature == -1:
        for student in student_key: # [time, label]
            pt = np.array([[i[1], int(i[0].split('_')[-1])] for i in student_key[student]])
            student_key[student] = pt
            pts.append(pt)
    pts = np.array(pts)
    
    plt.plot([i[0] for i in pts[0]], [i[1] for i in pts[0]])
    for i in range(1, 3):
        plt.plot([i[0] for i in pts[i]], [i[1] for i in pts[i]])
    plt.show()
    
    # dtw clustering
    print('fitting...')
    dtw_clusters = DTW_clusters(n_clusters = k, random_state=0, tol=10 ** -10).fit(pts)
    centers = dtw_clusters.cluster_centers_

    # build group dictionary
    print('predicting...')
    groups = dict()
    for student in student_key:
        belongs = dtw_clusters.predict([student_key[student]])[0]
        groups[student] = 'group_'+str(belongs)
    return groups

# do clustering works
def clustering(student_list, data, method):
    '''
    @param student_list: list of student id, in the form (string)student_id
    @param data: the actual data of the students, with data_key->data
    @param method: string from command line argument(s), decide how to clustering
    '''
    # TODO *yunfeiluo) do the actual clustering work, write to pkl file
    groups = dict()
    if method == 'one_for_each':
        groups = one_for_each(student_list)
    elif method[:10] == 'avg_stress':
        groups = avg_stress_cluster(student_list=student_list, data=data, k=int(method.split('_')[-2]))
    elif method[:7] == 'surveys':
        features = ['avg_hours_slept', 'mode_sleep_rating', 'avg_dead_line_per_week']
        k = int(method.split('_')[-2])
        groups = kmeans_features(student_list, features, k)
    elif method [:3] == 'dtw':
        k = int(method.split('_')[-2])
        feature = -1 # stress label
        groups = time_warping(student_list, data, feature, k)
        print('This function haven\'t implemented...')
    else:
        groups = one_for_each(student_list)

    # write to pkl file
    filepath = 'Data/student_groups/' + method + '.pkl'
    print('write to the file: ' + filepath)
    write_utils.data_structure_to_pickle(groups, filepath)

if __name__ == '__main__':
    # ##### Pickle #####
    print(os.path.abspath('Data/training_data/shuffled_splits/training_date_normalized_shuffled_splits_select_features_no_prev_stress_all_students.pkl'))
    data_file_path = 'Data/training_data/shuffled_splits/training_date_normalized_shuffled_splits_select_features_no_prev_stress_all_students.pkl'
    data = read_pickle(data_file_path)
    student_list = conversions.extract_distinct_student_idsfrom_keys(data['data'].keys())
    student_list = conversions.prepend_ids_with_string(student_list, "student_")
    
    method = None
    try:
        method = sys.argv[1]
    except:
        method = 'one_for_each'
    student_groups = clustering(student_list, data['data'], method)

    groups_file_path = 'Data/student_groups/' + method + '.pkl'
    print('get group file from: ' + groups_file_path)
    student_groups = read_pickle(groups_file_path) 
    
    # check how students are distributed
    print("student distribution: ")
    rev_groups = dict()
    for student in student_groups:
        if rev_groups.get(student_groups[student]) != None:
            rev_groups[student_groups[student]].append(student)
        else:
            rev_groups[student_groups[student]] = [student]
    for group in rev_groups:
        print(group + ': ' + str(rev_groups[group]))
    