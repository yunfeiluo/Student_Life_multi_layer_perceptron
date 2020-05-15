# Machine Learning Research Project: Grouplized Student Stress Prediction  
Title: Student_Life_multi_layer_perceptron  
Open source project, master branch origined from: https://github.com/Information-Fusion-Lab-Umass/MultiRes 
  
Works:  
a) contribution to the Machine Learning Research Project on the dataset: student_life  
b) Implementation of functions for the building and training model, clustering, and model evaluation  

# Code backup (Functionalities):

#   Multi-Layer_Perceptron (MLP) for each group of students.
#   Clustering students
a) K-Means based on average stress  
b) K-Means based on features from surveys  
c) Clustering based on DTW on Time-Series data  
#   Model Evaluation: 
a) K-Fold Cross Validation, Stratified by students, groups, student-id_stress-label, and group-id_stress_label  
b) leave one out validation, subject to students and groups  

# Dependencies 
python packet: Pytorch, tqdm, pandas, sklearn, pickle, numpy
