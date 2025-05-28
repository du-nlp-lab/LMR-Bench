import json
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
#from huggingface_hub import notebook_login
#notebook_login()
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm


def probing(rp_log_data_list, rp_question_data_list, labels_log, labels_question, tot_layer):
    list_acc = []
    list_f1 = []
    list_auc = []
    # print("Evaluating ...")
    for i in range(tot_layer):
        #print("i",i)
        #rp_log_data_var_name = f'rp_log_data_{i}'
        #rp_question_data_var_name = f'rp_question_data_{i}'
        #rp_log_data_i = globals()[rp_log_data_var_name]
        #rp_question_data_i = globals()[rp_question_data_var_name]
        rp_log_data_i = rp_log_data_list[i]
        rp_question_data_i = rp_question_data_list[i]
    
        X = np.concatenate((rp_log_data_i, rp_question_data_i), axis=0)
        y = np.concatenate((labels_log, labels_question), axis=0)
        
        # Merge data and labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
        # Initialize a random classifier, like Random Forest.
        classifier = LogisticRegression(penalty='l2')
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        # print("------------------------------------")
        # print("This is epoch",i)
        # print(f'Accuracy: {accuracy}')
    
        # Compute and print F1
        # 'binary' for bi-classification problems; 'micro', 'macro' or 'weighted' for multi-classification problems
        f1 = f1_score(y_test, y_pred, average='binary')  
        # print(f'F1 Score: {f1}')
    
        # Predict the probability of test dataset. (For ROC AUC, we need probabilities instead of label)
        y_prob = classifier.predict_proba(X_test)[:, 1]  # supposed + class is 1.
    
        # Calc and print ROC, AUC
        roc_auc = roc_auc_score(y_test, y_prob)
        # print(f'ROC AUC Score: {roc_auc}')
    
        list_acc.append(accuracy)
        list_f1.append(f1)
        list_auc.append(roc_auc)
    return list_acc, list_f1, list_auc
