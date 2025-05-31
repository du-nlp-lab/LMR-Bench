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

    return list_acc, list_f1, list_auc
