#!/usr/bin/env python
# coding: utf-8
download_data   = False    # Download from Kaggle
generate_data   = False     # Preprocess Data. If it is false it reads the data locally.
read_from_cloud = False    # Download Preprocessed Data From Cloud. If it is false it reads the files locally.
save_to_cloud   = False    # Save to cloud. 
bucket_address  = "s3://verafin-mitacs-ruben-chevez/"
project_folder  = "customer_batches"
model_name      = "checkpoint_model.h5"
project_path    = bucket_address + project_folder 
empty_padding_value           = -1
reduce_data_for_testing       = True
reduce_data_for_testing_value = 10000

#General
import json
import zipfile
import os
import subprocess
import math
import time
import progressbar
import pickle
import joblib
import s3fs
import copy
import traceback
from pathlib import Path

#Math & Visualization
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
import sklearn
from sklearn.metrics import confusion_matrix
sns.set()

import warnings
warnings.filterwarnings("ignore")

from utils import setupTF, runCommand, runCommands, downloadFromKaggle, testTry 
from models import MLModel, RNNModel
from data import downloadFromKaggle, normalizing_data, generating3DRNNInput, generateNewFeatures, separateInBatches, separateLabel, separatingTrainTest, normalize3DInput, read_data, readLocally, saveLocally, readDataFromCloud, saveToCloud  
from visualization import plot_roc_auc, pr_curve, print_confusion_matrix, visualize_data, printModelData, acc_plot, loss_plot, format_vertical_headers

print("Versions")
print("Tensorflow : ", tf.__version__)
print("Pandas     : ", pd.__version__)
print("Numpy      : ", pd.__version__) 
print("Sklearn    : ", sklearn.__version__) 

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

data, X_train, X_test, y_train, y_test, labels_hash = None, None, None, None, None, None
print("""
    DATA PREPROCESSING
    DOWNLOAD FROM KAGGLE: {}
    GENERATE DATA:        {}
    READ FROM CLOUD:      {}
    SAVE TO CLOUD:        {}
""".format(download_data, generate_data, read_from_cloud, save_to_cloud))

if(download_data):
    downloadFromKaggle(
        api_token = {"username":"rubencg195","key":"1a0667935c03c900bf8cc3b4538fa671"},
        kaggle_file_path='/home/ec2-user/.kaggle/kaggle.json',
        zip_file_path = "banksim1.zip"
    )
    
data = read_data(input_file_path="bs140513_032310.csv")

if(generate_data):
    print("\n\n{} {} {}\n\n".format( 10*"_ " , " GENERATE DATA " , 10*"_ "))
    visualize_data(data)
    labels_hash                        = normalizing_data(data)
    rnn_data, smaller_batches_rnn_data = generating3DRNNInput(data) 
    if(reduce_data_for_testing):
        print("REDUCE DATA FOR TESTING. DATA REDUCED FROM {} TO {}".format(rnn_data.shape[0], reduce_data_for_testing_value))
        rnn_data = rnn_data[:reduce_data_for_testing_value]
    rnn_mod_data                       = generateNewFeatures(rnn_data)
    X, grouped_X, y, grouped_y         = separateInBatches(rnn_mod_data, min_batch_size=25)
    X_norm                             = normalize3DInput(X)
    #Change all padding values to 0 in the labels
    print("# of padding values in labels :", len(y[y==empty_padding_value]))
    y[y==empty_padding_value] = 0
    X_train, X_test, y_train, y_test, X_val, y_val = separatingTrainTest(X_norm, y)

    saveLocally(
        rnn_data, rnn_mod_data, 
        X_train, y_train, X_test, 
        y_test, X_val, y_val, labels_hash
    )

    runCommands(["ls *.png", "ls *.data", "ls *.h5"])
    print("""SHAPES & KEYS:
    X_train          : {}
    y_train          : {}
    ________________________
    X_test           : {}
    y_test           : {}
    ________________________
    X_val            : {}
    y_val            : {}
    ________________________
    labels_hash Keys : {}
    """.format(
        X_train.shape, y_train.shape,
        X_test.shape,  y_test.shape,
        X_val.shape, y_val.shape, 
        labels_hash.keys() 
    ))
    
    columns=[
        "day", "age", "gender", "merchant", "category", "amount", 
        "curr_day_tr_n","ave_tr_p_day_amount", "tot_ave_tr_amount", "is_mer_new","com_tr_type", "com_mer",
        "fraud"
    ]
    excel_filename = 'data.xlsx'
    print("SAVING DATA TO EXCEL: ", excel_filename)
    writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
    bar    = progressbar.ProgressBar(max_value=len(rnn_mod_data))
    for i,b in enumerate(rnn_mod_data):
        pd.DataFrame(b, columns=columns).to_excel(writer, sheet_name='cust_{}'.format(i))
        bar.update(i+1)
    writer.save()

#     excel_filename = 'rnn_data.xlsx'
#     print("SAVING 3D NORM DATA TO EXCEL: ", excel_filename)
#     writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
#     bar    = progressbar.ProgressBar(max_value=len(X))
#     for i,b in enumerate(X):
#         pd.DataFrame(b, columns=columns[:-1]).to_excel(writer, sheet_name='i_{}_lbl_{}'.format(i, y[i]))
#         bar.update(i+1)
#     writer.save()

else:
    if(read_from_cloud):
        X_train, X_test, y_train, y_test, labels_hash = readDataFromCloud()
    else:
        X_train, y_train,  X_test, y_test, X_val, y_val, labels_hash, scaler  = readLocally()
        
    if(reduce_data_for_testing):
        print("REDUCE TRAIN DATA FOR TESTING. DATA REDUCED FROM {} TO {}".format(X_train.shape[0], reduce_data_for_testing_value))
        X_train = X_train[:reduce_data_for_testing_value]
        X_test  = X_test[:reduce_data_for_testing_value]
        X_val   = X_val[:reduce_data_for_testing_value]
        y_train = y_train[:reduce_data_for_testing_value]
        y_test  = y_test[:reduce_data_for_testing_value]
        y_val   = y_val[:reduce_data_for_testing_value]


# ## Model Setup

# In[235]:


#ONLY FOR TF 1.15
# session = tf.keras.backend.get_session()
# init = tf.global_variables_initializer()
# session.run(init)

param_grid = {
    "rnn_hidden_layers"         : [0, 1], 
    "rnn_hidden_layers_neurons" : [50, 100],  #[25, 50, 100], 
    "hidden_layers"             : [2],  #[0, 1, 2], 
    "hidden_layers_neurons"     : [200, 300],  #[25, 50, 100], 
    "loss"                      : ['binary_crossentropy'], # ["mse"], 
    "optimizer"                 : ['adam'], #[tf.keras.optimizers.SGD(lr=0.01)], #['adam'],
    "modelType"                 : ['LSTM', 'GRU'],
    "epochs"                    : [50], #[25, 50], # [1],
    "output_layer_activation"   : ['sigmoid'], #['relu'] #['softmax']  # ['sigmoid']
    "rnn_layer_activation"      : ["sigmoid"], 
    "hidden_layer_activation"   : ["sigmoid"],
    "dropout"                   : [True],  
    "dropout_rate"              : [0.2]
}
if(reduce_data_for_testing):
    param_grid = {
        "rnn_hidden_layers"         : [0, 1], 
        "rnn_hidden_layers_neurons" : [50], 
        "hidden_layers"             : [2],  
        "hidden_layers_neurons"     : [50],
        "loss"                      : ['binary_crossentropy'], 
        "optimizer"                 : ['adam'],
        "modelType"                 : ['LSTM'],
        "epochs"                    : [1],
        "output_layer_activation"   : ['sigmoid'],
        "rnn_layer_activation"      : ["sigmoid"], 
        "hidden_layer_activation"   : ["sigmoid"],
        "dropout"                   : [True],
        "dropout_rate"              : [0.2]
    }

n_batches        = X_train.shape[0]
batch_size       = X_train.shape[1]
n_features       = X_train.shape[2]
# n_pred_per_batch = y_train.shape[1]
rnn = RNNModel(
  input_shape=( batch_size , n_features  ),
  output_dim = 1,
  param_grid=param_grid,
  scoring=['accuracy', 'precision', 'recall', 'roc_auc', 'f1', 'average_precision' ],             # scoring=None, 
  refit= "recall", #'accuracy',        #"precision" , "recall"  100 recall catch everything                                                                   #True, 
#   scoring=None, 
#   refit=True, 
  verbose=2,
  output_file= model_name,
  early_stopping_monitor="val_recall",#'val_loss',
  model_checkpoint_monitor="val_recall", #'val_accuracy',
)

history = rnn.train( X_train, y_train, X_test, y_test )



print("Saving best estimator at {} and weights at {}".format("rnn_model.h5", "rnn_model_weights.h5"))
print(rnn.model.best_estimator_.model)
rnn.model.best_estimator_.model.save_weights(
    "FINAL_WEIGHTS_"+model_name
)
rnn.model.best_estimator_.model.save(
    "FINAL_"+model_name
)


print(history.__dict__)


cv_results_df = pd.DataFrame(history.cv_results_).round(3)

try:
    cv_results_df.to_csv("cv_results.history")
    cv_results_df.to_csv("cv_results.csv")
except:
    print("Couldn't save cv_results.history")

printModelData(cv_results_df)

selected_epochs = history.cv_results_["params"][0]['epochs']
print("""\n\nBEST MODEL HISTORY PER EPOCH
SELECTED EPOCHS   : {}
PARAMS            : {} \n""".format( 
  selected_epochs,
  history.best_params_
))

index_names = [
    "accuracy", "val_accuracy",
    "loss", "val_loss", 
    "precision", "val_precision", 
    "recall", "val_recall", 
    #"roc_auc", "val_roc_auc", 
    #"f1", "val_f1", 
#     "average_precision", "average_precision"
]
index_titles = [
    "TRAIN ACC", "TEST ACC", 
    "TRAIN LOSS", "TEST LOSS",
    "precision", "val_precision", 
    "recall", "val_recall", 
    #"roc_auc", "val_roc_auc", 
    #"f1", "val_f1", 
#     "average_precision", "average_precision"
]
print( pd.DataFrame([ np.round( history.best_estimator_.model.history.history[iname] , 3) for iname in index_names], 
#   columns=[ "EPOCH#{}".format(se+1) for se in range(selected_epochs)],
  index=index_titles
), "\n")

acc_plot(
    history.best_estimator_.model.history.history['accuracy'], 
    history.best_estimator_.model.history.history['val_accuracy']
)
loss_plot(
    history.best_estimator_.model.history.history['loss'],
    history.best_estimator_.model.history.history['val_loss'] 
)

print(rnn.model.best_estimator_)
y_pred = rnn.model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print_confusion_matrix(tn, fp, fn, tp)
plot_roc_auc(y_test, y_pred)
pr_curve(y_test, y_pred)



historyDict = copy.copy(history.__dict__)
print(historyDict.keys())
historyDict.pop("best_estimator_")
historyDict.pop("estimator")   

print("Saving history")
joblib.dump( historyDict, "rnn.history")
print("History saved successfully.")

# print("Saving history per parts")
# for k in history.__dict__.keys():
#     try:
#         print("Saving ", k, "->", "{}_rnn.history".format(k), "\n",  
#         history.__dict__[k].keys() if type(history.__dict__[k]) == dict else history.__dict__[k])
#         joblib.dump(  history.__dict__[k], "{}_rnn.history".format(k) )
#     except Exception as e:
#         print("Error saving:", "{}.history".format(k) ,"-", repr(e))
#         try:
#             print("Saving:", "{}.history.txt".format(k)) 
#             text_file = open("{}.history.txt".format(k), "w")
#             text_file.write(str(history.__dict__[k]))
#             text_file.close()
#         except Exception as ee:
#             print("Error saving:", "{}.txt".format(k),"-", repr(ee))
# print("History parts saved successfully.")


print("BEST ESTIMATOR")
print(rnn.model.best_estimator_.model)
print(rnn.model.best_estimator_.model.summary())
print(rnn.model.best_estimator_.model.get_config())


loaded_rnn = load_model("FINAL_"+model_name)
print(loaded_rnn.summary())
print(loaded_rnn.predict(X_test))

plt.show()


