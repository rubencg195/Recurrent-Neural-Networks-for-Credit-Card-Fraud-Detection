## Metrics
import joblib
import time
import copy
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold , StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

## Models
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
# from tensorflow.keras.backend.tensorflow_backend import set_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, PrecisionAtRecall
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Dropout, Flatten, Activation, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping,  ModelCheckpoint, CSVLogger, TensorBoard, ProgbarLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l1

class MLModel():
  def __init__(self):
    pass
  def visualize_data(self, data):
    pass
  def preprocess(self, data):
    X, y = None, None
    return X, y
  def create_model(self):
    pass
  def train(self, X, y):
    pass

class RNNModel(MLModel):
    def __init__(
      # GRID SEARCH & KERAS CLASSIFIER EXAMPLE https://www.kaggle.com/shujunge/gridsearchcv-with-keras
      self, 
      input_shape,
      output_dim,
      param_grid,
      scoring=None, refit=True, verbose=2,
      output_file="RNN_best_model.h5",
      early_stopping_monitor='val_loss',
      model_checkpoint_monitor='val_loss',
    ):
        self.input_shape = input_shape
        self.output_dim  = output_dim
        self.output_file = output_file
        self.callbacks   =  [ 
           EarlyStopping(monitor=early_stopping_monitor, mode='auto', verbose=verbose, patience=3) , 
           ModelCheckpoint( 
               self.output_file , monitor=model_checkpoint_monitor, 
               mode='auto', save_best_only=True, verbose=verbose
           ),
           CSVLogger(self.output_file+'_log.txt', append=True),
           TensorBoard(
                log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
                update_freq='epoch', profile_batch=2, embeddings_freq=0,
                embeddings_metadata=None
            ), 
            ProgbarLogger()
        ]
        print("\n\n{} {} {}\n\n".format( 10*"_ " , "CREATING RNN MODEL WITHOUT L1 REGULARIZATION" , 10*"_ "))
        
        
        print("\n\n{} {} {}\n\n".format( 10*"_ " , "INITIALIZING GRID SEARCH RNN MODEL" , 10*"_ "))
        # CALLBACKS EXPLANATION  https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
        self.model = GridSearchCV(
            estimator  = KerasClassifier( build_fn = self.create_model,  verbose=verbose),
            param_grid = param_grid,
            # scoring    = 'accuracy' , #['accuracy', 'precision'], 
            # refit      = 'precision',                  # For multiple metric evaluation, this needs to be a string denoting the scorer that would be used to find the best parameters for refitting the estimator at the end.
            n_jobs     = 1,#-1,                           # -1 means using all processors.
            pre_dispatch = "1*n_jobs",
            cv         = 10, 
            return_train_score = True,
            scoring=scoring, 
            refit=refit, 
            verbose=verbose,
        )
        print("""
        PARAMETERS:
        ________________________________
        input_shape :  {}
        output_dim  :  {}
        main scoring:  {}
        all scoring :  {}
        early_stopping_monitor   : {}
        model_checkpoint_monitor : {}
        verbose: {}
        callbacks: \n\n{}\n\n
        """.format( 
            input_shape, 
            output_dim , 
            refit , 
            scoring, 
            early_stopping_monitor, 
            model_checkpoint_monitor, 
            verbose,
            self.callbacks
        ))
        
        for k in param_grid: print( "{} : {}".format(k, param_grid[k] ) )
        print("\n\n")
    def create_model(
        self, hidden_layers, hidden_layers_neurons, loss, optimizer, 
        rnn_hidden_layers, rnn_hidden_layers_neurons, modelType="LSTM", 
        dropout=True, dropout_rate=0.2, output_layer_activation="relu", 
        rnn_layer_activation="relu", hidden_layer_activation="relu",  
    ):
        keras_eval_metric = [
            [
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn'), 
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
            ]
        ]
        return_sequences= True if rnn_hidden_layers > 0  else False
        print("\n\n{} {} {}\n\n".format( 10*"_ " , "CREATING ML MODEL" , 10*"_ "))
        print("""
        PARAMETERS:
        ________________________________ 
          rnn_hidden_layers:         {} 
          rnn_hidden_layers_neurons: {} 
          hidden_layers:             {} 
          hidden_layers_neurons:     {}
          loss:                      {}
          optimizer:                 {}
          modelType:                 {}
          dropout:                   {}
          dropout_rate:              {}
          input_shape:               {}
          output_dim:                {}
          output_layer_activation:   {}
          rnn_layer_activation:      {}
          hidden_layer_activation:   {}
          keras_eval_metric:         {}
          return_sequences:          {}
          callbacks:                 {}
          \n""".format(
            rnn_hidden_layers, rnn_hidden_layers_neurons, hidden_layers, hidden_layers_neurons, loss, optimizer, 
            modelType, dropout, dropout_rate, self.input_shape, self.output_dim, output_layer_activation,
            rnn_layer_activation, hidden_layer_activation, 
            keras_eval_metric, return_sequences, self.callbacks
        ))
        model = Sequential()
        #INPUT DIM EXPLANATION https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        
        
        if(modelType == "LSTM"):
            model.add(LSTM(units=rnn_hidden_layers_neurons , input_shape=self.input_shape, activation=rnn_layer_activation, return_sequences=return_sequences   ))
        elif(modelType == "GRU"):
            model.add(GRU( units=rnn_hidden_layers_neurons , input_shape=self.input_shape, activation=rnn_layer_activation, return_sequences=return_sequences ))
        elif(modelType == "SimpleRNN"):
            model.add(SimpleRNN( units=rnn_hidden_layers_neurons , input_shape=self.input_shape, activation=rnn_layer_activation, return_sequences=return_sequences   ))
        
        for i in range(rnn_hidden_layers):
            if(modelType == "LSTM"):
                model.add(LSTM(units=rnn_hidden_layers_neurons , activation=rnn_layer_activation  ))
            elif(modelType == "GRU"):
                model.add(GRU( units=rnn_hidden_layers_neurons , activation=rnn_layer_activation  ))
        
        for i in range(hidden_layers):
            #model.add(Dense(hidden_layers_neurons, kernel_regularizer=l1(0.01), bias_regularizer=l1(0.01) ))
            model.add(Dense(hidden_layers_neurons))
            model.add(Activation(hidden_layer_activation))
        if(dropout):
            model.add(Dropout(dropout_rate))
        model.add(Dense(self.output_dim )) # model.add(Dense(1, activation='sigmoid' )) 'softmax' 
        model.add(Activation(output_layer_activation))
        model.compile(loss=loss, optimizer=optimizer, 
                      metrics=keras_eval_metric 
                     ) #, metrics=['accuracy']
        print("\n\nMODEL SUMMARY: \n\n", model.summary())
        self.modelType = modelType
        self.model = model
        return model
    def train(self, X, y, X_test, y_test, class_weights=None):
        print("\n\n{} {} {}\n\n".format( 10*"_ " , "TRAINING RNN" , 10*"_ "))
        start=time.time()
        # ClASS WEIGHTS COMPUTATION https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
        if(class_weights is None):
            print("Generating Class Weights.")
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y.flatten()), y.flatten())
        else:
            print("Using Given Class Weights.", class_weights)
        print("""
        Class weights: \n{}\n{}\n
        for classes: \n{}\n
        # Frauds: {}
        # of Non-Frauds: {}
        """.format( 
            class_weights, 
            dict(enumerate(class_weights)),
            np.unique(y.flatten()), 
            np.count_nonzero(y == 1), 
            np.count_nonzero(y == 0),
        ))
        print("""INPUTS
        X:      {}
        y:      {}
        X_test: {}
        y_test: {}
        """.format(X.shape, y.shape, 
                   X_test.shape if X_test != None else None, 
                   y_test.shape  if X_test != None else None
        ))
        self.history = None
        if( X_test == None ):
            print("TRAINING WITH ONLY TRAINING DATA")
            self.history = self.model.fit(
              X, y, 
#               validation_data= (X_test, y_test),
              class_weight   = class_weights,
              callbacks      = self.callbacks ,
#               steps_per_epoch= X.shape[0],
#               epochs = 10,
#               validation_data=val_data_gen,
#               validation_steps=total_val
            )
        else:
            print("TRAINING WITH TEST & TRAINING DATA")
            self.history = self.model.fit(
              X, y, 
              validation_data= (X_test, y_test),
#               steps_per_epoch= X.shape[0],
#             validation_steps=X_test.shape[0]
              class_weight   = class_weights,
              callbacks      = self.callbacks   ,
#               epochs = 10,
            )
        print("\n\n{} {} {}\n\n".format( 10*"_ " , "RNN TRAINING RESULTS" , 10*"_ "))
        print("""
          BEST ESTIMATOR:          {} 
          BEST SCORE:              {}
          BEST PARAMS:             {}
          BEST INDEX IN CV SEARCH: {}
          SCORER FUNCTIONS:        {}
          \n
          HISTORY OBJ:             {}        
        \n\n""".format( 
          self.history.best_estimator_,
          self.history.best_score_ , 
          self.history.best_params_ ,
          self.history.best_index_,
          self.history.scorer_,
          self.history
        ))
        print("cv_results_dict: ")
        print(pd.DataFrame( self.history.cv_results_ ))
        # for params, mean_score, scores in self.history.cv_results_:
        # for params, mean_score, scores in self.history.grid_scores_:
          # print("\tMean: {}. Std: {}. Params: {}".format(scores.mean(), scores.std(), params))
        print("Total time: {:0.2f}  seconds or {:0.2f} minutes. Saving model to: {}".format(  time.time()-start , (time.time()-start)/60, self.output_file ))
        return self.history