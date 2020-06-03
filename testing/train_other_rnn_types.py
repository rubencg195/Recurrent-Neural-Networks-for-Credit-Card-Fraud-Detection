from models import RNNModel
import numpy as np
import pandas as pd
import joblib
from data import readLocally
from sklearn.metrics import confusion_matrix
from visualization import plot_roc_auc, pr_curve, format_vertical_headers, print_confusion_matrix, printModelData

X_train, y_train, X_test, y_test, X_val, y_val, labels_hash, scaler = readLocally()

#DIVIDE X_TEST IN HALF AND ADD IT TO X_TRAIN
test_half_pos   = int(len(X_test)/2)
total_data_size = len(X_train) + len(X_test) + len(X_val)

X_train = np.vstack( ( X_train ,  X_test[0: test_half_pos ] ))
y_train = np.append( y_train ,  y_test[0: test_half_pos ]  )

X_test  = X_test[ test_half_pos : ]
y_test  = y_test[ test_half_pos : ]

print("""INCREASED TRAIN SET -> NEW SIZES
X_train: {}
y_train: {}         {:0.0f}% {:0.2f}P% {:0.2f}N%
X_test : {}
y_test : {}         {:0.0f}% {:0.2f}P% {:0.2f}N%
X_val  : {}
y_val  : {}         {:0.0f}% {:0.2f}P% {:0.2f}N%
""".format(
    X_train.shape, y_train.shape, len(X_train) * 100 / total_data_size,  len(y_train[y_train==1]) * 100 / len(y_train) , len(y_train[y_train==0]) * 100 / len(y_train) ,
    X_test.shape, y_test.shape,   len(X_test)  * 100  / total_data_size, len(y_test[y_test==1])   * 100 / len(y_test)  , len(y_test[y_test==0])   * 100 / len(y_test) ,
    X_val.shape,  y_val.shape,    len(X_val)   * 100  / total_data_size, len(y_val[y_val==1])     * 100 / len(y_val)   , len(y_val[y_val==0])     * 100 / len(y_val) ,
))

#Reduce Data to half the size per batch

X_train1 = X_train[:,12:]
X_test1 = X_test[:,  12:]
X_val1 = X_val[:,  12: ]

X_train2 = X_train[:,-1].reshape( len(X_train), 1, 12 )
X_test2  = X_test[:,  -1].reshape( len(X_test), 1, 12 )
X_val2   = X_val[:,  -1].reshape( len(X_val), 1, 12 )

print("X_train 1: {}->{}".format(X_train.shape, X_train1.shape))
print("X_test  1: {}->{}".format(X_test.shape,  X_test1.shape))
print("X_val   1: {}->{}\n\n".format(X_val.shape,   X_val1.shape))

print("X_train 2: {}->{}".format(X_train.shape, X_train2.shape))
print("X_test  2: {}->{}".format(X_test.shape,  X_test2.shape))
print("X_val   2: {}->{}\n\n".format(X_val.shape,   X_val2.shape))

print("X_train 1 Sample: \n\n")
print(pd.DataFrame(X_train1[-1]))

print("X_train 2 Sample: \n\n")
print(pd.DataFrame(X_train2[-1]))

gru_param_grid = {
    'modelType': ['GRU'], 
    'dropout': [True],
    'dropout_rate': [0.2], 
    'epochs': [50], 
    'hidden_layer_activation': ['sigmoid'], 
    'hidden_layers': [2], 
    'hidden_layers_neurons': [300], 
    'loss': ['binary_crossentropy'], 
    'optimizer': ['adam'], 
    'output_layer_activation': ['sigmoid'], 
    'rnn_hidden_layers': [0], 
    'rnn_hidden_layers_neurons': [50], 
    'rnn_layer_activation': ['sigmoid']
}

# n_batches        = X_train.shape[0]
# batch_size       = X_train.shape[1]
# n_features       = X_train.shape[2]
# print(n_batches, batch_size, n_features)

# gru_param_grid_4 = gru_param_grid.copy()
# gru_param_grid_4['rnn_hidden_layers']= [1]

# print(gru_param_grid_4)

# gru_model_7 = RNNModel(
#   input_shape=( batch_size , n_features  ),
#   output_dim = 1,
#   param_grid=gru_param_grid_4,
#   scoring=['accuracy', 'precision', 'recall', 'roc_auc', 'f1', 'average_precision' ],  
#   refit= "recall",   
#   verbose=2,
#   output_file= "gru_double_data_2_rnn_layers_one_batch_1_layer_no_l1_checkpoints.h5",
#   early_stopping_monitor="val_recall",
#   model_checkpoint_monitor="val_recall"
# )

# gru_history_7 = gru_model_7.train( X_train, y_train, X_test, y_test, class_weights=None )
# print("SAVING gru_double_data_2_rnn_layers_one_batch_1_layer_no_l1.h5")
# gru_model_7.model.best_estimator_.model.save( "gru_double_data_2_rnn_layers_one_batch_1_layer_no_l1.h5" )

n_batches        = X_train2.shape[0]
batch_size       = X_train2.shape[1]
n_features       = X_train2.shape[2]
print(n_batches, batch_size, n_features)


from imblearn.over_sampling import SMOTE 

X_train_2D = X_train2.reshape(X_train2.shape[0], X_train2.shape[2])
print('Original dataset shape X {} => {}, y {}'.format( X_train2.shape, X_train_2D.shape, y_train.shape ))
print('P {} N {}'.format( len(y_train[y_train == 1]), len(y_train[y_train == 0]) ))
sm = SMOTE(random_state=42)
X_train_ov_2D, y_train_ov = sm.fit_resample(X_train_2D, y_train)
X_train_ov = X_train_ov_2D.reshape(X_train_ov_2D.shape[0], 1, X_train_ov_2D.shape[1])

print('Resampled dataset shape ',X_train_ov.shape,  X_train_ov_2D.shape, y_train_ov.shape, )
print('P {} N {}'.format( len(y_train_ov[y_train_ov == 1]), len(y_train_ov[y_train_ov == 0]) ))

gru_model_8 = RNNModel(
  input_shape=( batch_size , n_features  ),
  output_dim = 1,
  param_grid=gru_param_grid,
  scoring=['accuracy', 'precision', 'recall', 'f1', 'average_precision' ],  
  refit= "recall",   
  verbose=1,
  output_file= "gru_oversampled_data_one_batch_no_l1_checkpoints.h5",
  early_stopping_monitor="val_recall",
  model_checkpoint_monitor="val_recall"
)


gru_history_8 = gru_model_8.train( X_train_ov, y_train_ov, X_test2, y_test, class_weights=None )
print("SAVING..")
gru_model_8.model.best_estimator_.model.save( "gru_oversampled_data_one_batch_no_l1.h5" )