from models import RNNModel
import numpy as np
import pandas as pd
import joblib
from data import readLocally
from sklearn.metrics import confusion_matrix
from visualization import plot_roc_auc, pr_curve, format_vertical_headers, print_confusion_matrix, printModelData
from sklearn.utils import class_weight
from tensorflow.keras.models import load_model
from visualization import plot_roc_auc, pr_curve, print_confusion_matrix
from sklearn.metrics import confusion_matrix
from visualization import plot_pr_curves, plot_roc_curves

out_dir = "data/generated/"
import joblib


def load_train_data(one_batch=False):
    X_train = joblib.load( out_dir+"X_train_1B.data" )  if one_batch else joblib.load( out_dir+"X_train.data" ) 
    y_train = joblib.load( out_dir+"y_train.data" )
    print("""SHAPES & KEYS:
    X_train          : {} P: {:.2f} N: {:.2f}
    y_train          : {}
    ________________________
    """.format(
        X_train.shape,  
        len(y_train[y_train==1])*100/len(y_train),
        len(y_train[y_train==0])*100/len(y_train),
        y_train.shape,
    ))
    return X_train, y_train

def test_models():
    print("LOADING RF")
    rf  = joblib.load(out_dir+"rf.model")
    print("LOADING XGB")
    xgb = joblib.load(out_dir+"xgboost.model")
    print("LOADING GRU")
    gru_1_b = load_model(
        out_dir+"gru_paysim_2_rnn_layers_no_l1.h5")
    print("LOAD DATA")
    X_test  = joblib.load( out_dir+"X_test.data"  )
    y_test  = joblib.load( out_dir+"y_test.data"  )
    print(X_test.shape, y_test.shape)

    X_test_t =  X_test[:, -1]
    X_test_1_b = X_test_t.reshape(
        X_test_t.shape[0], 1, X_test_t.shape[1])
    models = [rf, xgb, gru_1_b]
    data   = [X_test_t, X_test_t, X_test_1_b]
    names  = ["RF", "XGB", "GRU"]
    
    pr_rc_data_val = np.array([]).reshape((0,3))
    
    
    for i, m in enumerate(models):
        X = data[i]
        print("EVALUATING ON : ", X.shape, y_test.shape)
        y_pred = m.predict_proba(X)
        print("PRED SHAPE: ", 
              y_pred.shape, y_pred.shape[1] > 1)
        y_score = y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred
        tn, fp, fn, tp = confusion_matrix(
            y_test, 
            y_score.round()
        ).ravel()
        print_confusion_matrix(tn, fp, fn, tp)
        pr_rc_data_val = np.vstack(
            pr_rc_data_val, 
            [y_test, y_score, names[i]],
        )
        
    plot_pr_curves(pr_rc_data_val=pr_rc_data_val, fig_path=out_dir+"PR.png")
#     plot_roc_curves(pr_rc_data_val)
        
        


def train_ensemble(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils import class_weight
    import time
    X_train = X_train[:, -1]
    print("X_train: ", X_train.shape)
    print(pd.DataFrame(X_train).head())
    
    class_weights = class_weight.compute_class_weight(
        'balanced', 
        np.unique(y_train.flatten()), y_train.flatten())
    clf = RandomForestClassifier(
        random_state=0, 
        class_weight = dict(enumerate(class_weights)),
        verbose=1)
    clf.fit(X_train, y_train)
    
    start=time.time()
    import joblib
    print("SAVING RF TO ", out_dir+"rf.model")
    joblib.dump(clf, out_dir+"rf.model")
    print("Total time: {:0.2f}  sec or {:0.2f} min.".format( 
        time.time()-start , 
        (time.time()-start)/60
    ))

    start=time.time()
    from xgboost import XGBClassifier
    xgb_clf = XGBClassifier( verbosity=1, random_state=0)
    xgb_clf.fit(X_train, y_train)
    print("SAVING XGB TO ", out_dir+"xgboost.model")
    joblib.dump(xgb_clf, out_dir+"xgboost.model")
    print("Total time: {:0.2f}  sec or {:0.2f} min.".format( 
        time.time()-start , 
        (time.time()-start)/60
    ))
    

def train_rnn(
    X_train, y_train,
    out_dir = "data/generated/",
    checkpoint_dir = out_dir+"gru_paysim_2_rnn_layers_no_l1_checkpoints.h5",
    model_dir = out_dir+"gru_paysim_2_rnn_layers_no_l1.h5"
):
    gru_param_grid = {
        'modelType': ['GRU'], 
        'dropout': [True],
        'dropout_rate': [0.2], 
        'epochs': [4], 
        'hidden_layer_activation': ['sigmoid'], 
        'hidden_layers':  [2], 
        'hidden_layers_neurons':[300], 
        'loss': ['binary_crossentropy'], 
        'optimizer': ['adam'], 
        'output_layer_activation': ['sigmoid'], 
        'rnn_hidden_layers':  [0], 
        'rnn_hidden_layers_neurons':[50], 
        'rnn_layer_activation': ['sigmoid']
    }

    # X_train = X_train[:,-1].reshape( len(X_train), 1, 12 )
    # # X_test  = X_test[:,  -1].reshape( len(X_test), 1, 12 )
    minimize_factor = 0.2
    X_train = X_train[:int(len(X_train)*minimize_factor)]
    y_train = y_train[:int(len(y_train)*minimize_factor)]

    X_train[X_train >=  1E308] = 0
    X_train[X_train <= -1E308] = 0


    print("""SHAPES & KEYS:
    X_train          : {} P: {:.2f} N: {:.2f}
    y_train          : {}
    ________________________
    """.format(
        X_train.shape,  
        len(y_train[y_train==1])*100/len(y_train),
        len(y_train[y_train==0])*100/len(y_train),
        y_train.shape,
    ))



    print(
        X_train.shape,
    #     X_test.shape
    )
    # joblib.dump(X_train, out_dir+"X_train_1B.data" )

    n_batches        = X_train.shape[0]
    batch_size       = X_train.shape[1]
    n_features       = X_train.shape[2]

    

    print("INPUT : ")
    print(n_batches, batch_size, n_features)
    
    print("SAVING TO : ")
    print(checkpoint_dir)
    print(model_dir)

    print("SAMPLES: \n X \n\n {} \n\n y {}\n".format(X_train[:5], y_train[:5]))

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train.flatten()), y_train.flatten())
    class_weights = dict(enumerate(class_weights))
    print("WEIGHTS", class_weights)
    gru_model = RNNModel(
      input_shape=( batch_size , n_features  ),
      output_dim = 1,
      param_grid=gru_param_grid,
      scoring=['accuracy', 'precision', 'recall', 'roc_auc', 'f1', 'average_precision' ],  
      refit= "precision", #"recall",   
      verbose=2, #1,
      output_file= checkpoint_dir,
      early_stopping_monitor="precision",#"val_recall",
      model_checkpoint_monitor="precision",#"val_recall",
    )
    gru_history = gru_model.train( 
        X_train, y_train, 
        None, None, #X_test, y_test, 
    #     class_weights=class_weights 
    )

    print("SAVING..")
    gru_model.model.best_estimator_.model.save( model_dir )



    # gru_model.model = gru_model.create_model(**gru_param_grid)


    # print("CCALLBACKS", gru_model.callbacks )
    # print(gru_model.model.summary)

    # # self.history = gru_model.model.fit(
    #     X_train, y_train, 
    # #     validation_data= (X_test, y_test),
    #     class_weight   = dict(enumerate(class_weights)),
    #     callbacks      = gru_model.callbacks   
    # )

    
    
    
    
    
    
    
    
    
    
    
    
    
# train_ensemble(X_train, y_train)
# test_models()

X_train, y_train = load_train_data(one_batch=False)
train_rnn(
    X_train, y_train,
    out_dir = out_dir,
    checkpoint_dir = out_dir+"gru_paysim_1_rnn_layers_3d_b_no_l1_DD_checkpoints.h5",
    model_dir = out_dir+"gru_paysim_1_rnn_layers_3d_b_no_l1_DD.h5"
)