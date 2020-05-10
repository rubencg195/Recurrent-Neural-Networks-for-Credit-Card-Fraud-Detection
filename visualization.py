#Math & Visualization
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
sns.set()

def plot_roc_auc(y_test, preds):
    '''
    Takes actual and predicted(probabilities) as input and plots the Receiver
    Operating Characteristic (ROC) curve
    ''' 
    fig = plt.figure()
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = np.round(auc(fpr, tpr), 2)
#     print("fpr: ", fpr)
#     print("tpr: ", tpr)
    print("ROC AUC: ", roc_auc)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.draw()
#     plt.savefig(  'ROCAUC.png' )

def pr_curve(y_test, y_pred):
    fig = plt.figure()
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    average_precision = np.round( average_precision_score(y_test, y_pred), 2)
    pr_auc = np.round( auc(recall, precision), 2)
    plt.title('PR CURVE')
#     print("PREC: ", precision)
#     print("REC: ", recall)
    print("PR-RC AUC: ", pr_auc)
    print("average_precision: ", average_precision)
    plt.plot(recall, precision, 'b', label = 'Ave PRE = %0.2f' % average_precision)
    plt.legend(loc = 'lower right')
#     plt.plot([0, 1], [0, 1],'r--')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.draw()
#     plt.savefig( 'PRCURVE.png' )
#     print("P: {}\nR: {}\nTHRES: {}".format(precision, recall, thresholds))

def acc_plot(acc, val_acc):
    fig = plt.figure()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.draw()
#     plt.savefig( 'BEST MODEL ACC' )
    
def loss_plot(loss, val_loss):
    fig = plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.draw()
#     plt.savefig( 'BEST MODEL LOSS' )

def printModelData(cv_results_df):
    for i, cvr in enumerate(cv_results_df.iterrows()):
        print("""{} MODEL # {} {}\n
            PARAM EPOCHS    :  {} HIDDEN LAYERS  :  {}   NEURONS PER HIDDEN LAYER:  {} 
            LOSS FUNCTION   : "{}" MODEL TYPE     : "{}" OPTIMIZER        : "{}"
            STD FIT TIME    :  {} MEAN SCORE TIME:  {} STD SCORE TIME   :   {} 
            MEAN TEST SCORE :  {} STD TEST SCORE :  {} RANK TEST SCORE  :   {} 
            MEAN TRAIN SCORE:  {} STD TRAIN SCORE:  {} 
            PARAMS: {}      
        """.format(
            10*" _" , i+1, 10*" _", 
            cv_results_df.at[i, "param_epochs"]     , cv_results_df.at[i, "param_hidden_layers"]  , cv_results_df.at[i, "param_hidden_layers_neurons"], cv_results_df.at[i, "param_rnn_hidden_layers"] ,  cv_results_df.at[i, "param_rnn_hidden_layers_neurons"] , 
            cv_results_df.at[i, "param_loss"]       , cv_results_df.at[i, "param_modelType"]      , cv_results_df.at[i, "param_optimizer"]            ,
            cv_results_df.at[i, "std_fit_time"]     , cv_results_df.at[i, "mean_score_time"]      , cv_results_df.at[i, "std_score_time"]             ,                                                                                                                      
    #         cv_results_df.at[i, "mean_test_score"]  , cv_results_df.at[i, "std_test_score"]       , cv_results_df.at[i, "rank_test_score"]            ,
            cv_results_df.at[i, "mean_test_accuracy"]  , cv_results_df.at[i, "std_test_accuracy"]       , cv_results_df.at[i, "rank_test_accuracy"]            ,
            cv_results_df.at[i, "mean_train_accuracy"] , cv_results_df.at[i, "std_train_accuracy"]      ,
    #         cv_results_df.at[i, "mean_train_score"] , cv_results_df.at[i, "std_train_score"]      ,
            cv_results_df.at[i, "params"]                     
        ))
    #     train_score_p_split = [ cv_results_df.at[i,"split{}_train_score".format(j)] for j in range(10) ]
    #     test_score_p_split  = [ cv_results_df.at[i,"split{}_test_score".format(j)] for j in range(10) ]
        train_score_p_split = [ cv_results_df.at[i,"split{}_train_accuracy".format(j)] for j in range(10) ]
        test_score_p_split  = [ cv_results_df.at[i,"split{}_test_accuracy".format(j)] for j in range(10) ]

        train_pre_p_split = [ cv_results_df.at[i,"split{}_train_precision".format(j)] for j in range(10) ]
        test_pre_p_split  = [ cv_results_df.at[i,"split{}_test_precision".format(j)] for j in range(10) ]

        train_rec_p_split = [ cv_results_df.at[i,"split{}_train_recall".format(j)] for j in range(10) ]
        test_rec_p_split  = [ cv_results_df.at[i,"split{}_test_recall".format(j)] for j in range(10) ]

        train_roc_auc_p_split = [ cv_results_df.at[i,"split{}_train_roc_auc".format(j)] for j in range(10) ]
        test_roc_auc_p_split  = [ cv_results_df.at[i,"split{}_test_roc_auc".format(j)] for j in range(10) ]

        train_average_precision_p_split = [ cv_results_df.at[i,"split{}_train_average_precision".format(j)] for j in range(10) ]
        test_average_precision_p_split  = [ cv_results_df.at[i,"split{}_test_average_precision".format(j)] for j in range(10) ]

        train_f1_p_split = [ cv_results_df.at[i,"split{}_train_f1".format(j)] for j in range(10) ]
        test_f1_p_split  = [ cv_results_df.at[i,"split{}_test_f1".format(j)] for j in range(10) ]

        index_titles = ["TRAIN", "TEST"]

        print("""\nACC PERFORMANCE PER SPLIT \n\n{}
            \n\n""".format( pd.DataFrame([ np.round(train_score_p_split, 3) , np.round(test_score_p_split, 3) ], 
                columns=[ "SPLIT#{}".format(se ) for se in range(10)],
                index=index_titles
        )))

        print("""\nPREC PERFORMANCE PER SPLIT \n\n{}
            \n\n""".format( pd.DataFrame([ np.round(train_pre_p_split, 3) , np.round(test_pre_p_split, 3) ], 
                columns=[ "SPLIT#{}".format(se ) for se in range(10)],
                index=index_titles
        )))

        print("""\nREC PERFORMANCE PER SPLIT \n\n{}
            \n\n""".format( pd.DataFrame([ np.round(train_rec_p_split, 3) , np.round(test_rec_p_split, 3) ], 
                columns=[ "SPLIT#{}".format(se ) for se in range(10)],
                index=index_titles
        )))

        print("""\nROC AUC PERFORMANCE PER SPLIT \n\n{}
            \n\n""".format( pd.DataFrame([ np.round(train_roc_auc_p_split, 3) , np.round(test_roc_auc_p_split, 3) ], 
                columns=[ "SPLIT#{}".format(se ) for se in range(10)],
                index=index_titles
        )))

        print("""\nAVE PRE PERFORMANCE PER SPLIT \n\n{}
            \n\n""".format( pd.DataFrame([ np.round(train_average_precision_p_split, 3) , np.round(test_average_precision_p_split, 3) ], 
                columns=[ "SPLIT#{}".format(se ) for se in range(10)],
                index=index_titles
        )))

        print("""\nF1 PERFORMANCE PER SPLIT \n\n{}
            \n\n""".format( pd.DataFrame([ np.round(train_f1_p_split, 3) , np.round(test_f1_p_split, 3) ], 
                columns=[ "SPLIT#{}".format(se ) for se in range(10)],
                index=index_titles
        )))


    #     plt.rcParams["figure.figsize"] = [16,9]
    #     plt.figure(figsize=(100,40))
    #     plt.rcParams.update({'font.size': 130})
        fig = plt.figure()
        plt.plot(train_score_p_split)
        plt.plot(test_score_p_split)
        plt.title('model accuracy')
        plt.text(1, 1, str(cv_results_df.at[i, "params"]), fontsize=8, style='oblique', ha='center',
             va='top', wrap=True)
        print( str(cv_results_df.at[i, "params"])  )
        plt.ylabel('accuracy')
        plt.xlabel('split')
        plt.legend(['train', 'test'], loc='upper left')
        plt.draw()
        img_name = "Model#{}_acc.png".format(i+1)
        print("\n\nSaving image with name: ",  img_name )
        plt.savefig( img_name)
        print("\n\n")

        fig = plt.figure()
        plt.plot(train_rec_p_split)
        plt.plot(train_rec_p_split)
        plt.title('model precision')
        plt.ylabel('precicion')
        plt.xlabel('split')
        plt.legend(['train', 'test'], loc='upper left')
        plt.draw()
        img_name = "Model#{}_prec.png".format(i+1)
        print("\n\nSaving image with name: ",  img_name )
        plt.savefig( img_name)

        fig = plt.figure()
        plt.plot(train_rec_p_split)
        plt.plot(test_rec_p_split)
        plt.title('model recall')
        plt.ylabel('recall')
        plt.xlabel('split')
        plt.legend(['train', 'test'], loc='upper left')
        plt.draw()
        img_name = "Model#{}_rec.png".format(i+1)
        print("\n\nSaving image with name: ",  img_name )
        plt.savefig( img_name)

        fig = plt.figure()
        plt.plot(train_roc_auc_p_split)
        plt.plot(test_roc_auc_p_split)
        plt.title('model auc')
        plt.ylabel('auc')
        plt.xlabel('split')
        plt.legend(['train', 'test'], loc='upper left')
        plt.draw()
        img_name = "Model#{}_auc.png".format(i+1)
        print("\n\nSaving image with name: ",  img_name )
        plt.savefig( img_name)

def print_confusion_matrix(tn, fp, fn, tp):
    print("""
            PREDICTED CLASSES
            POSITIVE   | NEGATIVE
    _____________________________
    ACTUAL   |         |
    POSITIVE | TP: {}   FN: {}
    _____________________________
    ACTUAL   |         |
    NEGATIVE | FP: {}   TN: {}
    """.format(tp, fn, fp, tn ))
    

def format_vertical_headers(df):
    """Display a dataframe with vertical column headers"""
    styles = [dict(selector="th", props=[('width', '5px')]),
              dict(selector="th.col_heading",
                   props=[("writing-mode", "vertical-rl"),
                          ('transform', 'rotateZ(-5deg)'), 
                          ('height', '30px'),
                          ('width', '10px'),
                          ('vertical-align', 'top')])]
    return (df.fillna(0).round(3).style.set_table_styles(styles))

def visualize_data(data):
  print("\n\n{} {} {}\n\n".format( 10*"_ " , "PIE CHART - FRAUD VS NON-FRAUD" , 10*"_ "))
  df_fraud= data[data['fraud']==1]
  num_transaction_total, num_transaction_fraud = len(data), len(df_fraud)
  num_transaction_total, num_transaction_fraud
  print("Total Transactions: {} \nTotal Fraud Transactions: {}".format(num_transaction_total, num_transaction_fraud) )
  percent_fraud = round((num_transaction_fraud / num_transaction_total) * 100, 2)
  percent_safe = 100 - percent_fraud
  percent_fraud, percent_safe
  print("% Safe Transactions: {} \n% Fraud Transactions: {}\n\n".format(percent_safe, percent_fraud) ) # plotting pie chart for percentage comparision: 'fraud' vs 'safe-transaction'
  fig1, ax1 = plt.subplots()
  plt.title("Figure 1. Fraud vs Safe Transaction in Percentage", fontsize = 20)
  labels = ['Fraud', 'Safe Transaction']
  sizes = [percent_fraud, percent_safe]
  explode = (0, 0.7)  # only "explode" the 2nd slice (i.e. 'Hogs')
  patches, texts, autotexts = ax1.pie(sizes,  labels=labels, autopct='%1.1f%%', shadow = True, explode=explode, startangle=130, colors = ['#ff6666', '#2d64bc'])
  texts[0].set_fontsize(30)
  texts[1].set_fontsize(18)
  matplotlib.rcParams['text.color'] = 'black'
  matplotlib.rcParams["font.size"] = 30
  plt.rcParams["figure.figsize"] = [6, 6]
  plt.show()
  plt.savefig('PIE_CHART_FRAUD_VS_NONFRAUD_visualize_data.png')

  print("\n\n{} {} {}\n\n".format( 10*"_ " , "COLUMN INFORMATION & PREVIEW" , 10*"_ "))
  # Extracting # of unique entires per column and their sample values
  num_unique = []
  sample_col_values = []
  for col in data.columns:
      num_unique.append(len(data[col].unique()))  # Counting number of unique values per each column
      sample_col_values.append(data[col].unique()[:3])  # taking 3 sample values from each column   
  # combining the sample values into a a=single string (commas-seperated)
  # ex)  from ['hi', 'hello', 'bye']  to   'hi, hello, bye'
  col_combined_entries = []
  for col_entries in sample_col_values:
      entry_string = ""
      for entry in col_entries:
          entry_string = entry_string + str(entry) + ', '
      col_combined_entries.append(entry_string[:-2])
  # Generating a list 'param_nature' that distinguishes features and targets
  param_nature = []
  for col in data.columns:
      if col == 'fraud':
          param_nature.append('Target')
      else:
          param_nature.append('Feature')
  # Generating Table1. Parameters Overview
  df_feature_overview = pd.DataFrame(np.transpose([param_nature, num_unique, col_combined_entries]), index = data.columns, columns = ['Parameter Nature', '# of Unique Entries', 'Sample Entries (First three values)'])
  print("\nTotal # of Values: {} \nShape: {} \n\n".format(len(data), data.shape))
  print(df_feature_overview)

  print("\n\n{} {} {}\n\n".format( 10*"_ " , "FRAUD VS NON-FRAUD AVE. AMOUNT & PERCENTAGE" , 10*"_ "))
  df_fraud     = data.loc[data.fraud == 1] 
  df_non_fraud = data.loc[data.fraud == 0]
  fraud_ave_amount_col          = df_fraud.groupby('category')['amount'].mean().round(2)
  non_fraud_ave_amount_col      = df_non_fraud.groupby('category')['amount'].mean().round(2)
  percentage_fraud_per_category = data.groupby('category')['fraud'].mean().round(3)*100
  amount_percentage_table       = pd.concat(
      [ fraud_ave_amount_col , non_fraud_ave_amount_col, percentage_fraud_per_category ],
      keys=["Fraudulent Ave. Amount","Non-Fraudulent Ave. Amount","Fraud Percent(%)"],
      axis=1, 
      sort=False
  ).sort_values(by=['Non-Fraudulent Ave. Amount'])
  print(amount_percentage_table)
  num_bins = 15                 # Number of sections where data will be segmented to be shown as a bar in the histogram. For example: The first bin is called "0~500"
  tran_amount = data['amount']
  n, bins, patches = plt.hist(tran_amount, num_bins, density = False, stacked = True, facecolor= '#f26a6a', alpha=0.5)
  plt.close()
  n_fraud = np.zeros(num_bins)
  for i in range(num_bins):
      for j in range(num_transaction_fraud):
          if bins[i] < df_fraud['amount'].iloc[j] <= bins[i+1]:  #??????
              n_fraud[i] += 1
  range_amount = []
  for i in range(num_bins):
      lower_lim, higher_lim = str(int(bins[i])), str(int(bins[i+1]))
      range_amount.append("$ " + lower_lim + " ~ " + higher_lim )
  df_hist = pd.DataFrame(index = range_amount)
  df_hist.index.name = 'Transaction Amount[$]'
  df_hist['# Total'] = n
  df_hist['# Fraud'] = n_fraud
  df_hist['# Safe'] = df_hist['# Total'] - df_hist['# Fraud']
  df_hist['% Fraud'] = (df_hist['# Fraud'] / df_hist['# Total'] * 100).round(2)
  df_hist['% Safe'] = (df_hist['# Safe'] / df_hist['# Total'] * 100).round(2)
  print(df_hist)

  print("\n\n{} {} {}\n\n".format( 10*"_ " , "FREQUENCY DISTRIBUTION OF TRANSACTION AMOUNTS" , 10*"_ "))  
  fig3 = plt.figure(figsize=(16,6))
  # Generating stacked bar-chart
  bars_fraud = plt.bar(range(num_bins), df_hist['# Safe'], width = 0.5, color = '#00e64d')
  bars_safe = plt.bar(range(num_bins), df_hist['# Fraud'], width = 0.5, bottom = df_hist['# Safe'], color='#ff6666')
  # Labeling
  plt.title("Figure 3. Frequency Distribution of Transaction Amount", fontsize = 20)
  plt.xticks(range(num_bins), range_amount, fontsize = 14)
  plt.yticks(fontsize = 14)
  plt.legend((bars_fraud[0], bars_safe[0]), ('Safe Transaction', 'Fraud'), loc=4, fontsize = 16)
  plt.xlabel('Ranges of Transaction Amount', fontsize=16)
  plt.ylabel('Number of Occurence', fontsize=16)
  # hiding top/right border
  ax = plt.gca()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  x = plt.gca().xaxis
  # rotate the tick labels for the x axis
  for item in x.get_ticklabels():
      item.set_rotation(50)
  plt.show()
  plt.savefig('FREQUENCY_DISTRIBUTION_OF_TRANSACTION_AMOUNTS_visualize_data.png')

  print("\n\n{} {} {}\n\n".format( 10*"_ " , "FRAUD PERCENTAGE AT DIFFERENT RANGES OF TRANSACTION AMOUNT" , 10*"_ "))  
  fig4 = plt.figure(figsize=(16,6))
  # Generating stacked bar-chart
  bars_fraud = plt.bar(range(num_bins), df_hist['% Safe'], width = 0.5, color = '#00e64d')
  bars_safe = plt.bar(range(num_bins), df_hist['% Fraud'], width = 0.5, bottom = df_hist['% Safe'], color='#ff6666')
  # Labeling
  plt.title("Figure 4. Fraud Percentage at Different Ranges of Transaction Amount", fontsize = 20)
  plt.xticks(range(num_bins), range_amount, fontsize = 14)
  plt.yticks(fontsize = 14)
  plt.legend((bars_fraud[0], bars_safe[0]), ('Safe Transaction', 'Fraud'), loc=4, fontsize = 16)
  plt.xlabel('Ranges of Transaction Amount', fontsize=16)
  plt.ylabel('Percentage', fontsize=16)
  plt.ylim(0, 100)
  x = plt.gca().xaxis
  # rotate the tick labels for the x axis
  for item in x.get_ticklabels():
      item.set_rotation(85)
  # hiding top/right border
  ax = plt.gca()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)    
  # bar-value display
  for bar in bars_safe:
      plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, str(int(bar.get_height())) + '%', 
                  ha='center', color='w', fontsize=13, rotation = 'vertical', weight = 'bold')
  plt.gca().text(bars_fraud[0].get_x() + bars_fraud[0].get_width()/2, bars_fraud[0].get_height() - 5, str(int(bars_fraud[0].get_height())) + '%', 
                  ha='center', color='black', fontsize=13, rotation = 'vertical', weight = 'bold')
  plt.show()
  plt.savefig('FRAUD_PERCENTAGE_AT_DIFFERENT_RANGES_OF_TRANSACTION_AMOUNT_visualize_data.png')

  print("\n\n{} {} {}\n\n".format( 10*"_ " , "FRAUD VS NON-FRAUD HISTOGRAM" , 10*"_ "))  
  plt.figure(figsize=(30,10))
  sns.boxplot(x=data['category'],y=data.amount)
  plt.title("Boxplot for the Amount spend in category")
  plt.ylim(0,4000)
  plt.legend()
  plt.show()
  plt.savefig('FRAUD_VS_NONFRAUD_HISTOGRAM_visualize_data.png')

  print("\n\n{} {} {}\n\n".format( 10*"_ " , "FRAUD DISTRIBUTION PER AGE" , 10*"_ "))  
  age_comparison_df = df_fraud.groupby('age')['fraud'].agg(['count']).reset_index().rename(columns={'age':'Age','count' : '# of Fraud'}).sort_values(by='# of Fraud')
  age_df = pd.DataFrame([ ["'0'", "<=18"], ["'1'", "19-25"], ["'2'", "26-35"], ["'3'", "36-45"], ["'4'", "46-55"], ["'5'", "56-65"], ["'6'", ">65"], ["'U'", "Unknown"], ], columns=["Age", "Label"])
  age_comparison_df = pd.merge(age_comparison_df, age_df, on="Age", how="outer")
  age_comparison_df['Age'] = age_comparison_df['Age'].map(lambda x: x.strip("'"))
  age_comparison_df = age_comparison_df.sort_index(by=["Age"])
  age_comparison_df = age_comparison_df[["Age", "Label", '# of Fraud']]
  print(age_comparison_df)
    
    
