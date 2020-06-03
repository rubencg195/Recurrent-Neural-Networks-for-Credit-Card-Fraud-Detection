import zipfile
import os
import subprocess
import math
import time
import pickle
import joblib
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import progressbar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
empty_padding_value = -1

def downloadFromKaggle(
    api_token = {"username":"rubencg195","key":"1a0667935c03c900bf8cc3b4538fa671"},
    kaggle_file_path='/home/ec2-user/.kaggle/kaggle.json',
    zip_file_path = "banksim1.zip"
    ):
    runCommands([
        "rm -rf "+str(Path(kaggle_file_path).parent),
        "mkdir -p "+str(Path(kaggle_file_path).parent)
    ])
    with open(kaggle_file_path, 'w+') as file:
        json.dump(api_token, file)
    runCommands([
        "chmod 600 "+kaggle_file_path,
        "kaggle datasets download -d ntnu-testimon/banksim1 --force"
    ])
    zip_ref = zipfile.ZipFile(zip_file_path, 'r')
    zip_ref.extractall()
    zip_ref.close()
    runCommand("ls *.csv")
    
    
def normalizing_data(data):
  # Generate Hash Maps to be able to convert from numerical to categorical later.
  print("\n\n{} {} {}\n\n".format( 10*"_ " , "CATEGORICAL VALUES TO NUMERICAL - HASHMAP GENERATION" , 10*"_ "))
  tmp_df = data[:]
  col_categorical = tmp_df.select_dtypes(include= ['object']).columns
  print( "Features Types: \n\n{}\n\n".format(tmp_df.dtypes) )
  print( "Categorical Features: {}\n\n".format(col_categorical) )
  print( "\nHash maps previews:\n" )
  labels_hash = dict()
  for col_name in col_categorical:                         
    tmp_df[col_name] = tmp_df[col_name].astype('category') 
    labels_hash[col_name] = pd.DataFrame(  zip( tmp_df[col_name].cat.codes, tmp_df[col_name] ) , columns=["Index", "Label"] ).drop_duplicates(subset=['Index'])
    print("{} {} {} \n {}".format(10*"_", col_name , 10*"_", labels_hash[col_name].head() ) )
  # Converting categorical entries to integers
  tmp_df[col_categorical] = tmp_df[col_categorical].apply(lambda x: x.cat.codes)
  # seperatign data columns and target columns
  col_names = tmp_df.columns.tolist()
  col_names_features = col_names[0:len(col_names)-1]
  col_name_label     = col_names[-1]
  # Declaring 'data-dataframe'  and 'target-dataframe'
  X = tmp_df[col_names_features]
  y = tmp_df[col_name_label]

  print("\n\n{} {} {}\n\n".format( 10*"_ " , "FEATURE IMPORTANCE" , 10*"_ "))
  dt = DecisionTreeClassifier()
  print("\n\n{}\n\n".format(dt))
  dt.fit(X, y)
  # sorted-feature importances from the preliminary decision tree 
  ds_fi = pd.Series(dt.feature_importances_ , index = col_names_features).sort_values(ascending= False)
  # plotting feature imporance bar-graph
  fig2 = plt.figure(figsize=(13,5))
  # Generating stacked bar-chart
  bars_ft = plt.bar(range(len(ds_fi)), ds_fi, width = .8, color = '#2d64bc')
  # Labeling
  ttl = plt.title("Figure 2. Feature Importances", fontsize = 20).set_position([0.45, 1.1])
  plt.xticks(range(len(ds_fi)), ds_fi.index, fontsize = 14)
  # plot-dejunking
  ax = plt.gca()
  ax.yaxis.set_visible(False) # hide entire y axis (both ticks and labels)
  ax.xaxis.set_ticks_position('none')  # hide only the ticks and keep the label
  plt.xticks(rotation='vertical')
  # hide the frame
  for spine in plt.gca().spines.values():
    spine.set_visible(False)
  # value displaying
  rects = ax.patches  
  labels = ds_fi.values.round(2)
  for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, label, ha='center', va='bottom', fontsize = 13)
  
  plt.show()
  plt.savefig('FEATURE_IMPORTANCE_normalizing_data.png')
  return labels_hash

def generating3DRNNInput(data, generate_with_max_size=True):
    tmp_df = data[:]
    col_categorical        = tmp_df.select_dtypes(include= ['object']).columns
    for col_name in col_categorical:                         #????
        tmp_df[col_name] = tmp_df[col_name].astype('category') 
    tmp_df[col_categorical]  = tmp_df[col_categorical].apply(lambda x: x.cat.codes)
    X = tmp_df.iloc[:, :-1]
    y = tmp_df.fraud
    print("\n\n{} {} {}\n\n".format( 10*"_ " , "VISUALIZATION BEFORE TRANSFORMATION" , 10*"_ "))
    print("Total Fraud vs Non-Fraud Transactions Count: \n\n{}\n".format(y.value_counts()))
    print("Number of customers: ",  X["customer"].nunique() )
    print("Ratio of positive frauds vs total dataset: {:0.2f}%".format( ( y[y==1].count() /len(X)) *100  ))
    # The RNNs requiere various a 3D input of groups of less of 300 samples per group for better performance. One option is to divide the datasets per day. 
    # Even by dividing per day, each day has more than 7K data points per data group.
    mean_samples_per_customer = X["customer"].value_counts().mean()
    max_samples_per_customer = X["customer"].value_counts().max()
    print("\nTransactions per customer.\n\tMean: {:0.1f}\n\tMax:  {:0.0f} \n\tNumber of Batches Using Max Amount as Fixed Size: {:0.1f} ~ {}\n\tNumber of Batches Using Mean Amount as Fixed Size: {:0.1f} ~ {}".format(
        mean_samples_per_customer, 
        max_samples_per_customer, 
        len(X) / max_samples_per_customer ,
        math.ceil(len(X) / max_samples_per_customer ),
        len(X) / mean_samples_per_customer ,
        math.ceil(len(X) / mean_samples_per_customer )
    ))
    mean_samples_per_day = X["step" ].value_counts().mean()
    max_samples_per_day = X["step"].value_counts().max()
    print("\n\nSamples per Step (day): \n\tMean: {:0.0f} \n\tMax: {} \n\tNumber of Batches Using Max Amount as Fixed Size: {:0.1f} ~ {}\n\tNumber of Batches Using Mean Amount as Fixed Size: {:0.1f} ~ {}".format(
        mean_samples_per_day, max_samples_per_day, 
        len(X) / max_samples_per_day ,
        math.ceil(len(X) / max_samples_per_day ),
        len(X) / mean_samples_per_day ,
        math.ceil(len(X) / mean_samples_per_day )
    ))

    print("\n\n{} {} {}\n\n".format( 10*"_ " , "GROUPING TRANSACTIONS BY CUSTOMER ID" , 10*"_ "))
    customer_batches = dict()
    count = 0
    for x in tmp_df.groupby(["customer"]):
        customer_batches[x[0]] = x[1] 
    min_trans_per_cust = np.min( [ customer_batches[i].shape[0] for i in customer_batches]  ) 
    mean_trans_per_cust = np.mean( [ customer_batches[i].shape[0] for i in customer_batches]  ) 
    max_trans_per_cust = np.max( [ customer_batches[i].shape[0] for i in customer_batches]  ) 
    n_features =  tmp_df.shape[1]
    print("LEN: ", len(customer_batches))
    print("# Feaures: ", n_features )
    print("MIN TRANS PER CUST: ", min_trans_per_cust )
    print("MEAN TRANS PER CUST: ", mean_trans_per_cust, " ~ ", math.ceil(mean_trans_per_cust) )
    print("MAX TRANS PER CUST: ", max_trans_per_cust)
    print("EXAMPLE: \n\n", customer_batches[0] )

    # See how many groups of customer transactions are above the average size of 145
    print("\n\nSCATTER PLOT SHOWING SIZES OF BATCHES GROUPED BY CUSTOMER ID: \n\n" )
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    ax.scatter(range( len(customer_batches) ), [ customer_batches[i].shape[0] for i in customer_batches] , color='r')
    ax.plot([0, len(customer_batches)], [145, 145], "--")
    ax.set_xlabel('Batch Index')
    ax.set_ylabel('Batch Size')
    ax.set_title('Customer Batches Sizes')
    plt.show()
    plt.savefig('SCATTER_PLOT_SHOWING_SIZES_OF_BATCHES_GROUPED_BY_CUSTOMER_ID_generating3DRNNInput.png')

    print("\n\nBOX PLOT SHOWING MEAN SIZE OF BATCHES GROUPED BY CUSTOMER ID: \n\n" )
    fig1, ax1 = plt.subplots()
    ax1.set_title('Batch Sizes')
    ax1.boxplot([ customer_batches[i].shape[0] for i in customer_batches],  vert=False)
    plt.show()
    plt.savefig('BOX_PLOT_SHOWING_SIZES_OF_BATCHES_GROUPED_BY_CUSTOMER_ID_generating3DRNNInput.png')

    
    np_customer_batches = list()
    total_empty_rows_added = 0
    np_customer_batches_3d = np.array([])
    
    if generate_with_max_size:
        print("\n\n{} {} {}\n\n".format( 10*"_ " , "GENERATING 3D INPUT WITH BATCHES OF SIZE "+str(max_trans_per_cust) , 10*"_ "))
        # Full 3D Array as an input for the LSTM
        #     empty_padding_value = -1
        bar = progressbar.ProgressBar(max_value=len(customer_batches))
        for k_i,  k in enumerate(customer_batches):
        # for k in customer_batches:
            empty_rows_to_add =  max_trans_per_cust - customer_batches[k].shape[0]
            z = np.full( ( empty_rows_to_add , n_features ), empty_padding_value )
            np_customer_batches.append( np.r_[ customer_batches[k].values , z] )
            total_empty_rows_added += empty_rows_to_add
            bar.update(k_i+1)
        np_customer_batches_3d = np.array(np_customer_batches)
        mean_frauds_per_batch       = np.mean( [ len(b[-1][b[-1] == 1 ]) for b in np_customer_batches_3d ]  )  
        percentage_frauds_per_batch = np.mean( [ len(b[-1][b[-1] == 1 ]) / max_trans_per_cust for b in np_customer_batches_3d ]  )  * 100
        print(
        """
        The batches are separated by customer id. To be able to use the batches as input for the RNN, 
        it needs to have a static size. That is why the batch size is defined by the max number of 
        transactions done by the customers ({}). If one of the customers have done less transactions,
        the rest of the empty space is filled with {} values. The final array size is {}.\n\n""".format(max_trans_per_cust, empty_padding_value, np_customer_batches_3d.shape ),
        "\nTotal Empty Rows Added: ", total_empty_rows_added, 
        "\nPercentage of Empty Rows Added Compared to Total # of Data Points: %", np.round(total_empty_rows_added / (max_trans_per_cust * len(np_customer_batches) ) * 100 , 2), 
        "\nNew Shape: ", np_customer_batches_3d.shape,
        "\nMean of frauds per batch: ", np.round(mean_frauds_per_batch, 5),
        "\nPercentage of frauds per batch: ", np.round(percentage_frauds_per_batch, 5),
        )

    np_customer_batches = list()
    np_left_over_transactions = np.empty(shape=[0, n_features])
    total_empty_rows_added = 0
    mean_trans_per_cust = math.ceil(mean_trans_per_cust)
    print("\n\n{} {} {}\n\n".format( 10*"_ " , "GENERATING 3D INPUT WITH BATCHES OF SIZE "+str(mean_trans_per_cust) , 10*"_ "))
    bar = progressbar.ProgressBar(max_value=len(customer_batches))
    for k_i,  k in enumerate(customer_batches):
        if( mean_trans_per_cust > customer_batches[k].shape[0] ):
            empty_rows_to_add =  mean_trans_per_cust - customer_batches[k].shape[0]
            z = np.full( ( empty_rows_to_add , n_features ) , empty_padding_value )
            np_customer_batches.append( np.r_[ customer_batches[k].values , z] )
            total_empty_rows_added += empty_rows_to_add
        else:
            np_customer_batches.append( np.array(customer_batches[k][0:mean_trans_per_cust].values) ) 
            np_left_over_transactions = np.r_[ np_left_over_transactions , customer_batches[k][mean_trans_per_cust:].values ]  #axis 0 to append vertically.
        bar.update(k_i+1)
    left_over_n_batches = math.ceil( len(np_left_over_transactions) / mean_trans_per_cust )
    left_over_z = np.full( ( (left_over_n_batches * mean_trans_per_cust ) - len(np_left_over_transactions)  , n_features ),  empty_padding_value )
    np_left_over_transactions    = np.r_[ np_left_over_transactions , left_over_z ] 
    np_left_over_transactions_3d = np.reshape(np_left_over_transactions, (left_over_n_batches, mean_trans_per_cust, n_features )  )
    np_shifted_customer_batches_3d       = np.r_[ np.array(np_customer_batches) , np_left_over_transactions_3d ] 
    total_empty_rows = total_empty_rows_added + len(left_over_z)
    mean_frauds_per_batch       = np.mean( [ len(b[-1][b[-1] == 1 ]) for b in np_shifted_customer_batches_3d ]  )  
    percentage_frauds_per_batch = np.mean( [ len(b[-1][b[-1] == 1 ]) / mean_trans_per_cust for b in np_shifted_customer_batches_3d ]  )  * 100
    print(
      """
      The batches are separated by customer id. To be able to use the batches as input for the RNN, 
      it needs to have a static size. That is why the batch size is defined by the average number of 
      transactions done by the customers ({}). If one of the customers have done less transactions,
      the rest of the empty space is filled with {} values. The final array size is {}. The difference
      between this new more compacted version than previous which uses the max amount of transactions
      per customers is that if a customer has more than the average number of transactions, these 
      transactions are saved in a separate array called left_overs. The left overs are then shaped as
      a 3D array and appended to the main array. The problem with this array which is more efficient in 
      space and has less empty rows is that the mayority of batches are arranged by customer ID but the
      last batches are in disorder, having transactions from many customers.\n\n""".format(mean_trans_per_cust, empty_padding_value, np_shifted_customer_batches_3d.shape ),
      "\nTotal Empty Rows Added: ", total_empty_rows_added, 
      "\n% Empty Rows Added: %", (total_empty_rows_added / (mean_trans_per_cust * len(np_customer_batches) ) * 100 ), 
      "\nNew Shape: ", np.array(np_customer_batches).shape,
      "\nLeft overs: ", np_left_over_transactions.shape,
      "\nLeft overs %: ", np.round( len(np_left_over_transactions)  / (mean_trans_per_cust * len(np_customer_batches) ) * 100, 1 ),
      "\nLeft overs new Shape: ", np_left_over_transactions_3d.shape,
      "\nLeft overs empty rows: ", len(left_over_z),
      "\nLeft overs empty rows percentage (%) over total dataset: ", np.round( len(left_over_z)  / (mean_trans_per_cust * len(np_customer_batches) ) * 100, 4 ),
      "\nTotal Empty # Rows and %: ", total_empty_rows, " - ", ( total_empty_rows / (mean_trans_per_cust * len(np_customer_batches) ) ) * 100,
      "\nMean of frauds per batch: ", mean_frauds_per_batch,
      "\nPercentage of frauds per batch: ", percentage_frauds_per_batch,
      "\nFinal 3D Array Shape (Customer Batches + Left overs): ", np_shifted_customer_batches_3d.shape,
    )
    return np_customer_batches_3d, np_shifted_customer_batches_3d


def generateNewFeatures(data):
    print("\n\n{} {} {}\n\n".format( 10*"_ " , "GENERATING NEW FEATURES" , 10*"_ "))
    print(
    """
        The following features will created using the original Data. 
        The data generated is calculated inside each batch or group
        or transactions grouped by custmer ID. Each calculation takes
        all the data points before the current transaction in which the
        loop index is currently located.

        \t Current day number of transactions  - "curr_day_tr_n"
        \t Average transaction amount per day  - "ave_tr_p_day_amount"
        \t Total average transaction amount 
        \t From the beggining to current time  - "tot_ave_tr_amount"
        \t Is the merchant new?                - "is_mer_new"
        \t What is the common transaction type - "com_tr_type"
        \t What is the common merchant ID      - "com_mer"
        \n\n""".format(),
    )
    # Customer ID column will be disposed. The ID is the same as the batch position in the array
    columns=[
        "day", "age", "gender", "merchant", "category", "amount", 
        "curr_day_tr_n","ave_tr_p_day_amount", "tot_ave_tr_amount", "is_mer_new","com_tr_type", "com_mer",
        "fraud"
  ]       #Original Cols : step ,customer , age, gender , merchant, category, amount , fraud 
    new_data = list()
    start_time = time.time()
    bar = progressbar.ProgressBar(max_value=len(data)) 
    print(30*"_ ", "\n\n")
    for b_i, b in enumerate( data ):
        new_batch = pd.DataFrame(columns=columns)
        for t_i, trans in enumerate( b ) :
            step_col, merchant_col, cat_col, amount_col, is_fraud_col = 0, 4, 5, 6, 7
            current_merchant = trans[merchant_col] # Merchant column
            current_day      = trans[step_col]     # Day(step) column
            current_cat      = trans[cat_col]     # Trans type column
            is_new_merchant           = 0 if len( b[ :t_i, merchant_col ][ b[ :t_i, merchant_col ] == trans[merchant_col] ] ) > 0 else 1
            ave_trans_amount          = np.around(np.mean( b[:t_i, amount_col ] ), 2 )
            most_common_trans_type    =  np.bincount( b[:t_i,  cat_col ][b[:t_i, cat_col ] > 0].astype(int) ).argmax()  if  len(np.bincount( b[:t_i, cat_col ][b[:t_i, cat_col ] > 0].astype(int) )) > 0 else -1
            most_common_merchant      =  np.bincount( b[:t_i,  merchant_col ][b[:t_i, merchant_col ] > 0].astype(int) ).argmax()  if  len(np.bincount( b[:t_i, merchant_col ][b[:t_i, merchant_col ] > 0].astype(int) )) > 0 else -1
            ave_n_trans_per_day       =  np.round(pd.DataFrame(b[:t_i, [step_col, amount_col]][ b[:t_i, step_col ] != -1 ], index=None, columns=None).groupby(by=0).mean().reset_index().values[:, 1].mean() , 2)
            n_trans_this_day          =  len( b[ :t_i + 1, step_col ][ b[ :t_i + 1, step_col ] == trans[step_col] ] )
            ave_amount_for_curr_trans_type =  np.around(np.mean( b[:t_i+1, amount_col ][ b[:t_i+1, cat_col ] == trans[cat_col] ] ), 2 )
            tr_data = {
            "day": current_day, "age": trans[2], "gender": trans[3], "merchant": current_merchant, "category": current_cat, "amount" : trans[amount_col], 
            "curr_day_tr_n" : n_trans_this_day ,"ave_tr_p_day_amount": ave_n_trans_per_day, "tot_ave_tr_amount": ave_trans_amount, "is_mer_new": is_new_merchant, "com_tr_type" : most_common_trans_type, "com_mer": most_common_merchant,
            "fraud" : trans[is_fraud_col]
            }
            new_batch = new_batch.append( tr_data , ignore_index=True)
            # print(new_batch)
            # break
        new_data.append(new_batch.fillna(empty_padding_value).values)
        bar.update(b_i)
    new_data = np.array(new_data)
    delta_time = time.time() - start_time
    print("--- {:0.2f} s seconds or {:0.2f} minutes ---".format(delta_time, delta_time/60 ))
    print(new_data.shape)
    print(new_data[0])
    return new_data


def separatePaySimInBatches(customer_batches, min_batch_size=15, generate_grouped_batches=False):
    #print(customer_batches.shape)
    #min_batch_size = np.min([ pd.DataFrame(cb)[ pd.DataFrame(cb)[0] != -1 ].shape[0] for cb in customer_batches ])
    #max_batch_size = np.max([ pd.DataFrame(cb)[ pd.DataFrame(cb)[0] != -1 ].shape[0] for cb in customer_batches ])
    #print("MIN ", min_batch_size, " MAX ", max_batch_size)
    #padding_value = 0 
    padding_value                = empty_padding_value
    column_to_check              = 0
    # new_customer_batches         = list()
    # new_labels                   = list()
    # new_customer_batches         = np.array([]).reshape((-1, 25, 12))
    data_batches  = customer_batches[:, :, :-1]
    label_batches = customer_batches[:, :, -1]
    
    new_customer_batches         = np.full(data_batches.shape[0] * data_batches.shape[1] , np.nan, dtype=object )
    new_labels                   = np.full(data_batches.shape[0] * data_batches.shape[1] , np.nan, dtype=object )
    new_grouped_customer_batches = list()
    new_grouped_customer_labels  = list()
    columns=[
        "day", "age", "gender", "merchant", "category", "amount", 
        "curr_day_tr_n","ave_tr_p_day_amount", "tot_ave_tr_amount", "is_mer_new","com_tr_type", "com_mer",
#         "fraud"
    ]  
    print("\n\n{} {} {}\n\n".format( 10*"_ " , "SEPARATING IN BATCHES OF "+str(min_batch_size) , 10*"_ "))
    print("""
    In the following procedure, the batches separaeted by customer and
    generated in the previous function are iterated. Each transaction
    that has been padded with value of '{}' will be deleted. new batches
    of size '{}' will be generated using sliding window through each 
    transaction in every customer batch. Padding will be added for batches 
    with less transactions than the min amount '{}'.
    Input Shape:                 {}
    Data Shape:                  {}
    Label Len:                   {}
    Padding Value:               {}
    Column to check for 
    padding values to delete     {}
    transaction in the new batch
    """.format(
        padding_value, min_batch_size, min_batch_size,
        customer_batches.shape, data_batches.shape, label_batches.shape,
        padding_value , column_to_check ))
    
    bar           = progressbar.ProgressBar(max_value=len(customer_batches))
    cb_count = 0
    skipped_rows = 0
    for cb_i, cb in enumerate(data_batches):
        cb_df = pd.DataFrame(cb)
#         cb_df = cb_df[cb_df[column_to_check] != padding_value ]
        # grouped_customer_batches = list()
        # grouped_customer_labels = list()
        # grouped_customer_batches = np.empty( len(cb_df), dtype=object )
        # grouped_customer_labels  = np.empty( len(cb_df), dtype=int )
        for i, trans in enumerate(cb_df.values):
            label      =  label_batches[cb_i, i]
#             print("{}. Transactions Shape \n\n{}\n\n labels shape {}".format( i, trans, label ))
            if(label == -1):
                skipped_rows +=1
                continue
            init_index           = 0 if i+1 <= min_batch_size else i+1 - min_batch_size
            trans_before_current = cb_df[ init_index:i+1 ]
            n_features           = cb_df.shape[1]
            empty_rows_to_add    = 0 if min_batch_size <= len(trans_before_current) else min_batch_size - len(trans_before_current)
            z = np.full( ( empty_rows_to_add , n_features ), padding_value )
            # new_batch = pd.DataFrame( np.r_[ z, trans_before_current.values ] )
            #print(i, " trans before ", len(trans_before_current)," empty_rows_to_add ", empty_rows_to_add, " original shape ", cb_df.shape, " new shape ",new_batch.shape )
            #print(new_batch.tail() ,  "\n____________" )
            # new_batch = np.array(new_batch.values)
            # new_customer_batches.append( new_batch )
            
            # new_customer_batches.append( np.r_[ z, trans_before_current.values ]  )
            batch =  np.r_[ z, trans_before_current.values ]  
            # print(new_customer_batches.shape, batch.shape, label)
            
            # new_customer_batches = np.vstack(( new_customer_batches , [batch] ) )
            # new_labels = np.append(new_labels, label )
            
            new_customer_batches[ (data_batches.shape[1] * cb_i) + i ] = batch
            new_labels[ (data_batches.shape[1] * cb_i) + i ]           = label
            
            # new_customer_batches[i] =  np.r_[ z, trans_before_current.values ] 
            # new_labels[i]            =  label
            
            # grouped_customer_batches.append(new_batch  )
            # grouped_customer_labels.append(label)
#         break
        # new_grouped_customer_batches.append( np.array(grouped_customer_batches ) )
        # new_grouped_customer_labels.append(  np.array(grouped_customer_labels  ) )
        #print(cb_df.shape, "\n", cb_df.tail(), "\n____________" )
        #break
        cb_count += 1
        bar.update(cb_count)
    # new_customer_batches         = np.array(new_customer_batches)
    # new_labels                   = np.array(new_labels)
    # new_grouped_customer_batches = np.array( new_grouped_customer_batches)
    # new_grouped_customer_labels  = np.array( new_grouped_customer_labels)    
    
    X         = new_customer_batches 
    y         = new_labels 
    # grouped_X = new_grouped_customer_batches
    # grouped_y = new_grouped_customer_labels
    
    grouped_X, grouped_y = np.array([]), np.array([]) 
    
    #REPLACE PADDING WITH 0
#     X[ X == padding_value ] = 0
#     y[ y == padding_value ] = 0
#     grouped_X[ grouped_X == padding_value ] = 0
#     grouped_y[ grouped_y == padding_value ] = 0
    
    # len_per_cust_group    = [len(gc) for gc in grouped_X ]
    # frauds_per_cust_group = [len(fgc[fgc==1]) for fgc in grouped_y ]
#     print("""
#     X Shape: {}
#     y Shape: {}
#     X Grouped/Cust Shape: ( {} , ~ MIN:{}|AVE:|{}|MAX:{} , {}  )
#     y Grouped/Cust Shape: ({}, ~ MIN:{}|AVE:|{}|MAX:{} )
    
#     # TRANSACTIONS GROUPS PER CUSTOMER
#     Min                      : {} 
#     Max                      : {} 
#     Ave                      : {}
#     Total Transaction Groups : {}
#     # Frauds & %             : {}  - {}%
#     # Non-Fraud & %          : {}  - {}%
#     % Frauds per customer    : %
#     Cust Id with most fraud  : 
#         ID  :  #-Frauds:  of #Trans  
#         Note: The id is not the original. It has to be transformed using the label_hash.
#     Skipped rows due to having all -1: {}
#     """.format(
#         X.shape,
#         y.shape,
#         len(grouped_X), 
#         min(len_per_cust_group), np.round(np.average(len_per_cust_group), 2), max(len_per_cust_group), n_features,
#         len(grouped_y), min(len_per_cust_group), np.round(np.average(len_per_cust_group), 2), max(len_per_cust_group), 
#         min(len_per_cust_group), 
#         max(len_per_cust_group), 
#         np.round(np.average(len_per_cust_group), 2),
#         len(X),
#         len(y[y==1]), np.round(len(y[y==1])*100/len(y), 2),
#         len(y[y==0]), np.round(len(y[y==0])*100/len(y), 2),
#         # np.round(np.average(frauds_per_cust_group), 2),
#         # np.argmax(frauds_per_cust_group), 
#         # frauds_per_cust_group[np.argmax(frauds_per_cust_group)], 
#         # len(grouped_y[np.argmax(frauds_per_cust_group)]), 
#         skipped_rows
#     ))
    
#     print("Tail Sample of X \n\n{}\n")
#     print(pd.DataFrame(X[0], 
# #                        columns=columns, index=None
#                       ).tail())
#     print("Sample of y \n\n{}\n".format( y[0] ))
    return X, grouped_X, y , grouped_y


def separateInBatches(customer_batches, min_batch_size=15):
    #print(customer_batches.shape)
    #min_batch_size = np.min([ pd.DataFrame(cb)[ pd.DataFrame(cb)[0] != -1 ].shape[0] for cb in customer_batches ])
    #max_batch_size = np.max([ pd.DataFrame(cb)[ pd.DataFrame(cb)[0] != -1 ].shape[0] for cb in customer_batches ])
    #print("MIN ", min_batch_size, " MAX ", max_batch_size)
    #padding_value = 0 
    padding_value                = empty_padding_value
    column_to_check              = 0
    new_customer_batches         = list()
    new_labels                   = list()
    new_grouped_customer_batches = list()
    new_grouped_customer_labels  = list()
    data_batches  = customer_batches[:, :, :-1]
    label_batches = customer_batches[:, :, -1]
    columns=[
        "day", "age", "gender", "merchant", "category", "amount", 
        "curr_day_tr_n","ave_tr_p_day_amount", "tot_ave_tr_amount", "is_mer_new","com_tr_type", "com_mer",
#         "fraud"
    ]  
    print("\n\n{} {} {}\n\n".format( 10*"_ " , "SEPARATING IN BATCHES OF "+str(min_batch_size) , 10*"_ "))
    print("""
    In the following procedure, the batches separaeted by customer and
    generated in the previous function are iterated. Each transaction
    that has been padded with value of '{}' will be deleted. new batches
    of size '{}' will be generated using sliding window through each 
    transaction in every customer batch. Padding will be added for batches 
    with less transactions than the min amount '{}'.
    Input Shape:                 {}
    Data Shape:                  {}
    Label Len:                   {}
    Padding Value:               {}
    Column to check for 
    padding values to delete     {}
    transaction in the new batch
    """.format(
        padding_value, min_batch_size, min_batch_size,
        customer_batches.shape, data_batches.shape, label_batches.shape,
        padding_value , column_to_check ))
    
    bar           = progressbar.ProgressBar(max_value=len(customer_batches))
    cb_count = 0
    skipped_rows = 0
    for cb_i, cb in enumerate(data_batches):
        cb_df = pd.DataFrame(cb)
#         cb_df = cb_df[cb_df[column_to_check] != padding_value ]
        grouped_customer_batches = list()
        grouped_customer_labels = list()
        for i, trans in enumerate(cb_df.values):
            label      =  label_batches[cb_i, i]
#             print("{}. Transactions Shape \n\n{}\n\n labels shape {}".format( i, trans, label ))
            if(label == -1):
                skipped_rows +=1
                continue
            init_index           = 0 if i+1 <= min_batch_size else i+1 - min_batch_size
            trans_before_current = cb_df[ init_index:i+1 ]
            n_features           = cb_df.shape[1]
            empty_rows_to_add    = 0 if min_batch_size <= len(trans_before_current) else min_batch_size - len(trans_before_current)
            z = np.full( ( empty_rows_to_add , n_features ), padding_value )
            new_batch = pd.DataFrame( np.r_[ z, trans_before_current.values ] )
            #print(i, " trans before ", len(trans_before_current)," empty_rows_to_add ", empty_rows_to_add, " original shape ", cb_df.shape, " new shape ",new_batch.shape )
            #print(new_batch.tail() ,  "\n____________" )
            new_batch = np.array(new_batch.values)
            new_customer_batches.append( new_batch )
            new_labels.append(label)
            grouped_customer_batches.append(new_batch  )
            grouped_customer_labels.append(label)
#         break
        new_grouped_customer_batches.append( np.array(grouped_customer_batches ) )
        new_grouped_customer_labels.append(  np.array(grouped_customer_labels  ) )
        #print(cb_df.shape, "\n", cb_df.tail(), "\n____________" )
        #break
        cb_count += 1
        bar.update(cb_count)
    new_customer_batches         = np.array(new_customer_batches)
    new_labels                   = np.array(new_labels)
    new_grouped_customer_batches = np.array( new_grouped_customer_batches)
    new_grouped_customer_labels  = np.array( new_grouped_customer_labels)    
    
    X         = new_customer_batches 
    grouped_X = new_grouped_customer_batches
    y         = new_labels 
    grouped_y = new_grouped_customer_labels
    
    #REPLACE PADDING WITH 0
#     X[ X == padding_value ] = 0
#     y[ y == padding_value ] = 0
#     grouped_X[ grouped_X == padding_value ] = 0
#     grouped_y[ grouped_y == padding_value ] = 0
    
    len_per_cust_group    = [len(gc) for gc in grouped_X ]
    frauds_per_cust_group = [len(fgc[fgc==1]) for fgc in grouped_y ]
    print("""
    X Shape: {}
    y Shape: {}
    X Grouped/Cust Shape: ( {} , ~ MIN:{}|AVE:|{}|MAX:{} , {}  )
    y Grouped/Cust Shape: ({}, ~ MIN:{}|AVE:|{}|MAX:{} )
    
    # TRANSACTIONS GROUPS PER CUSTOMER
    Min                      : {} 
    Max                      : {} 
    Ave                      : {}
    Total Transaction Groups : {}
    # Frauds & %             : {}  - {}%
    # Non-Fraud & %          : {}  - {}%
    % Frauds per customer    : {}%
    Cust Id with most fraud  : 
        ID  : {} #-Frauds: {} of #Trans {} 
        Note: The id is not the original. It has to be transformed using the label_hash.
    Skipped rows due to having all -1: {}
    """.format(
        X.shape,
        y.shape,
        len(grouped_X), min(len_per_cust_group), np.round(np.average(len_per_cust_group), 2), max(len_per_cust_group), n_features,
        len(grouped_y), min(len_per_cust_group), np.round(np.average(len_per_cust_group), 2), max(len_per_cust_group), 
        min(len_per_cust_group), 
        max(len_per_cust_group), 
        np.round(np.average(len_per_cust_group), 2),
        len(X),
        len(y[y==1]), np.round(len(y[y==1])*100/len(y), 2),
        len(y[y==0]), np.round(len(y[y==0])*100/len(y), 2),
        np.round(np.average(frauds_per_cust_group), 2),
        np.argmax(frauds_per_cust_group), frauds_per_cust_group[np.argmax(frauds_per_cust_group)], len(grouped_y[np.argmax(frauds_per_cust_group)]), 
        skipped_rows
    ))
    
    print("Tail Sample of X \n\n{}\n")
    print(pd.DataFrame(X[0], 
#                        columns=columns, index=None
                      ).tail())
    print("Sample of y \n\n{}\n".format( y[0] ))
    return X, grouped_X, y , grouped_y


def separateLabel(data):
    print("\n\n{} {} {}\n\n".format( 10*"_ " , "SEPARATING X & y FOR TRAINING" , 10*"_ "))
    X = data[:, :, 0:-1]
    y = data[:, :, -1]
    #Replacing -1 values to 0
#     y[ data[:, :, -1] == -1] = 0
    print("X Shape: {} Y Shape: {}".format(X.shape, y.shape))
    return X, y

def separatingTrainTest(X, y, test_size=0.2, val_size=0.2):
    print("\n\n{} {} {}\n\n".format( 10*"_ " , "SEPARATING TEST & TRAIN" , 10*"_ "))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size+val_size, 
        random_state=1,
        shuffle=True,
#         stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=val_size, 
        random_state=1,
        shuffle=True,
#         stratify=y
    )

    print("""
    X-TRAIN Shape: {}
    Y-TRAIN Shape: {} #-Frauds: {} #-Non-Frauds: {}
    X-TEST Shape:  {}
    Y-YEST Shape:  {} #-Frauds: {} #-Non-Frauds: {}
    Total-#-Frauds: {} Total-#-Non-Frauds: {}
    \n""".format(
      X_train.shape, 
      y_train.shape, np.count_nonzero( y_train == 1 ), np.count_nonzero( y_train == 0 ),
      X_test.shape, 
      y_test.shape,  np.count_nonzero( y_test  == 1 ), np.count_nonzero( y_test  == 0 ),
      np.count_nonzero( y  == 1 ), np.count_nonzero( y  == 0 )
    ))

    return X_train, X_test, y_train, y_test, X_val, y_val

def normalize3DInput(data, filename="scaler.data"):
    print("\n\n{} {} {}\n\n".format( 10*"_ " , "SEPARATING TEST & TRAIN" , 10*"_ "))
    n_batches        = data.shape[0]
    batch_size       = data.shape[1]
    n_features       = data.shape[2]
    tmp_data         = data.reshape( (n_batches * batch_size, n_features ) )
    print("Converting 3D to 2D for easy processing. Batch Sample: \n\n {} \n\n Original Array Shape: {}. Temporary array with shape: {}\n".format( data[0], data.shape, tmp_data.shape )) #(4112, 265, 12)

    min_max_scaler    = MinMaxScaler()
    data_norm         = min_max_scaler.fit_transform(tmp_data)                      # ROBUST SCALER ANOTHER OPTION
    scaler_max        = min_max_scaler.data_max_                      
    scaler_min        = min_max_scaler.data_min_                      
    scaler_scale      = min_max_scaler.scale_                      
    scaler_data_range = min_max_scaler.data_range_    
    scaler_params     = min_max_scaler.get_params(deep=True)
    data_norm         = data_norm.reshape( (n_batches, batch_size, n_features) )
    
    print("""
    SCALER INFORMATION
    MAX:    {}
    MIN:    {}
    SCALE:  {}
    RANGE:  {}
    PARAMS: {}
    Data Normalized and reshaped to a 3D array. 
    Current Shape: {} 
    Saving scaler to file: {}
    """.format( 
        scaler_max,                             
        scaler_min,                    
        scaler_scale,                        
        scaler_data_range,   
        scaler_params,
        data_norm.shape, 
        data_norm[0] ,
        filename
    )) #(4112, 265, 12)
    joblib.dump(min_max_scaler, filename)
    return data_norm


def read_data(input_file_path):
    print("\n\n{} {} {}\n\n".format( 10*"_ " , "IMPORT DATA FROM CSV" , 10*"_ "))
    data = pd.read_csv(input_file_path)
    print("Deleting the columns 'zipcodeOri','zipMerchant' because all the fields are equal.\n\n")
    del data['zipcodeOri']
    del data['zipMerchant']
    print("Data Shape: {} \n\nPreview: \n\n {} \n\n Data Information: \n".format( data.shape, data.head() ))
    print("\n{}\nDoes it has null values? {}".format(data.info(), data.isnull().values.any() ))
    return data

def readLocally():
    print("\n\n{} {} {}\n\n".format( 10*"_ " , " READ DATA LOCALLY " , 10*"_ "))
    
    X_train     = pickle.load( open( "X_train.data"       , "rb" ) ) 
    y_train     = pickle.load( open( "y_train.data"       , "rb" ) )
    
    X_test      = pickle.load( open( "X_test.data"        , "rb" ) )
    y_test      = pickle.load( open( "y_test.data"        , "rb" ) )
    
    X_val      = pickle.load( open( "X_val.data"         , "rb" ) )
    y_val      = pickle.load( open( "y_val.data"         , "rb" ) )
    
    labels_hash = pickle.load( open( "labels_hash.data"   , "rb" ) )
    scaler      = joblib.load( open( "scaler.data"   , "rb" ) )
    
    total_size = len(y_train)+len(y_test)+len(y_val)
    print("""\n\nSHAPES & KEYS:
    X_train          : {}   -> {:0.0f}%
    y_train          : {}
    X_test           : {}   -> {:0.0f}%
    y_test           : {}
    X_val            : {}   -> {:0.0f}%
    y_val            : {}
    ______________________
    Total Data Size  : {}
    labels_hash Keys : {}
    
    TRAIN DATA
    ______________________
    Positives        : {}   -> {:0.2f}%
    Negatives        : {}   -> {:0.2f}%
    
    TEST DATA
    ______________________
    Positives        : {}   -> {:0.2f}%
    Negatives        : {}   -> {:0.2f}%   
    
    VAL DATA
    ______________________
    Positives        : {}   -> {:0.2f}%
    Negatives        : {}   -> {:0.2f}%
    """.format(
        X_train.shape, (len(y_train) * 100 / total_size),
        y_train.shape, 
        X_test.shape,  (len(y_test) * 100 / total_size),
        y_test.shape, 
        X_val.shape,   (len(y_val) * 100 / total_size),
        y_val.shape, 
        total_size,
        labels_hash.keys(),
        len(y_train[y_train==1]), len(y_train[y_train==1])*100/len(y_train),
        len(y_train[y_train==0]), len(y_train[y_train==0])*100/len(y_train),
        len(y_test[y_test==1]), len(y_test[y_test==1])*100/len(y_test),
        len(y_test[y_test==0]), len(y_test[y_test==0])*100/len(y_test),
        len(y_val[y_val==1]), len(y_val[y_val==1])*100/len(y_val),
        len(y_val[y_val==0]), len(y_val[y_val==0])*100/len(y_val),
        
    ))
    return X_train, y_train, X_test, y_test, X_val, y_val, labels_hash , scaler

def saveLocally(
    rnn_data, rnn_mod_data, X_train, y_train, X_test, y_test, X_val, y_val, labels_hash
):
    pickle.dump( rnn_data     , open( "rnn_data.data"      , "wb" ) ) 
    pickle.dump( rnn_mod_data , open( "rnn_mod_data.data"  , "wb" ) ) 
    pickle.dump( X_train      , open( "X_train.data"       , "wb" ) ) 
    pickle.dump( X_test       , open( "X_test.data"        , "wb" ) )
    pickle.dump( X_val        , open( "X_val.data"         , "wb" ) )
    pickle.dump( y_train      , open( "y_train.data"       , "wb" ) )
    pickle.dump( y_test       , open( "y_test.data"        , "wb" ) )
    pickle.dump( y_val        , open( "y_val.data"         , "wb" ) )
    pickle.dump( labels_hash  , open( "labels_hash.data"   , "wb" ) )


def saveToCloud( 
    X_train, X_test, y_train, y_test, X_val, y_val, history, rnn, model_name, 
    home_dir="/home/ec2-user/SageMaker/"):
    print("\n\n{} {} {}\n\n".format( 10*"_ " , " SAVE TO CLOUD " , 10*"_ "))
    
    img_bucket_path   = project_path+"/images"
    data_bucket_path  = project_path+"/data"
    model_bucket_path = project_path+"/models"
    
    print("\n\nCOPYING IMAGES FILES ({})\n\n".format(img_bucket_path))
    runCommand('aws s3 cp {} {} --recursive --exclude="*" --include="{}"'.format(home_dir, img_bucket_path, "*.png"))
    print("\n\nCOPYING DATA FILES ({})\n\n".format(data_bucket_path))
    runCommand('aws s3 cp {} {} --recursive --exclude="*" --include="{}"'.format(home_dir, data_bucket_path, "*.data"))
    print("\n\nCOPYING MODEL FILES ({})\n\n".format(model_bucket_path))
    runCommand('aws s3 cp {} {} --recursive --exclude="*" --include="{}"'.format(home_dir, model_bucket_path, "*.h5"))
    
    print("\n\nImages Directory\n\n")
    try: 
        print( s3fs.S3FileSystem().ls(img_bucket_path) ); 
    except: 
        print("No Files In Folder.")
    print("\n\nData Directory\n\n")
    try: 
        print( s3fs.S3FileSystem().ls(data_bucket_path) ); 
    except: 
        print("No Files In Folder.")
    print("\n\nModel Directory\n\n")
    try: 
        print( s3fs.S3FileSystem().ls(model_bucket_path) );
    except: 
        print("No Files In Folder.")

def readDataFromCloud():
    print("\n\n{} {} {}\n\n".format( 10*"_ " , " READ DATA FROM CLOUD " , 10*"_ "))
    
    data_bucket_path = project_path+"/data"
    print("\n\nDownloading data from: "+data_bucket_path+"\n\n")
#     get_ipython().system('aws s3 cp {data_bucket_path}/X_train.data X_train.data')
#     get_ipython().system('aws s3 cp {data_bucket_path}/X_test.data  X_test.data')
#     get_ipython().system('aws s3 cp {data_bucket_path}/y_train.data y_train.data')
#     get_ipython().system('aws s3 cp {data_bucket_path}/y_test.data  y_test.data')
#     get_ipython().system('aws s3 cp {data_bucket_path}/labels_hash.data  labels_hash.data')
    
    print("\n\nList the data files.\n\n")
#     get_ipython().system('pwd')
#     get_ipython().system('ls -la *.data')

#     root_path   = "/content/gdrive/My Drive/Verafin/"
    root_path   = "/home/ec2-user/SageMaker/"
    
    X_train     = pickle.load( open( "X_train.data"       , "rb" ) ) 
    y_train     = pickle.load( open( "y_train.data"       , "rb" ) )
    
    X_test      = pickle.load( open( "X_test.data"        , "rb" ) )
    y_test      = pickle.load( open( "y_test.data"        , "rb" ) )
    
    X_val      = pickle.load( open( "X_val.data"         , "rb" ) )
    y_val      = pickle.load( open( "y_val.data"         , "rb" ) )
    
    labels_hash = pickle.load( open( "labels_hash.data"   , "rb" ) )
    
    print("""\n\nSHAPES & KEYS:
    X_train          : {}
    y_train          : {}
    X_test           : {}
    y_test           : {}
    X_val            : {}
    y_val            : {}
    labels_hash Keys : {}
    """.format(
        X_train.shape, y_train.shape, 
        X_test.shape,  y_test.shape, 
        X_val.shape,  y_val.shape, 
        labels_hash.keys() 
    ))
    return X_train, y_train, X_test, y_test, X_val, y_val, labels_hash 
