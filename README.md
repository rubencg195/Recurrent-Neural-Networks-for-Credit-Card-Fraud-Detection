
# RECURRENT NEURAL NETWORKS FOR CREDIT CARD FRAUD DETECTION
_______

# DESCRIPTION

Fraudulent activity is hard to detect due to the low amounts of incidents among the millions  of credit  card transactions,  but  it can  cost millions  of dollars  in  monetary losses and legal costs to financial institutions.  The lack of open-source fraud labelled data and the unbalanced nature of such, make it hard for engineers and computer/data scientists  to  design  solutions  to  detect  financial  fraud.   Recurrent  Neural  Networks(RNNs) are a promising machine learning approach for fraud detection due to their ability to learn patterns to time series.  In this study, the use of RNNs for fraud detection  was  explored.   To  do  this,  first  we  gathered  the  BankSim  synthetic  and publicly available credit card transaction dataset.  Next, RNNs variations were design and tried.  Finally, a performance assessment of RNNs for fraud was made.

# GOAL

To train different RNN variations to find the best performing credit card detection model using the BankSim and PaySim data.

# MAIN FILES INDEX

**BANKSIM**
____

**ORIGINAL DATA:**  /data/banksim/\*.csv

**PROCESSED DATA:**  /data/banksim/\*.data

**MODELS:** /models/banksim/\*.h5

**VALIDATION MODELS:** /models/banksim/\*.model

**PAYSIM**
____
**ORIGINAL DATA:**  /data/paysim\*.csv

**PROCESSED DATA:**

*  (The data exceeds GitHub's size limits. These files were not included.) 

* To generate new PaySim data

	* Clone the [PaySim repository](https://github.com/EdgarLopezPhD/PaySim)
	
	* Install the [MASON libraries](https://cs.gmu.edu/~eclab/projects/mason/)
	
	* Run paysim main java class using Netbeans or other JAVA IDE using the default parameters found in the Paysim.properties file.
	
		* The only parameter modified was "nbSteps" which was increased from 720 to 1440
		
		* The parameters file used during the study is available in [data/paysim/PaySim.properties](data/paysim/PaySim.properties)
		
**MODELS:** /models/paysim/\*.h5

**VALIDATION MODELS:** /models/paysim\/*.model

# JUPYTER NOTEBOOKS

1. **BankSim.ipynb**

- Process BankSim Data.

2. **BankSim_RNN_Variations.ipynb**

- Training Different RNN Variations on the BankSim Data Set. 

3. **PaySim.ipynb**

- Process PaySim Data.

4. **Benchmarking.ipynb**

- Compare BankSim & PaySim Models Versus the Validation Models

# PYTHON FILES

1. **data.py**

- Used for BankSim's Data Transformations.

2. **models.py**

-  Contains the ML Models Parent Class

3. **visualization.py**

- Contains the Data Visualization & Benchmarking Methods.

4. **utils.py**

- Contains Shared & Useful Methods.

# FOLDERS

1. **data:** Contains the preprocessed variables generated from the BankSim data.

2. **images:** Contains the images generated by the BankSim pipeline.

3. **models:** Contains all generated BankSim & PaySim models.

4. **testing:** A set of python files used as a playing environment to test different combinations that later were introduced to the main python files.


# RESEARCH PAPER

<p>Read it: <a href="https://github.com/rubencg195/Recurrent-Neural-Networks-for-Credit-Card-Fraud-Detection/blob/master/mitacs/RUBEN_CHEVEZ_VERAFIN_MITACS_PAPER_V3.pdf">HERE</a>.</p>
