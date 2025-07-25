============================================================================================================================
Files for preprocessing the data:

To run the preprocessing, you must include the following csv files from Kaggle in the "/data" directory:
-	mitbih_test.csv
-	mitbih_train.csv
-	ptbdb_abnormal.csv
-	ptbdb_normal.csv

The files from Kaggle can be found here: https://www.kaggle.com/shayanfazeli/heartbeat 

- Supervised Preprocessing.ipynb
Split the PTB and MIT-BIH datasets for training and testing and save the processed data in files.

- Unsupervised Preprocessing.ipynb
Apply transformations to the MIT dataset, label the data, split them for training and testing and store in files.

============================================================================================================================
Files for initial experiments in the “/supervised” directory:

-	MITBIH supervised models.ipynb
-	PTB supervised models.ipynb
These files must be run after running Supervised Preprocessing.ipynb.

============================================================================================================================

Files for main implementation in the "/representational" directory:

-	Representational Models.ipynb
Trains the self-supervised models. This file must be run after Unsupervised Preprocessing.ipynb

-	MITBIH Transfer Models.ipynb
Experiment with the transfer models for MIT-BIH. This file must be run after Representational Models.ipynb and Supervised Preprocessing.ipynb

-	PTB Transfer Models.ipynb
Experiment with the transfer models for PTB. This file must be run after Representational Models.ipynb and Supervised Preprocessing.ipynb

-	MITBIH Transfer Model Ensemble.ipynb
Experiment with the final ensemble for MIT-BIH. This file must be run after MITBIH Transfer Models.ipynb and Supervised Preprocessing.ipynb

-	PTB Transfer Model Ensemble.ipynb
Experiment with the final ensemble for PTB. This file must be run after PTB Transfer Models.ipynb and Supervised Preprocessing.ipynb

============================================================================================================================