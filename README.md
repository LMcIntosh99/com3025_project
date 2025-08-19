# ECG Dataset Preprocessing and Modeling

This repository contains code for preprocessing and modeling ECG signals using both supervised and self-supervised learning techniques. It was developed as part of a group university project for the Advanced AI module and is no longer actively maintained.

## Data Requirements

To run the preprocessing, place the following CSV files from [this Kaggle dataset](https://www.kaggle.com/shayanfazeli/heartbeat) into the `/data` directory:

- `mitbih_test.csv`
- `mitbih_train.csv`
- `ptbdb_abnormal.csv`
- `ptbdb_normal.csv`

---

## Preprocessing Notebooks

### 1. **Supervised Preprocessing**
`Supervised Preprocessing.ipynb`  
- Splits the PTB and MIT-BIH datasets into training and testing sets.
- Saves the processed data into files.

### 2. **Unsupervised Preprocessing**
`Unsupervised Preprocessing.ipynb`  
- Applies transformations to the MIT-BIH dataset.
- Labels, splits, and saves the data for training and testing.

---

## Initial Experiments (in `/supervised` directory)

Run **after** completing `Supervised Preprocessing.ipynb`:

- `MITBIH supervised models.ipynb`
- `PTB supervised models.ipynb`

---

## Main Implementation (in `/representational` directory)

### 1. **Self-Supervised Training**
`Representational Models.ipynb`  
- Trains the self-supervised models.  
- **Run after** `Unsupervised Preprocessing.ipynb`.

### 2. **Transfer Learning**

- `MITBIH Transfer Models.ipynb`  
  - Experiments with transfer models for MIT-BIH.  
  - **Run after** `Representational Models.ipynb` and `Supervised Preprocessing.ipynb`.

- `PTB Transfer Models.ipynb`  
  - Experiments with transfer models for PTB.  
  - **Run after** `Representational Models.ipynb` and `Supervised Preprocessing.ipynb`.

### 3. **Model Ensembles**

- `MITBIH Transfer Model Ensemble.ipynb`  
  - Final ensemble experiments for MIT-BIH.  
  - **Run after** `MITBIH Transfer Models.ipynb` and `Supervised Preprocessing.ipynb`.

- `PTB Transfer Model Ensemble.ipynb`  
  - Final ensemble experiments for PTB.  
  - **Run after** `PTB Transfer Models.ipynb` and `Supervised Preprocessing.ipynb`.

