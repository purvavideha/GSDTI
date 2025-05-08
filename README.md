# GS-DTI: A Graph-Structure-Aware Framework Leveraging Large Language Models for Drug–Target Interaction Prediction

![DTI](https://github.com/user-attachments/assets/483567f5-0302-484e-9374-a2409cccf3bf)
<!-- Optional -->

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![Open Issues](https://img.shields.io/github/issues/your-username/repo-name)](https://github.com/your-username/repo-name/issues)


## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Data preparing ](#datapreparing )
- [Usage](#usage)
- [Dataset Information](#datasetinformation)

## Features
- using KPGT(https://github.com/lihan97/KPGT) for drug feature extraction
- using graph transformer on esm2 generated features for protein feature
- using MLP for interaction prediction

## Installation


### Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/purvavideha/GSDTI.git
cd GSDTI

# Create and activate conda environment
conda env create -f environment.yml
conda activate env-name  # Replace with your environment name
```
## Data preparing 
### data file format
get your data in the format as of data/BindingDB df_less1000.csv
and run the following code to get distinct drugs and targets for later preprocessing
```bash
import pandas as pd
df = pd.read_csv("yourfile.csv")
drugs_df = df[['Drug_ID', 'Drug']].drop_duplicates().rename(columns={'Drug': 'smiles'}).reset_index(drop=True)
drugs_df.to_csv("drugs.csv", index=False)
targets_df = df[['Target_ID', 'Target']].drop_duplicates().reset_index(drop=True)
targets_df.to_csv("targets.csv", index=False)
 ```

### drug data preprocessing
first prepare your drugs.csv as mentioned 
follow the guide in KPGT(https://github.com/lihan97/KPGT) for drug feature extraction,
create its own env for this step only
- ```bash
  git clone https://github.com/lihan97/KPGT.git
  cd KPGT
  conda env create
  conda activate KPGT
  ```
- Then Download the pre-trained model at: https://figshare.com/s/d488f30c23946cf6898f.
  unzip it and put it in the KPGT/models/ directory.
  bring your drugs.csv to KPGT/datasets,rename it to your_dataset.csv
  ```bash
  python preprocess_downstream_dataset.py --data_path ../datasets/ --dataset your_dataset
  python extract_features.py --config base --model_path ../models/pretrained/base/base.pth --data_path ../datasets/ --dataset your_dataset
  ```
finally,put /home/hfcloudy/KPGT/datasets/bind_drugs/kpgt_base.npz into data/yourdataset/drugs 
### protein data preprocessing
1.prepare your targets.csv

2.change path in protfeature.py and run it to get prot_rep.pkl,put it into data/yourdataset/targets (take BindingDB as yourdataset for example )
```bash
python protfeature.py
mv prot_rep.pkl  data/yourdataset/targets
```
3.prepare the raw .pdb or use esmfold to generate .pdb for your protein,put them to data/yourdataset/targets/esmfold and use build_graph.py to generate graph features for your protein in .pt which are saved to data/yourdataset/targets/graph by default
## Usage

### 1. Train on BindingDB only  
after preprocessing  BindingDB data
```bash
deepspeed train_graph_bindingdb.py
```
*Trains a model exclusively on BindingDB data*

### 2. Train on BindingDB + Evaluate on DAVIS  
after preprocessing both BindingDB data and DAVIS data
```bash
deepspeed train_graph_davis.py
```
*Trains on BindingDB then cross-validates performance on DAVIS dataset*
### 3. Train with similarity matrix 
simimlarity matrix computed based structure shows the similarity between different drugs or proteins, aligning this similarity within  drugs or proteins features generated will minorly benefit or harm training performance,you can use similarity_matrix.py to generate such similarity matrix of either protein or drugs and pass load_sim to traininng_args to test performance change. 
## Dataset Information
- **BindingDB**: Large-scale drug-target interaction database
- **DAVIS**: Benchmark dataset for binding affinity prediction


