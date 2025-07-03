# GS-DTI: A Graph-Structure-Aware Framework Leveraging Large Language Models for Drug–Target Interaction Prediction

![DTI]![19](https://github.com/user-attachments/assets/040551d3-0413-4f24-947c-920b9e24a817)

<!-- Optional -->


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Data-preparing ](#data-preparing )
- [Usage](#usage)
- [Dataset-Information](#dataset-information)

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

## Data-preparing 
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
finally,put KPGT/datasets/bind_drugs/kpgt_base.npz into data/yourdataset/drugs 
### protein data preprocessing
1.prepare your targets.csv

2.change path in protfeature.py and run it to get prot_rep.pkl,put it into data/yourdataset/targets (take BindingDB as yourdataset for example )
```bash
python protfeature.py
mv prot_rep.pkl  data/yourdataset/targets
```
3.prepare the raw .pdb or use esmfold to generate .pdb for your protein,put them to data/yourdataset/targets/esmfold and use build_graph.py to generate graph features for your protein in .pt which are saved to data/yourdataset/targets/graph by default.

Here is a guide to use esmfold to generate .pdb for your protein
```bash
import torch
import esm
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()
targets_df = pd.read_csv("targets.csv")

# Output directory
output_dir = "pdbs"
os.makedirs(output_dir, exist_ok=True)
def generate_pdb(sequence, target_id):
    with torch.no_grad():
        output = model.infer_pdb(sequence)
    pdb_path = os.path.join(output_dir, f"{target_id}.pdb")
    with open(pdb_path, "w") as f:
        f.write(output)
    return pdb_path
# Iterate and predict
for _, row in tqdm(targets_df.iterrows(), total=len(targets_df)):
    target_id = row["Target_ID"]
    sequence = row["Target"]
    try:
        generate_pdb(sequence, target_id)
    except Exception as e:
        print(f"[ERROR] {target_id}: {e}")
```
### simmatrix generating for contrastive learning
```bash
python sim_matrix.py
```
### processed data for quick start
you can directly use processed data at https://drive.google.com/file/d/1vLY3FkcrnaSZpOL8u5UUbA6EWoecaWhx/view?usp=drive_link for train and test on BindingDB
## Usage

### 1. Train on BindingDB and evaluate on Davis
after preprocessing  BindingDB data
```bash
python train_davis_intracl.py
```
*Trains on BindingDB then cross-validates performance on DAVIS dataset*

### 2. Train on other train/val/test sets 
after preprocessing your data to our format,change related dataset path in training script,and run
```bash
python train_yourdataset_intracl.py
```
*Trains ,validate and test on your dataset*

## Dataset-Information
- **BindingDB**: Large-scale drug-target interaction database
- **DAVIS**: Benchmark dataset for binding affinity prediction
- **BIOSNAP**: Stanford‑maintained library of biomedical network datasets


