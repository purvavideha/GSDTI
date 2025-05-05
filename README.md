# GS-DTI: A Graph-Structure-Aware Framework Leveraging Large Language Models for Drug–Target Interaction Prediction

  ![Model overview] ![DTI](https://github.com/user-attachments/assets/483567f5-0302-484e-9374-a2409cccf3bf)
<!-- Optional -->


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![Open Issues](https://img.shields.io/github/issues/your-username/repo-name)](https://github.com/your-username/repo-name/issues)

A clear description of your project and its purpose.

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
### drug data preprocessing
follow the guide in KPGT(https://github.com/lihan97/KPGT) for drug feature extraction to get kpgt.npz
### protein data preprocessing
1.prepare your data in a format similar to data/BindingDB/targets/targets.csv
2.change path in protfeature.py and run it to get prot_rep.pkl
3.prepare the raw .pdb for your protein and use build_graph.py to generate graph features for your protein in .pt
## Usage

### 1. Train on BindingDB only  
```bash
deepspeed train_graph_bindingdb.py
```
*Trains a model exclusively on BindingDB data*

### 2. Train on BindingDB + Evaluate on DAVIS  
```bash
deepspeed train_graph_davis.py
```
*Trains on BindingDB then cross-validates performance on DAVIS dataset*
### 3. Train with similarity matrix 
simimlarity matrix computed based structure shows the similarity between different drugs or proteins, aligning this similarity within  drugs or proteins features generated will minorly benefit or harm training performance,you can use similarity_matrix.py to generate such similarity matrix of either protein or drugs and pass load_sim to traininng_args to test performance change. 
## Dataset Information
- **BindingDB**: Large-scale drug-target interaction database
- **DAVIS**: Benchmark dataset for binding affinity prediction


