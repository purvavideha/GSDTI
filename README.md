# GS-DTI: A Graph-Structure-Aware Framework Leveraging Large Language Models for Drug–Target Interaction Prediction

  ![Model overview](https://github.com/user-attachments/assets/df2fc688-54ad-42a1-bc99-201528d6367a) <!-- Optional -->

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![Open Issues](https://img.shields.io/github/issues/your-username/repo-name)](https://github.com/your-username/repo-name/issues)

A clear description of your project and its purpose.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- using KPGT(https://github.com/lihan97/KPGT) for drug feature extraction
- using graph transformer on esm2 generated features for protein feature
- using MLP for interaction prediction

## Installation


### Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/purvavideha/GraphDTI.git
cd GraphDTI

# Create and activate conda environment
conda env create -f environment.yml
conda activate env-name  # Replace with your environment name
```
## Data preparing 

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

## Dataset Information
- **BindingDB**: Large-scale drug-target interaction database
- **DAVIS**: Benchmark dataset for binding affinity prediction


