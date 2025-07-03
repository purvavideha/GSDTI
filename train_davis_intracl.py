from inspect import Parameter
import os, sys
# from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
# from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader, Dataset
from model import *
import pickle
from utils import *
import wandb
from torch_geometric.data import Data, Batch
from loss import *
from sklearn.model_selection import train_test_split

from accelerate import Accelerator
from accelerate.utils import LoggerType




drug_path = '/GS-DTI/data/bindingDB/drugs/drugs.csv'
target_path = '/GS-DTI/data/bindingDB/targets/targets.csv'
drug_feature_path = '/GS-DTI/data/bindingDB/drugs/kpgt_base.npz'
graph_folderpath = '/GS-DTI/data/bindingDB/targets/graph'
data_path = '/GS-DTI/data/bindingDB/df_less1000.csv'
tanimoto_sim_path = '/GS-DTI/data/bindingDB/drugs/drug_simmatrix.npz'
tmscore_path = '/GS-DTI/data/bindingDB/targets/target_simmatrix.npz'

davis_data_path = '/GS-DTI/data/DAVIS_processed/df_new.csv'
davis_drug_path = '/GS-DTI/data/DAVIS_processed/davis_drugs/davis_drugs.csv'
davis_target_path = '/GS-DTI/data/DAVIS_processed/davis_targets/davis_targets_new.csv'
davis_drug_feature_path = '/GS-DTI/data/DAVIS_processed/davis_drugs/kpgt_base.npz'
davis_graph_folderpath = '/GS-DTI/data/DAVIS_processed/davis_targets/graph'

result_save_dir = '/GS-DTI/results'

train_drug_feature = np.load(drug_feature_path)
train_drug_feature = train_drug_feature['fps']

train_drug_df = pd.read_csv(drug_path)
train_drug_ids = train_drug_df['Drug_ID']
train_drug_dict = dict(zip(train_drug_ids, train_drug_feature))

train_target_df = pd.read_csv(target_path)

tanimoto_matrix = np.load(tanimoto_sim_path)['arr_0']
tm_score_matrix = np.load(tmscore_path)['arr_0']



davis_drug_feature = np.load(davis_drug_feature_path)
davis_drug_feature = davis_drug_feature['fps']

davis_drug_df = pd.read_csv(davis_drug_path)
davis_drug_ids = davis_drug_df['Drug_ID']
davis_drug_dict = dict(zip(davis_drug_ids, davis_drug_feature))

davis_target_df = pd.read_csv(davis_target_path)



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # create train valid test dataset
    train_df = pd.read_csv(data_path)
    train_df, validation_df = train_test_split(
    train_df,
    test_size=0.1,  # 测试集占20%
    stratify=train_df['Label'],  # 按照Label列进行分层抽样
    random_state=42  # 固定随机种子，保证结果可复现
    )
    test_df = pd.read_csv(davis_data_path)
    
    
    train_dataset = GraphDataset_withsim(train_df, train_drug_df, train_target_df, train_drug_dict, graph_folderpath)
    test_dataset = GraphDataset_withsim(test_df, davis_drug_df, davis_target_df, davis_drug_dict, davis_graph_folderpath)
    valid_dataset = GraphDataset_withsim(validation_df, train_drug_df, train_target_df, train_drug_dict, graph_folderpath)
    print('dataset created')
    
    model = GraphDTI_bi(train_drug_feature[0].shape[0], 1280, 2, surface_feature=False)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    print('model created')
    
    epochs = 30
    learning_rate = 0.00005
    batch_size = 64
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    
    weights = torch.tensor([1.0, 1.0]).cuda()  # 仍然给正样本更高权重
    focal_criterion = FocalLoss(alpha=1, gamma=2, weight=weights)
    drug_contrastive_criterion = NTXentContrastiveLoss(temperature=0.07, sim_threshold=0.8)
    target_contrastive_criterion = NTXentContrastiveLoss(temperature=0.07, sim_threshold=0.5)

    
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=lambda batch: custom_collate_fn(batch, tanimoto_matrix, tm_score_matrix))
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=lambda batch: custom_collate_fn_test(batch))
    val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=lambda batch: custom_collate_fn_test(batch))
    
    best_f1 = 0
    best_epoch = 0
    best_predictions = None
    best_labels = None
    best_probs = None
    
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        train_loss, train_acc, train_acc_neg, train_acc_pos = train_cl(model, train_loader, optimizer, drug_contrastive_criterion, target_contrastive_criterion, focal_criterion, device, epoch)
        val_loss, val_acc, val_acc_neg, val_acc_pos, val_f1, val_preds, val_labels, val_probs = evaluate_cl(model, val_loader, focal_criterion, device)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            
            best_predictions = val_preds.copy()
            best_labels = val_labels.copy()
            best_probs = val_probs.copy()

        
    
    results_df = pd.DataFrame({
        'true_label': best_labels,
        'predicted_label': best_predictions,
        'prob_class_0': best_probs[:, 0],
        'prob_class_1': best_probs[:, 1],
    })
    
    results_df.to_csv(os.path.join(result_save_dir, 'davis_result.csv'), index=False)
    