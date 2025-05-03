from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
import os
import pickle
import argparse
import pandas as pd
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from loss import FocalLoss, DTIContrastiveLoss, SimilarityLoss
import numba
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from datetime import datetime
logger = None
import csv
def getLogger():
    global logger
    def initLogger():
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        logger = logging.getLogger("main")
        # 设置日志格式
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        log_datefmt = "%Y-%m-%d %H:%M:%S"

        logger.setLevel(logging.INFO)  
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # 控制台日志级别
        console_formatter = logging.Formatter(log_format, datefmt=log_datefmt)
        console_handler.setFormatter(console_formatter)
        file_handler = logging.FileHandler(f"logs/{datetime.now()}.log".replace(' ','|').replace(":","-").replace('.','|',1), mode="w")  # 日志文件路径，追加模式
        file_handler.setLevel(logging.INFO)  # 文件日志级别
        file_formatter = logging.Formatter(log_format, datefmt=log_datefmt)
        file_handler.setFormatter(file_formatter)

        # 将处理器添加到日志器
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        return logger
    if logger is None:
        logger = initLogger()
    return logger

class MyDataset(Dataset):
    def __init__(self, data1, data2, labels,dg_inx,tg_inx):
        self.data1= data1
        self.data2= data2
        self.labels = labels  
        self.dg_inx=dg_inx
        self.tg_inx=tg_inx
    def __getitem__(self, index):    
        drug, prot, label,dg_inx,tg_inx = self.data1[index], self.data2[index], self.labels[index],self.dg_inx[index], self.tg_inx[index]
        if torch.cuda.is_available():
            drug = drug.cuda()
            prot = prot.cuda()
            label = label.cuda()
        return drug, prot, label,dg_inx,tg_inx

    def __len__(self):
        return len(self.data1)
    
    
    
class GraphDataset(Dataset):
    def __init__(self, data1, data2, labels):
        self.data1= data1
        self.data2= data2
        self.labels = labels  

    def __getitem__(self, index):    
        drug, prot, label = self.data1[index], self.data2[index], self.labels[index]
        if torch.cuda.is_available():
            drug = drug.cuda()
            prot = prot.cuda()
            label = label.cuda()
        return drug, prot, label

    def __len__(self):
        return len(self.data1)
    

class LazyGraphDataset(Dataset):
    def __init__(self, df, drug_dict, graph_folderpath):
        """
        延迟加载数据集
        :param df: 包含 Drug_ID、Target_ID 和 Label 的 DataFrame
        :param drug_dict: 药物特征的字典 {Drug_ID: feature_array}
        :param graph_folderpath: 目标图数据的存储路径
        """
        self.df = df
        self.drug_dict = drug_dict
        self.graph_folderpath = graph_folderpath

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取当前索引的行数据
        row = self.df.iloc[idx]
        drug_id = row['Drug_ID']
        target_id = row['Target_ID']
        label = row['Label']

        # 加载药物特征
        drug_feature = torch.tensor(self.drug_dict[drug_id], dtype=torch.float32)

        # 加载目标图数据
        graph_file_path = os.path.join(self.graph_folderpath, f"{target_id}.pt")
        target_feature = torch.load(graph_file_path)

        # 转换标签为张量
        label = torch.tensor(label, dtype=torch.long)

        return drug_feature, target_feature, label

    
    
class GraphDataset_withsim(Dataset):
    def __init__(self, df, drug_df, target_df, drug_dict, graph_folderpath, surface_feature_path=''):
        """
        延迟加载数据集，并加载相似度矩阵
        :param df: 包含 Drug_ID、Target_ID 和 Label 的 DataFrame
        :param drug_dict: 药物特征的字典 {Drug_ID: feature_array}
        :param graph_folderpath: 目标图数据的存储路径
        """
        self.df = df#.reset_index(drop=True)  # 确保索引连续
        #self.drug_df = drug_df.reset_index(drop=True)  # 药物 DataFrame            
        #self.target_df = target_df.reset_index(drop=True)  # 靶标 DataFrame   
        self.drug_df = drug_df
        self.target_df = target_df
        self.drug_dict = drug_dict
        self.graph_folderpath = graph_folderpath
        # self.surface_feature_path = surface_feature_path
        if surface_feature_path != '':
            with open(surface_feature_path, 'rb') as f:
                surface_features = pickle.load(f)
            self.surface_features = surface_features
        
        self.drug_id_to_index = {drug_id: idx for idx, drug_id in enumerate(self.drug_df['Drug_ID'])}
        self.target_id_to_index = {target_id: idx for idx, target_id in enumerate(self.target_df['Target_ID'])}

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        按索引动态加载数据
        """
        # 获取当前索引的行数据
        row = self.df.iloc[idx]
        drug_id = row['Drug_ID']
        target_id = row['Target_ID']
        label = row['Label']

        # 加载药物特征
        drug_feature = torch.tensor(self.drug_dict[drug_id], dtype=torch.float32).cuda()

        # 加载目标图数据（PyTorch Geometric 格式）
        graph_file_path = os.path.join(self.graph_folderpath, f"{target_id}.pt")
        target_feature = torch.load(graph_file_path,weights_only=False).cuda()  # 返回 PyG 的 Data 对象

        # 转换标签为张量
        label = torch.tensor(label, dtype=torch.long).cuda() 
        
        # 获取 Drug 和 Target 在各自 DataFrame 中的索引
        drug_idx = self.drug_id_to_index[drug_id]  # 从 drug_df 中查找索引
        target_idx = self.target_id_to_index[target_id]  # 从 target_df 中查找索引
        # if self.surface_features is not None:
        #     surface_feature = torch.tensor(self.surface_features[target_idx], dtype=torch.float32).cuda()
        #     target_feature.surface_feature = surface_feature
        
        # tanimoto_similarities = torch.tensor(self.tanimoto_matrix[drug_idx], dtype=torch.float32).cuda()
        # tm_score_similarities = torch.tensor(self.tm_score_matrix[target_idx], dtype=torch.float32).cuda()
        

        # 返回元组
        return drug_feature, target_feature, label, drug_idx, target_idx

def custom_collate_fn_zero(batch):
    drugs, prots, labels,drug_idx, target_idx = zip(*batch)  # 拆分 batch 中的数据
    prots_batch = Batch.from_data_list(prots)  # 将 prot 批量化为 PyG 的 Batch
    return torch.stack(drugs), prots_batch, torch.stack(labels),drug_idx, target_idx  # 返回批量化后的数据   
    
def custom_collate_fn(batch, tanimoto_matrix, tm_score_matrix):
    drugs, prots, labels, drug_idx, target_idx = zip(*batch)  # 拆分 batch 中的数据
    prots_batch = Batch.from_data_list(prots)  # 将 prot 批量化为 PyG 的 Batch
    # 将索引转换为张量
    drug_indices = torch.tensor(drug_idx, dtype=torch.long)
    target_indices = torch.tensor(target_idx, dtype=torch.long)
    tanimoto_similarities = torch.tensor(
        tanimoto_matrix[np.ix_(drug_indices, drug_indices)],  # 提取子矩阵
        dtype=torch.float32
    ).cuda()
    tm_score_similarities = torch.tensor(
        tm_score_matrix[np.ix_(target_indices, target_indices)],  # 提取子矩阵
        dtype=torch.float32
    ).cuda()
    
    return torch.stack(drugs), prots_batch, torch.stack(labels), tanimoto_similarities, tm_score_similarities  # 返回批量化后的数据
def custom_collate_fn_my(batch, tanimoto_matrix, tm_score_matrix):
    drugs, prots, labels, drug_idx, target_idx = zip(*batch)  # 拆分 batch 中的数据
    drug_indices = torch.tensor(drug_idx, dtype=torch.long)
    target_indices = torch.tensor(target_idx, dtype=torch.long)
    tanimoto_similarities = torch.tensor(
        tanimoto_matrix[np.ix_(drug_indices, drug_indices)],  # 提取子矩阵
        dtype=torch.float32
    ).cuda()
    tm_score_similarities = torch.tensor(
        tm_score_matrix[np.ix_(target_indices, target_indices)],  # 提取子矩阵
        dtype=torch.float32
    ).cuda()
    
    return torch.stack(drugs), torch.stack(prots), torch.stack(labels), tanimoto_similarities, tm_score_similarities     
    
def build_dataset_fromdf(df, drug_dict, target_dict,drug_id_to_index,target_id_to_index):
    drug_ids = df['Drug_ID'].values
    target_ids = df['Target_ID'].values
    labels = df['Label'].values

    drug_features = [drug_dict[drug_id] for drug_id in drug_ids]
    target_features = [target_dict[target_id] for target_id in target_ids]
    
    drug_features = torch.from_numpy(np.array(drug_features))
    target_features = torch.from_numpy(np.array(target_features))
    labels = torch.from_numpy(np.array(labels))
    drug_inx= [drug_id_to_index[id] for id in drug_ids]
    target_inx= [target_id_to_index[id] for id in target_ids]
    dataset = MyDataset(drug_features, target_features, labels,drug_inx,target_inx)
    
    return dataset


def build_graphdataset_fromdf(df, drug_dict, graph_folderpath):
    
    graph_file_ids = {os.path.splitext(f)[0] for f in os.listdir(graph_folderpath) if f.endswith('.pt')}
    
    df = df[df['Target_ID'].astype(str).isin(graph_file_ids)]
    
    drug_ids = df['Drug_ID'].values
    target_ids = df['Target_ID'].values
    labels = df['Label'].values

    drug_features = [drug_dict[drug_id] for drug_id in drug_ids]
    target_features = []
    
    for target_id in target_ids:
        graph_file_path = os.path.join(graph_folderpath, f"{target_id}.pt")
        graph_data = torch.load(graph_file_path)
        target_features.append(graph_data)


    drug_features = torch.from_numpy(np.array(drug_features))
    # target_features = torch.from_numpy(np.array(target_features))
    labels = torch.from_numpy(np.array(labels))
    
    dataset = GraphDataset(drug_features, target_features, labels)
    
    return dataset


@numba.jit()
def calc_accuracy(predicted : np.ndarray, labels : np.ndarray):
    pass

def train_mlp_sim(model, train_loader, optimizer, criterion, device,sim_criterion=None,deepspeed=False):
    if hasattr(model, 'model2'):
       model.model2.train()
    else:
        model.train()
    train_loss = 0
    correct = 0
    total = 0
    correct_0 = 0
    correct_1 = 0
    total_0 = 0
    total_1 = 0
    all_labels = []
    all_predictions = []
    all_probs = []  # For AUROC and AUPRC
    for drugs, prots, labels, drug_sim, target_sim in train_loader:
        drugs, prots, labels = drugs.to(device), prots.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        if isinstance(criterion,DTIContrastiveLoss):
           emb1,emb2=model(drugs.to(torch.float32), prots)
           loss= criterion(emb1,emb2, labels)
        else:    
           outputs,emb1,emb2 = model(drugs.to(torch.float32), prots)
           loss = criterion(outputs, labels)
        if sim_criterion:
            drug_sim_loss = 0.1*sim_criterion(emb1, drug_sim)
            target_sim_loss = 0.2*sim_criterion(emb2, target_sim)
            loss+=drug_sim_loss+target_sim_loss
        # Backward pass and optimization
        if not deepspeed:
            loss.backward()
            optimizer.step()
        else:
            model.backward(loss)
            model.step()

        train_loss += loss.item()
        if isinstance(criterion,DTIContrastiveLoss):
            cosine_similarity = torch.sum(emb1 * emb2, dim=-1)
            thresholds = torch.linspace(0.0, 1.0, steps=100)  # 100 thresholds from -1 to 1

        # Initialize variables to store the best threshold and maximum correct predictions
            best_threshold = model.threshold
            max_correct = 0

        # Iterate through each threshold
            for threshold in thresholds:
            # Compute predictions based on the current threshold
                predicted = (cosine_similarity >= threshold).int()

            # Count the number of correct predictions
                correct = (predicted == labels).sum().item()

            # Update the best threshold if this one is better
                if correct > max_correct:
                   max_correct = correct
                   best_threshold = threshold

        # Update the model's threshold
            model.threshold = best_threshold
            #print(model.threshold)
            predicted= (cosine_similarity >=model.threshold).int()

        else:
           _, predicted = torch.max(outputs.data, 1)
           probs = torch.softmax(outputs.data, dim=1)[:, 1]
           all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())  # Ground-truth labels
        all_predictions.extend(predicted.cpu().numpy())  # Predicted classes
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(labels.size(0)):
            if labels[i] == 0:
                total_0 += 1
                if predicted[i] == 0:
                    correct_0 += 1
            elif labels[i] == 1:
                total_1 += 1
                if predicted[i] == 1:
                    correct_1 += 1

    accuracy = 100 * correct / total
    acc_0 = correct_0 / total_0 if total_0 > 0 else 0
    acc_1 = correct_1 / total_1 if total_1 > 0 else 0
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()  # True Negatives, False Positives, False Negatives, True 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Matthews Correlation Coefficient (MCC)
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = numerator / denominator if denominator > 0 else 0   
    auroc,auprc=None,None 
    if not isinstance(criterion,DTIContrastiveLoss):
       auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
       auprc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
    return train_loss / len(train_loader), accuracy, acc_0, acc_1,precision,recall,f1_score,mcc,auroc,auprc


def evaluate_mlp_sim(model, data_loader, criterion, device,sim_criterion=None,log=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    correct_0 = 0
    correct_1 = 0
    total_0 = 0
    total_1 = 0
    all_labels = []
    all_predictions = []
    all_probs = []  # For AUROC and AUPRC
    current_time = datetime.now()
    csv_file =f"outputs_labels_log_fold{current_time}.csv"
    with torch.no_grad():
        for drugs, prots, labels, drug_sim, target_sim in data_loader:
            drugs, prots, labels = drugs.to(device), prots.to(device), labels.to(device)

            if isinstance(criterion,DTIContrastiveLoss):
               emb1,emb2=model(drugs.to(torch.float32), prots)
               loss= criterion(emb1,emb2, labels)
            else:    
               outputs,emb1,emb2 = model(drugs.to(torch.float32), prots)
               if log:
                outputs_list = outputs.tolist()  # Convert (b, 2) tensor to a list of lists
                labels_list = labels.tolist()    # Convert (b,) tensor to a list
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)

                    # Write header only if the file is empty
                    if file.tell() == 0:  # Check if the file is empty
                        writer.writerow(["Output_1", "Output_2", "Label"])  # Header row
                    # Write each output-label pair
                    for output, label in zip(outputs_list, labels_list):
                        writer.writerow(output + [label])  # Combine output values and label
               loss = criterion(outputs, labels)
            if sim_criterion:
               drug_sim_loss = 0.1*sim_criterion(emb1, drug_sim)
               target_sim_loss = 0.2*sim_criterion(emb2, target_sim)
               loss+=drug_sim_loss+target_sim_loss
            total_loss += loss.item()

            if isinstance(criterion,DTIContrastiveLoss):
                cosine_similarity = torch.sum(emb1 * emb2, dim=-1)
                predicted= (cosine_similarity >= model.threshold).int()
          
            else:
                _, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs.data, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())  # Probabilities for AUROC/AUPRC
            all_labels.extend(labels.cpu().numpy())  # Ground-truth labels
            all_predictions.extend(predicted.cpu().numpy())  # Predicted classes
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(labels.size(0)):
                if labels[i] == 0:
                    total_0 += 1
                    if predicted[i] == 0:
                        correct_0 += 1
                elif labels[i] == 1:
                    total_1 += 1
                    if predicted[i] == 1:
                        correct_1 += 1

    accuracy = 100 * correct / total
    acc_0 = correct_0 / total_0 if total_0 > 0 else 0
    acc_1 = correct_1 / total_1 if total_1 > 0 else 0
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()  # True Negatives, False Positives, False Negatives, True 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Matthews Correlation Coefficient (MCC)
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = numerator / denominator if denominator > 0 else 0   
    auroc,auprc=None,None 
    if not isinstance(criterion,DTIContrastiveLoss):
       auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
       auprc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
    
    return total_loss / len(data_loader), accuracy, acc_0, acc_1,precision,recall,f1_score,mcc,auroc,auprc


def train(model, train_loader, optimizer, criterion, device,sim_criterion=None,deepspeed=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    correct_0 = 0
    correct_1 = 0
    total_0 = 0
    total_1 = 0
    all_labels = []
    all_predictions = []
    all_probs = []  # For AUROC and AUPRC
    for drugs, prots, labels, _, _ in train_loader:
        drugs, prots, labels = drugs.to(device), prots.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        if isinstance(criterion,DTIContrastiveLoss):
           emb1,emb2=model(drugs.to(torch.float32), prots)
           loss= criterion(emb1,emb2, labels)
        else:    
           outputs,emb1, emb2 = model(drugs.to(torch.float32), prots)
           loss = criterion(outputs.float(), labels)
        
        # Backward pass and optimization
        if not deepspeed:
            loss.backward()
            optimizer.step()
        else:
            model.backward(loss)
            model.step()


        total_loss+= loss.item()
        if isinstance(criterion,DTIContrastiveLoss):
            cosine_similarity = torch.sum(emb1 * emb2, dim=-1)
            thresholds = torch.linspace(0.0, 1.0, steps=100)  # 100 thresholds from -1 to 1

        # Initialize variables to store the best threshold and maximum correct predictions
            best_threshold = model.threshold
            max_correct = 0

        # Iterate through each threshold
            for threshold in thresholds:
            # Compute predictions based on the current threshold
                predicted = (cosine_similarity >= threshold).int()

            # Count the number of correct predictions
                correct = (predicted == labels).sum().item()

            # Update the best threshold if this one is better
                if correct > max_correct:
                   max_correct = correct
                   best_threshold = threshold

        # Update the model's threshold
            model.threshold = best_threshold
            #print(model.threshold)
            predicted= (cosine_similarity >=model.threshold).int()

        else:
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs.data, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())  # Probabilities for AUROC/AUPRC
        all_labels.extend(labels.cpu().numpy())  # Ground-truth labels
        all_predictions.extend(predicted.cpu().numpy())  # Predicted classes
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(labels.size(0)):
            if labels[i] == 0:
                total_0 += 1
                if predicted[i] == 0:
                    correct_0 += 1
            elif labels[i] == 1:
                total_1 += 1
                if predicted[i] == 1:
                    correct_1 += 1

    accuracy = 100 * correct / total
    acc_0 = correct_0 / total_0 if total_0 > 0 else 0
    acc_1 = correct_1 / total_1 if total_1 > 0 else 0
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()  # True Negatives, False Positives, False Negatives, True 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Matthews Correlation Coefficient (MCC)
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = numerator / denominator if denominator > 0 else 0   
    auroc,auprc=None,None 
    if not isinstance(criterion,DTIContrastiveLoss):
       auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
       auprc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
    
    return total_loss / len(train_loader), accuracy, acc_0, acc_1,precision,recall,f1_score,mcc,auroc,auprc

def evaluate(model, data_loader, criterion, device,sim_criterion=None,log=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    correct_0 = 0
    correct_1 = 0
    total_0 = 0
    total_1 = 0
    all_labels = []
    all_predictions = []
    all_probs = []  # For AUROC and AUPRC
    current_time = datetime.now()
    csv_file =f"outputs_labels_log_fold{current_time}.csv"
    with torch.no_grad():
        for drugs, prots, labels, _, _ in data_loader:
            drugs, prots, labels = drugs.to(device), prots.to(device), labels.to(device)

            if isinstance(criterion,DTIContrastiveLoss):
              emb1,emb2=model(drugs.to(torch.float32), prots)
              loss= criterion(emb1,emb2, labels)
            else:    
              outputs,emb1, emb2 = model(drugs.to(torch.float32), prots)
              if log:
                outputs_list = outputs.tolist()  # Convert (b, 2) tensor to a list of lists
                labels_list = labels.tolist()    # Convert (b,) tensor to a list
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)

                    # Write header only if the file is empty
                    if file.tell() == 0:  # Check if the file is empty
                        writer.writerow(["Output_1", "Output_2", "Label"])  # Header row
                    # Write each output-label pair
                    for output, label in zip(outputs_list, labels_list):
                        writer.writerow(output + [label])  # Combine output values and label
              loss = criterion(outputs.float(), labels)
            total_loss += loss.item()
            if isinstance(criterion,DTIContrastiveLoss):
                cosine_similarity = torch.sum(emb1 * emb2, dim=-1)
                predicted= (cosine_similarity >= model.threshold).int()

            else:
                _, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs.data, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())  # Probabilities for AUROC/AUPRC
            all_labels.extend(labels.cpu().numpy())  # Ground-truth labels
            all_predictions.extend(predicted.cpu().numpy())  # Predicted classes
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(labels.size(0)):
                if labels[i] == 0:
                    total_0 += 1
                    if predicted[i] == 0:
                        correct_0 += 1
                elif labels[i] == 1:
                    total_1 += 1
                    if predicted[i] == 1:
                        correct_1 += 1

    accuracy = 100 * correct / total
    acc_0 = correct_0 / total_0 if total_0 > 0 else 0
    acc_1 = correct_1 / total_1 if total_1 > 0 else 0

    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()  # True Negatives, False Positives, False Negatives, True 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Matthews Correlation Coefficient (MCC)
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = numerator / denominator if denominator > 0 else 0   
    auroc,auprc=None,None 
    if not isinstance(criterion,DTIContrastiveLoss):
       auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
       auprc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
    
    return total_loss / len(data_loader), accuracy, acc_0, acc_1,precision,recall,f1_score,mcc,auroc,auprc





def train_cl(model, train_loader, optimizer, cl_criterion, ce_criterion, sim_criterion, device, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    correct_0 = 0
    correct_1 = 0
    total_0 = 0
    total_1 = 0
    
    alpha_cl = 0.5  
    alpha_ce = 1   
    beta = 0.1        
    gamma = 0.2 

    for drugs, prots, labels, drug_sim, target_sim in train_loader:
        # drugs, prots, labels = drugs.to(device), prots.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()

        outputs, drug_emb, target_emb = model(drugs.to(torch.float32), prots)
        # if epoch < 10:
        #     loss = ce_criterion(outputs, labels)
        # elif epoch >=10:    
        # cl_loss = cl_criterion(drug_emb, target_emb, labels)
        ce_loss = ce_criterion(outputs, labels)
        drug_sim_loss = sim_criterion(drug_emb, drug_sim)
        target_sim_loss = sim_criterion(target_emb, target_sim)
            
            # loss = alpha_cl * cl_loss + alpha_ce * ce_loss + beta * drug_sim_loss + gamma * target_sim_loss
        loss = alpha_ce * ce_loss + beta * drug_sim_loss + gamma * target_sim_loss
        
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(labels.size(0)):
            if labels[i] == 0:
                total_0 += 1
                if predicted[i] == 0:
                    correct_0 += 1
            elif labels[i] == 1:
                total_1 += 1
                if predicted[i] == 1:
                    correct_1 += 1

    accuracy = 100 * correct / total
    acc_0 = correct_0 / total_0 if total_0 > 0 else 0
    acc_1 = correct_1 / total_1 if total_1 > 0 else 0
    
    return train_loss / len(train_loader), accuracy, acc_0, acc_1


def evaluate_cl(model, data_loader, cl_criterion, ce_criterion, sim_criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    correct_0 = 0
    correct_1 = 0
    total_0 = 0
    total_1 = 0
    
    # alpha_cl = 0.5  
    # alpha_ce = 0.5   
    # beta = 0.2        
    # gamma = 0.2 
    
    
    with torch.no_grad():
        for drugs, prots, labels, drug_sim, target_sim in data_loader:
            drugs, prots, labels = drugs.to(device), prots.to(device), labels.to(device)

            outputs, drug_emb, target_emb = model(drugs.to(torch.float32), prots)
            # cl_loss = cl_criterion(drug_emb, target_emb, labels)
            ce_loss = ce_criterion(outputs, labels)
            # drug_sim_loss = sim_criterion(drug_emb, drug_sim)
            # target_sim_loss = sim_criterion(target_emb, target_sim)
            loss = ce_loss
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(labels.size(0)):
                if labels[i] == 0:
                    total_0 += 1
                    if predicted[i] == 0:
                        correct_0 += 1
                elif labels[i] == 1:
                    total_1 += 1
                    if predicted[i] == 1:
                        correct_1 += 1

    accuracy = 100 * correct / total
    acc_0 = correct_0 / total_0 if total_0 > 0 else 0
    acc_1 = correct_1 / total_1 if total_1 > 0 else 0

    accuracy = 100 * correct / total
    return total_loss / len(data_loader), accuracy, acc_0, acc_1





def protein_graph(sequence, edge_index, esm_embed):
    seq_code = aa2idx(sequence)
    seq_code = torch.IntTensor(seq_code)
    # add edge to pairs whose distances are more possible under 8.25
    #row, col = edge_index
    edge_index = torch.LongTensor(edge_index)
    # if AF_embed == None:
    #     data = Data(x=seq_code, edge_index=edge_index)
    # else:
    data = Data(x=torch.from_numpy(esm_embed), edge_index=edge_index, native_x=seq_code)
    return data


def aa2idx(seq):
    # convert letters into numbers
    abc = np.array(list("ARNDCQEGHILKMFPSTWYVX"), dtype='|S1').view(np.uint8)
    idx = np.array(list(seq), dtype='|S1').view(np.uint8)
    for i in range(abc.shape[0]):
        idx[idx == abc[i]] = i

    # treat all unknown characters as gaps
    idx[idx > 20] = 20
    return idx


def load_predicted_PDB(pdbfile):
    # Generate (diagonalized) C_alpha distance matrix from a pdbfile
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    residues = [r for r in structure.get_residues() if is_aa(r)]
    amino_acid_sequence = ''.join([residue.get_resname() for residue in residues])

    # sequence from atom lines
    # records = SeqIO.parse(pdbfile, 'pdb-atom')
    # seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            try:
                one = residues[x]["CA"].get_coord()
                two = residues[y]["CA"].get_coord()
                distances[x, y] = np.linalg.norm(one-two)
            except KeyError:
            # 如果没有 CA 原子，设置距离为 NaN 或其他值
                distances[x, y] = 999
            

    return distances, amino_acid_sequence


def load_predicted_PDB2(pdbfile):
    restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}


    restype_3to1 = {v: k for k, v in restype_1to3.items()}
    
    pdb = pdbfile
    parser = PDBParser()

    struct = parser.get_structure("x", pdb)
    model = struct[0]
    chain_id = list(model.child_dict.keys())[0]
    chain = model[chain_id]
    Ca_array = []
    sequence = ''
    seq_idx_list = list(chain.child_dict.keys())
    seq_len = seq_idx_list[-1][1] - seq_idx_list[0][1] + 1

    for idx in range(seq_idx_list[0][1], seq_idx_list[-1][1]+1):
        try:
            Ca_array.append(chain[(' ', idx, ' ')]['CA'].get_coord())
        except:
            Ca_array.append([np.nan, np.nan, np.nan])
        try:
            sequence += restype_3to1[chain[(' ', idx, ' ')].get_resname()]
        except:
            sequence += 'X'

    #print(sequence)
    Ca_array = np.array(Ca_array)
        
    resi_num = Ca_array.shape[0]
    G = np.dot(Ca_array, Ca_array.T)
    H = np.tile(np.diag(G), (resi_num,1))
    dismap = (H + H.T - 2*G)**0.5
    
    return dismap, sequence


def load_predicted_PDB3(pdbfile):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdbfile)
    
    sequence = []
    ca_coords = []

    # 遍历结构中的链
    for model in structure:
        for chain in model:
            for residue in chain:
                # 跳过非标准氨基酸或溶剂分子
                if residue.id[0] != " ":
                    continue
                # 提取氨基酸序列
                if "CA" in residue:
                    sequence.append(residue.resname)  # 三字母代码
                    ca_coords.append(residue["CA"].coord)  # Cα 原子坐标

    # 将三字母氨基酸代码转换为单字母代码
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'SEC': 'U'
    }
    sequence = ''.join([aa_map.get(res, 'X') for res in sequence])  # 转换为单字母
    
    
    num_atoms = np.array(ca_coords).shape[0]
    distance_matrix = np.zeros((num_atoms, num_atoms))
    
    for i in range(num_atoms):
        for j in range(i, num_atoms):
            distance = np.linalg.norm(ca_coords[i] - ca_coords[j])  # 欧几里得距离
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # 矩阵对称

    return distance_matrix, sequence
# Define a mapping from three-letter amino acid codes to one-letter codes


def load_predicted_PDB4(pdbfile):
    """
    Parse SEQRES records from PDB file content and return the full sequence.
    
    Args:
        pdb_lines (list of str): Lines from the PDB file.
    
    Returns:
        str: The sequence as a string of one-letter codes.
    """
    three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    sequence = []
    with open(pdbfile, 'r') as pdb_lines:
        for line in pdb_lines:
            # Check if the line starts with SEQRES
            if line.startswith("SEQRES") and line[11]=='A':
                # Extract the residue names (columns 20-70)
                residues = line[19:].split()
                sequence.extend(residues)

    # Convert three-letter codes to one-letter codes
    one_letter_sequence = ''.join(three_to_one[res] for res in sequence if res in three_to_one)
    return one_letter_sequence

def train_zero(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    correct_0 = 0
    correct_1 = 0
    total_0 = 0
    total_1 = 0

    for drugs, prots, labels in train_loader:
        drugs, prots, labels = drugs.to(device), prots.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(drugs.to(torch.float32), prots)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(labels.size(0)):
            if labels[i] == 0:
                total_0 += 1
                if predicted[i] == 0:
                    correct_0 += 1
            elif labels[i] == 1:
                total_1 += 1
                if predicted[i] == 1:
                    correct_1 += 1

    accuracy = 100 * correct / total
    acc_0 = correct_0 / total_0 if total_0 > 0 else 0
    acc_1 = correct_1 / total_1 if total_1 > 0 else 0
    
    return train_loss / len(train_loader), accuracy, acc_0, acc_1


def evaluate_zero(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    correct_0 = 0
    correct_1 = 0
    total_0 = 0
    total_1 = 0
    
    
    with torch.no_grad():
        for drugs, prots, labels in data_loader:
            drugs, prots, labels = drugs.to(device), prots.to(device), labels.to(device)

            outputs = model(drugs.to(torch.float32), prots)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(labels.size(0)):
                if labels[i] == 0:
                    total_0 += 1
                    if predicted[i] == 0:
                        correct_0 += 1
                elif labels[i] == 1:
                    total_1 += 1
                    if predicted[i] == 1:
                        correct_1 += 1

    accuracy = 100 * correct / total
    acc_0 = correct_0 / total_0 if total_0 > 0 else 0
    acc_1 = correct_1 / total_1 if total_1 > 0 else 0

    accuracy = 100 * correct / total
    return total_loss / len(data_loader), accuracy, acc_0, acc_1


class TrainGraphTask:
    def __init__(self, args):
        self.args = args
        # datasets
        self.train_dataset = None
        self.test_dataset = None
        self.valid_dataset = None 
        self.tanimoto_matrix : np.ndarray = None
        self.tm_score_matric : np.ndarray = None
        # device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # criterion
        self.ce_loss : nn.CrossEntropyLoss = None
        self.cl_loss : DTIContrastiveLoss = None
        self.sim_loss_fn : SimilarityLoss = None
        self.focal_loss : FocalLoss = None
        
    @staticmethod
    def addArgs(ps : argparse.ArgumentParser):
        ps = TrainGraphTask.addDataPathArgs(ps)
        ps = TrainGraphTask.addHyperParameterArgs(ps)
        return ps
 
    @staticmethod
    def addDataPathArgs(ps : argparse.ArgumentParser):
        ps.add_argument("--drug-path", required=True)
        ps.add_argument("--target-path", required=True)
        ps.add_argument("--drug-feature-path", required=True)
        ps.add_argument("--graph-folder-path", required=True)
        ps.add_argument("--data-path", required=True)
        return ps

    @staticmethod
    def addHyperParameterArgs(ps : argparse.ArgumentParser):
        ps.add_argument("--classify-loss", choices=['CE', 'Focal'], default='CE')
        ps.add_argument("--neg-samples-weight", type=float, default=2.0)
        ps.add_argument("--num-epoch", default=50, type=int)
        ps.add_argument("--batch-size", default=128, type=int)
        ps.add_argument("--base-lr", default=0.0001, type=float)
        ps.add_argument("--with-sim-loss", action='store_true', default=False)
        ps.add_argument("--rand-seed", default=42, type=int)
        ps.add_argument("--test-ratio", default=0.2, type=float)
        return ps

    def loadDatasets(self, args : argparse.Namespace):
        drug_feature = np.load(args.drug_feature_path)
        drug_feature = drug_feature['fps']

        drug_df = pd.read_csv(args.drug_path)
        drug_ids = drug_df['Drug_ID']
        drug_dict = dict(zip(drug_ids, drug_feature))
        
        data_df = pd.read_csv(args.data_path)
        train_df, test_df = train_test_split(
        data_df,
            test_size=args.test_ratio,     # 测试集占20%
            stratify=data_df['Label'],     # 按照Label列进行分层抽样
            random_state=args.rand_seed    # 固定随机种子，保证结果可复现
        )
        
        if args.with_sim_loss:
            target_df = pd.read_csv(args.target_path)
            self.train_dataset = GraphDataset_withsim(train_df, drug_df, target_df, drug_dict, args.graph_folder_path)
            self.test_dataset = GraphDataset_withsim(test_df, drug_df, target_df, drug_dict, args.graph_folder_path)
            self.valid_dataset = GraphDataset_withsim(test_df, drug_df, target_df, drug_dict, args.graph_folder_path)
        else:
            self.train_dataset = LazyGraphDataset(train_df, drug_dict, args.graph_folder_path)
            self.test_dataset = LazyGraphDataset(test_df, drug_dict, args.graph_folder_path)
            self.valid_dataset = LazyGraphDataset(test_df, drug_dict, args.graph_folder_path)

    def getCriterion(self, args : argparse.Namespace):
        if args.classify_loss == 'CE':
            weight = torch.tensor([1.0, args.neg_samples_weight]).cuda()
            self.ce_loss = nn.CrossEntropyLoss(weight=weight)
            self.cl_loss = DTIContrastiveLoss(margin=1.0, hard_negative_threshold=0.4, hard_negative_weight=1, positive_weight=1)
            self.sim_loss_fn = SimilarityLoss()
        elif args.classify_loss == 'Focal':
            weight = torch.tensor([1.0, args.neg_samples_weight]).cuda()
            self.focal_loss = FocalLoss(weight=weight)
