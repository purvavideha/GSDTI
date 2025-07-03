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
from sklearn.metrics import matthews_corrcoef, precision_score, f1_score, recall_score



class MyDataset(Dataset):
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
        self.df = df.reset_index(drop=True)  # 确保索引连续
        self.drug_df = drug_df.reset_index(drop=True)  # 药物 DataFrame
        self.target_df = target_df.reset_index(drop=True)  # 靶标 DataFrame
        self.drug_dict = drug_dict
        self.graph_folderpath = graph_folderpath
        self.surface_feature_path = surface_feature_path
        if surface_feature_path != '':
            with open(surface_feature_path, 'rb') as f:
                surface_features = pickle.load(f)
            self.surface_features = surface_features
        else: 
            self.surface_features = None
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
        target_feature = torch.load(graph_file_path).cuda()  # 返回 PyG 的 Data 对象

        # 转换标签为张量
        label = torch.tensor(label, dtype=torch.long).cuda() 
        
        # 获取 Drug 和 Target 在各自 DataFrame 中的索引
        drug_idx = self.drug_id_to_index[drug_id]  # 从 drug_df 中查找索引
        target_idx = self.target_id_to_index[target_id]  # 从 target_df 中查找索引
        if self.surface_features is not None:
            surface_feature = torch.tensor(self.surface_features[target_idx], dtype=torch.float32).cuda()
            target_feature.surface_feature = surface_feature
        
        # tanimoto_similarities = torch.tensor(self.tanimoto_matrix[drug_idx], dtype=torch.float32).cuda()
        # tm_score_similarities = torch.tensor(self.tm_score_matrix[target_idx], dtype=torch.float32).cuda()
        

        # 返回元组
        return drug_feature, target_feature, label, drug_idx, target_idx

    
    
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



def custom_collate_fn_test(batch):
    """
    专用于测试和验证的collate函数，不返回相似度矩阵
    """
    drugs, prots, labels, drug_idx, target_idx = zip(*batch)  # 拆分 batch 中的数据
    prots_batch = Batch.from_data_list(prots)  # 将 prot 批量化为 PyG 的 Batch
    
    return torch.stack(drugs), prots_batch, torch.stack(labels)

    
    
def build_dataset_fromdf(df, drug_dict, target_dict):
    drug_ids = df['Drug_ID'].values
    target_ids = df['Target_ID'].values
    labels = df['Label'].values

    drug_features = [drug_dict[drug_id] for drug_id in drug_ids]
    target_features = [target_dict[target_id] for target_id in target_ids]
    
    drug_features = torch.from_numpy(np.array(drug_features))
    target_features = torch.from_numpy(np.array(target_features))
    labels = torch.from_numpy(np.array(labels))
    
    dataset = MyDataset(drug_features, target_features, labels)
    
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




def train_cl(model, train_loader, optimizer, drug_contrastive_criterion, target_contrastive_criterion ,ce_criterion, device, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    correct_0 = 0
    correct_1 = 0
    total_0 = 0
    total_1 = 0
    
    
    alpha_ce = 1   
    beta = 0.05
    gamma = 0.05
    
    
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
        
        drug_contrastive_loss = drug_contrastive_criterion(drug_emb, drug_sim)
        # target contrastive loss
        target_contrastive_loss = target_contrastive_criterion(target_emb, target_sim)
            
            # loss = alpha_cl * cl_loss + alpha_ce * ce_loss + beta * drug_sim_loss + gamma * target_sim_loss
        loss = alpha_ce * ce_loss + beta * drug_contrastive_loss + gamma * target_contrastive_loss
        
        # print(f'ce loss: {ce_loss}')
        # print(f'drug cl loss: {drug_contrastive_loss}')
        # print(f'target cl loss: {target_contrastive_loss}')
        
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


def evaluate_cl(model, data_loader, ce_criterion, device):
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
    all_predictions = []
    all_labels = []
    all_probs = []
    
    
    with torch.no_grad():
        for batch_data in data_loader:
            if len(batch_data) == 5:  
                drugs, prots, labels, drug_sim, target_sim = batch_data
            else:
                drugs, prots, labels = batch_data
            drugs, prots, labels = drugs.to(device), prots.to(device), labels.to(device)

            outputs, drug_emb, target_emb = model(drugs.to(torch.float32), prots)
            # cl_loss = cl_criterion(drug_emb, target_emb, labels)
            ce_loss = ce_criterion(outputs, labels)
            # drug_sim_loss = sim_criterion(drug_emb, drug_sim)
            # target_sim_loss = sim_criterion(target_emb, target_sim)
            loss = ce_loss
            total_loss += loss.item()

            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            
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
    
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    f1 = f1_score(all_labels, all_predictions)

    accuracy = 100 * correct / total
    return total_loss / len(data_loader), accuracy, acc_0, acc_1, f1, all_predictions, all_labels, all_probs



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