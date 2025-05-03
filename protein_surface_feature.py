import freesasa
from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
import os
import pickle as pkl

target_path = '/ibex/ai/home/liy0f/qinze/project/drug-target/data/BindingDB/targets/targets.csv'
df_target = pd.read_csv(target_path)
esmfile_path = '/ibex/ai/home/liy0f/qinze/project/drug-target/data/BindingDB/targets/esmfold'


def extract_sasa(pdbfile):
    """
    使用 FreeSASA 计算蛋白质每个残基的溶剂可及表面积 (SASA)。

    参数:
    pdbfile: str
        输入的 PDB 文件路径。

    返回:
    np.array
        每个残基的总 SASA 值数组。
    """
    # 使用 FreeSASA 加载结构并计算 SASA
    structure = freesasa.Structure(pdbfile)  # 加载 PDB 文件为 FreeSASA 结构
    result = freesasa.calc(structure)        # 计算 SASA

    # 获取所有残基的 SASA
    residue_areas = result.residueAreas()    # 返回嵌套字典 {链: {残基: ResidueArea}}

    sasa_values = []
    for chain_id, residues in residue_areas.items():  # 第一层字典：链
        for residue_id, residue_area in residues.items():  # 第二层字典：残基
            sasa_values.append(residue_area.total)  # 提取 ResidueArea 的总 SASA

    return np.array(sasa_values)  # 返回 NumPy 数组


# 疏水性得分表（Kyte-Doolittle Hydropathy Index）
hydropathy_index = {
    'A': 1.8,  # Alanine
    'C': 2.5,  # Cysteine
    'D': -3.5, # Aspartic Acid
    'E': -3.5, # Glutamic Acid
    'F': 2.8,  # Phenylalanine
    'G': -0.4, # Glycine
    'H': -3.2, # Histidine
    'I': 4.5,  # Isoleucine
    'K': -3.9, # Lysine
    'L': 3.8,  # Leucine
    'M': 1.9,  # Methionine
    'N': -3.5, # Asparagine
    'P': -1.6, # Proline
    'Q': -3.5, # Glutamine
    'R': -4.5, # Arginine
    'S': -0.8, # Serine
    'T': -0.7, # Threonine
    'V': 4.2,  # Valine
    'W': -0.9, # Tryptophan
    'Y': -1.3  # Tyrosine
}


def extract_hydropathy(seq):
    """
    根据氨基酸序列提取疏水性特征
    """
    return np.array([hydropathy_index.get(aa, 0) for aa in seq])


def extract_charge(seq):
    """
    根据氨基酸序列提取电荷特征
    """
    charge_map = {
    'K': 1,  # Lysine (LYS)
    'R': 1,  # Arginine (ARG)
    'H': 1,  # Histidine (HIS, 正电荷)
    'D': -1, # Aspartic Acid (ASP)
    'E': -1  # Glutamic Acid (GLU, 负电荷)
}
    return np.array([charge_map.get(aa, 0) for aa in seq])  # 无电荷残基标记为 0


surface_features = []
for index, row in df_target.iterrows():
    target_id = row['Target_ID']
    seq = row['Target']
    pdbfile_path = os.path.join(esmfile_path, f"{target_id}.pdb")
    sasa = extract_sasa(pdbfile_path)
    hydropathy = extract_hydropathy(seq)
    charge = extract_charge(seq)
    surface_feature = np.stack([sasa, hydropathy, charge], axis=1)
    surface_features.append(surface_feature)
    
    
# np.save('/ibex/ai/home/liy0f/qinze/project/drug-target/data/BindingDB/targets/surface_feature.npy', np.array(surface_features))

with open('/ibex/ai/home/liy0f/qinze/project/drug-target/data/BindingDB/targets/surface_feature.pkl', 'wb') as file:
    pkl.dump(surface_features, file)