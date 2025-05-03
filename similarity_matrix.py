from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import pandas as pd
import tmscoring
import os
from multiprocessing import Pool



drug_path = 'data/BindingDB/drugs.csv'
drug_df = pd.read_csv(drug_path)

target_path = 'data/BindingDB/targets.csv'
target_df = pd.read_csv(target_path)

davis_drug_path = '/home/jyjiang/drug_interaction/data/DAVIS_processed/davis_drugs/davis_drugs.csv'
davis_drug_df = pd.read_csv(davis_drug_path)

davis_target_path = '/home/jyjiang/drug_interaction/data/DAVIS_processed/davis_targets/davis_targets_new.csv'
davis_target_df = pd.read_csv(davis_target_path)
def compute_and_save_tanimoto_matrix(drug_df, save_path):
    """
    计算并保存 Drug 的 Tanimoto 相似度矩阵
    :param drug_df: 包含 'Drug_ID' 和 'Drug' (SMILES) 列的 DataFrame
    :param save_path: 保存相似度矩阵的路径
    """
    drug_smiles = drug_df['smiles'].tolist()  # 提取 SMILES
    drug_ids = drug_df['Drug_ID'].tolist()  # 提取 Drug_ID
    drug_mols = [Chem.MolFromSmiles(smiles) for smiles in drug_smiles]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) for mol in drug_mols]

    num_drugs = len(fingerprints)
    tanimoto_matrix = np.zeros((num_drugs, num_drugs))
    for i in range(num_drugs):
        for j in range(num_drugs):
            if i <= j:  # 矩阵对称
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                tanimoto_matrix[i, j] = sim
                tanimoto_matrix[j, i] = sim

    # 保存矩阵并返回
    np.savez(save_path, tanimoto_matrix)
    print(f"Tanimoto similarity matrix saved to: {save_path}")

# 示例使用
compute_and_save_tanimoto_matrix(drug_df=drug_df, save_path="/home/jyjiang/drug_interaction/data/BindingDB/drugsim_matrix.npz")


def compute_tm_score(pair):
    """
    计算两个 PDB 文件的 TM-score
    :param pair: (pdb_file_1, pdb_file_2)
    :return: (index_1, index_2, tm_score)
    """
    pdb_file_1, pdb_file_2, i, j = pair
    try:
        alignment = tmscoring.TMscoring(pdb_file_1, pdb_file_2)
        tm_score = alignment.tmscore(**alignment.get_current_values())
    except Exception as e:
        print(f"Error computing TM-score for {pdb_file_1} and {pdb_file_2}: {e}")
        tm_score = 0.0  # 如果出错，返回默认值
    return i, j, tm_score


def compute_and_save_tm_score_matrix_parallel(target_df, pdb_folder, save_path, num_processes=128):
    """
    使用并行计算 Target 的 TM-score 相似度矩阵
    :param target_df: 包含 'Target_ID' 列的 DataFrame
    :param pdb_folder: 存储 PDB 文件的文件夹路径
    :param save_path: 保存相似度矩阵的路径
    :param num_processes: 并行进程数量
    """
    target_ids = target_df['Target_ID'].tolist()  # 提取 Target_ID
    num_targets = len(target_ids)

    # 创建所有 PDB 文件路径
    pdb_files = [os.path.join(pdb_folder, f"{target_id}.pdb") for target_id in target_ids]

    # 准备任务列表：所有需要计算的 (pdb_file_1, pdb_file_2, i, j) 对
    tasks = []
    for i in range(num_targets):
        for j in range(i, num_targets):  # 只计算上三角部分
            tasks.append((pdb_files[i], pdb_files[j], i, j))

    # 初始化 TM-score 矩阵
    tm_score_matrix = np.zeros((num_targets, num_targets))

    # 使用进程池并行计算
    with Pool(processes=num_processes) as pool:
        results = pool.map(compute_tm_score, tasks)

    # 收集结果并填充矩阵
    for i, j, tm_score in results:
        tm_score_matrix[i, j] = tm_score
        tm_score_matrix[j, i] = tm_score  # 对称矩阵

    # 保存矩阵并返回
    np.savez_compressed(save_path, tm_score_matrix)
    print(f"TM-score similarity matrix saved to: {save_path}")


def compute_and_save_tm_score_matrix(target_df, pdb_folder, save_path):
    """
    计算并保存 Target 的 TM-score 相似度矩阵
    :param target_df: 包含 'Target_ID' 列的 DataFrame
    :param pdb_folder: 存储 PDB 文件的文件夹路径
    :param save_path: 保存相似度矩阵的路径
    """
    target_ids = target_df['Target_ID'].tolist()  # 提取 Target_ID
    num_targets = len(target_ids)
    # parser = PDBParser()

    # 加载所有 PDB 文件
    pdb_files = []
    for target_id in target_ids:
        pdb_path = os.path.join(pdb_folder, f"{target_id}.pdb")
        pdb_files.append(pdb_path)
        # structures[target_id] = parser.get_structure(target_id, pdb_path)

    # 初始化 TM-score 矩阵
    tm_score_matrix = np.zeros((num_targets, num_targets))
    for i in range(num_targets):
        for j in range(num_targets):
            if i <= j:  # 矩阵对称
                alignment = tmscoring.TMscoring(pdb_files[i], pdb_files[j])
                tm_score_matrix[i, j] = alignment.tmscore(**alignment.get_current_values())
                tm_score_matrix[j, i] = alignment.tmscore(**alignment.get_current_values())




    # 保存矩阵并返回
    np.savez_compressed(save_path, tm_score_matrix)
    print(f"TM-score similarity matrix saved to: {save_path}")

# 示例使用
compute_and_save_tm_score_matrix_parallel(target_df=target_df, pdb_folder="/home/jyjiang/drug_interaction/data/BindingDB/targets/esmfold", save_path='/home/jyjiang/drug_interaction/data/BindingDB/targets/target_simmatrix.npz')