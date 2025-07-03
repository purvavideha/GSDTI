import numpy as np
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tmscoring
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pandas as pd



def compute_and_save_tanimoto_matrix(drug_df, save_path):
    """
    计算并保存 Drug 的 Tanimoto 相似度矩阵
    :param drug_df: 包含 'Drug_ID' 和 'Drug' (SMILES) 列的 DataFrame
    :param save_path: 保存相似度矩阵的路径
    """
    drug_smiles = drug_df['SMILES'].tolist()  # 提取 SMILES
    drug_ids = drug_df['DrugBank ID'].tolist()  # 提取 Drug_ID
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
    


def compute_tm_score(pair):
    pdb_file_1, pdb_file_2, i, j = pair
    if i == j:
        return (i, j, 1.0)
    try:
        alignment = tmscoring.TMscoring(pdb_file_1, pdb_file_2)
        tm_score = alignment.tmscore(**alignment.get_current_values())
    except Exception as e:
        print(f"Error computing TM-score for {pdb_file_1} and {pdb_file_2}: {e}")
        tm_score = 0.0
    return (i, j, tm_score)


def compute_and_save_tm_score_matrix_parallel_optimized(target_df, pdb_folder, save_path, num_processes=None, batch_size=1000):
    """
    高效并行计算并保存 TM-score 相似度矩阵，支持进度显示与分批写盘
    """
    target_ids = target_df['Gene'].tolist()
    num_targets = len(target_ids)
    pdb_files = [os.path.join(pdb_folder, f"{target_id}.pdb") for target_id in target_ids]

    # 只生成上三角任务，i==j直接赋1.0，不计算
    tasks = []
    for i in range(num_targets):
        for j in range(i+1, num_targets):  # i < j
            tasks.append((pdb_files[i], pdb_files[j], i, j))

    # 结果直接写入memmap文件，避免大矩阵占用内存
    tm_score_matrix = np.memmap(save_path, dtype='float32', mode='w+', shape=(num_targets, num_targets))
    tm_score_matrix[:] = 0.0
    for idx in range(num_targets):
        tm_score_matrix[idx, idx] = 1.0

    num_proc = num_processes or cpu_count()
    print(f"Using {num_proc} processes, total tasks: {len(tasks)}")
    with Pool(processes=num_proc) as pool:
        for i, j, tm_score in tqdm(pool.imap_unordered(compute_tm_score, tasks, chunksize=4), total=len(tasks)):
            tm_score_matrix[i, j] = tm_score
            tm_score_matrix[j, i] = tm_score  # 对称

    # 强制刷新到磁盘
    tm_score_matrix.flush()
    print(f"TM-score similarity matrix saved to: {save_path}")

# 用法示例
if __name__ == "__main__":
    drug_path = '/GS-DTI/data/BioSNAP/full_data/drug.csv'
    target_path = '/GS-DTI/data/BioSNAP/full_data/target.csv'
    drug_df = pd.read_csv(drug_path)
    target_df = pd.read_csv(target_path)
    pdb_folder = "/GS-DTI/data/BioSNAP/full_data/esmfold"
    save_path = "/GS-DTI/data/BioSNAP/full_data/sim_matix/target_simmatrix.dat"
    drug_save_path = '/GS-DTI/data/BioSNAP/full_data/sim_matix/drug_simmatrix.npz'
    compute_and_save_tanimoto_matrix(drug_df, drug_save_path)
    compute_and_save_tm_score_matrix_parallel_optimized(target_df, pdb_folder, save_path, num_processes=64)