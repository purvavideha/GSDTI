import argparse
import os, sys
import pandas as pd
import numpy as np
import pickle
import gzip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from loss import *
from model import *
from utils import *
import deepspeed
import wandb
from torch.utils.data.distributed import DistributedSampler
davis_drug_path = 'data/DAVIS_processed/davis_drugs/davis_drugs.csv'
davis_target_path = 'data/DAVIS_processed/davis_targets/davis_targets_new.csv'
davis_drug_feature_path = 'data/DAVIS_processed/davis_drugs/kpgt_base.npz'

davis_protein_feature_path = 'data/DAVIS_processed/davis_targets/prot_repnew.pkl'
davis_data_path = 'data/DAVIS_processed/df_new.csv'
davis_graph_folderpath = 'data/DAVIS_processed/davis_targets/graph'
bindingdb_drug_path = 'data/BindingDB/drugs.csv'
bindingdb_target_path = 'data/BindingDB/targets.csv'
bindingdb_drug_feature_path = 'data/BindingDB/kpgt_base.npz'

bindingdb_protein_feature_path = 'data/BindingDB/prot_rep.pkl.gz'
bindingdb_data_path = 'data/BindingDB/df_less1000.csv'
bindingdb_graph_folderpath = 'data/BindingDB/targets/graph'
bindingdb_tanimoto_sim_path = 'data/BindingDB/drugsim_matrix.npz'
bindingdb_tmscore_path = 'data/BindingDB/targets/target_simmatrix.npz'
bindingdb_tanimoto_matrix = np.load(bindingdb_tanimoto_sim_path)['arr_0']
bindingdb_tm_score_matrix = np.load(bindingdb_tmscore_path)['arr_0']
davis_tanimoto_sim_path = 'data/DAVIS_processed/davis_drugs/drugsim_matrix.npz'
davis_tmscore_path = 'data/DAVIS_processed/davis_targets/target_simmatrix.npz'
davis_tanimoto_matrix = np.load(davis_tanimoto_sim_path)['arr_0']
davis_tm_score_matrix = np.load(davis_tmscore_path)['arr_0']

# load feature
davis_drug_feature = np.load(davis_drug_feature_path)
davis_drug_feature = davis_drug_feature['fps']
with  open(davis_protein_feature_path, 'rb') as f:
    davis_protein_feature = pickle.load(f)
# average to get same length
davis_target_feature = []
for feature in davis_protein_feature:
    feature = feature.squeeze(0)
    davis_target_feature.append(feature[1 : len(feature[0]) - 1].mean(0))
# build dict of id and feature     
davis_drug_df = pd.read_csv(davis_drug_path)
davis_target_df = pd.read_csv(davis_target_path)
davis_drug_ids = davis_drug_df['Drug_ID']
davis_target_ids = davis_target_df['Target_ID']
davis_drug_dict = dict(zip(davis_drug_ids, davis_drug_feature))

davis_data_df=pd.read_csv(davis_data_path)
davis_data_df['Label']=davis_data_df['Label'].astype(int)




bindingdb_drug_feature = np.load(bindingdb_drug_feature_path)
bindingdb_drug_feature = bindingdb_drug_feature['fps']
with  gzip.open(bindingdb_protein_feature_path, 'rb') as f:
    bindingdb_protein_feature = pickle.load(f)
# average to get same length
bindingdb_target_feature = []
for feature in bindingdb_protein_feature:
    feature = feature.squeeze(0)
    bindingdb_target_feature.append(feature[1 : len(feature[0]) - 1].mean(0))
# build dict of id and feature     
bindingdb_drug_df = pd.read_csv(bindingdb_drug_path)
bindingdb_target_df = pd.read_csv(bindingdb_target_path)
bindingdb_drug_ids = bindingdb_drug_df['Drug_ID']
bindingdb_target_ids = bindingdb_target_df['Target_ID']
bindingdb_drug_dict = dict(zip(bindingdb_drug_ids, bindingdb_drug_feature))

bindingdb_data_df = pd.read_csv(bindingdb_data_path)

def train_drug_graph(train_dataset, test_dataset, valid_dataset, model, criterion2, sim_loss_fn, args):
    # Set device
    device = torch.device(f'cuda:{args.local_rank}')

    # Initialize optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    train_sampler = DistributedSampler(train_dataset, num_replicas=4, rank=args.local_rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=4, rank=args.local_rank)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=4, rank=args.local_rank)
    # DataLoader setup
    if args.load_sim:
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler,shuffle=True,
                                  collate_fn=lambda batch: custom_collate_fn(batch, tanimoto_matrix, tm_score_matrix))
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, sampler=test_sampler,shuffle=False,
                                 collate_fn=lambda batch: custom_collate_fn(batch, tanimoto_matrix, tm_score_matrix))
        val_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, sampler=val_sampler,shuffle=False,
                                collate_fn=lambda batch: custom_collate_fn(batch, tanimoto_matrix, tm_score_matrix))
        t_f = train_mlp_sim
        e_f = evaluate_mlp_sim
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,sampler=train_sampler ,collate_fn=custom_collate_fn_zero)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, sampler=test_sampler ,collate_fn=custom_collate_fn_zero)
        val_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=custom_collate_fn_zero)
        t_f = train
        e_f = evaluate

    # Initialize Weights & Biases (wandb) if enabled
    if args.use_wandb:
        wandb.login(key=args.wandb_key)
        run = wandb.init(
            project="drug-target_graph",
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                'weight': weight
            },
            name='bindingdb_davis2',
            notes='bindingdb_crossvalid2'
        )

    print('Second stage training')

    # DeepSpeed configuration
    ds_config = {
        "train_batch_size": args.batch_size ,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": False},
        "optimizer": {
    "type": "Adam",
    "params": {
      "lr": args.learning_rate,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.0001
    }},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"}
        },
    }

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        #optimizer=optimizer,
        config=ds_config,
    )

    b_accnp = 0
    best_epoch = -1
    b_st = model_engine.state_dict()

    # Second stage training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        train_metrics = t_f(model_engine, train_loader, optimizer, criterion2, device,deepspeed=True)
        val_metrics = e_f(model_engine, val_loader, criterion2, device)

        # Save the best model
        if val_metrics[2] + val_metrics[3] > b_accnp:
            b_accnp = val_metrics[2] + val_metrics[3]
            best_epoch = epoch
            b_st = model_engine.state_dict()

        # Log metrics to wandb
        if args.local_rank==0:
            if args.use_wandb:
                (train_loss, accuracy, acc_0, acc_1, precision, recall, f1_score, mcc, auroc, auprc) = train_metrics
                wandb.log({
                    'train_loss': train_loss,
                    'train_accuracy': accuracy,
                    'train_acc_0': acc_0,
                    'train_acc_1': acc_1,
                    'train_precision': precision,
                    'train_recall': recall,
                    'train_f1_score': f1_score,
                    'train_mcc': mcc,
                    'train_auroc': auroc,
                    'train_auprc': auprc
                })
                (valid_loss, val_accuracy, val_acc_0, val_acc_1, val_precision, val_recall, val_f1_score, val_mcc, val_auroc, val_auprc) = val_metrics
                wandb.log({
                    'valid_loss': valid_loss,
                    'valid_accuracy': val_accuracy,
                    'valid_acc_0': val_acc_0,
                    'valid_acc_1': val_acc_1,
                    'valid_precision': val_precision,
                    'valid_recall': val_recall,
                    'valid_f1_score': val_f1_score,
                    'valid_mcc': val_mcc,
                    'valid_auroc': val_auroc,
                    'valid_auprc': val_auprc
                })

            print(f"Train Metrics: {train_metrics}")
            print(f"Validation Metrics: {val_metrics}")
        if epoch+1 % 10 == 0:
            model_engine.save_checkpoint('.', f"model_epoch_{epoch}")
    val_metrics = e_f(model_engine, val_loader, criterion2, device,log=True)
    # Save the best model
    torch.save(b_st, f'best_model{best_epoch}.pth')
    print(f"Best model saved at epoch {best_epoch}")

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Train drug-target interaction model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for DataLoader")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_key", type=str, default='', help="API key for Weights & Biases")
    parser.add_argument("--load_sim", action="store_true", help="Use similarity-based DataLoader collate function")
    parser.add_argument("--local_rank", type=int, default=0, help="for distribute learning")
    args = parser.parse_args()
    davis_data_df = pd.read_csv(davis_data_path)
    bindingdb_data_df=pd.read_csv(bindingdb_data_path)
    train_df = pd.read_csv('data/BindingDB/df_less1000.csv')
    test_df=pd.read_csv('data/DAVIS_processed/df_new.csv')
    train_dataset = GraphDataset_withsim(train_df, bindingdb_drug_df, bindingdb_target_df, bindingdb_drug_dict, bindingdb_graph_folderpath)
    test_dataset = GraphDataset_withsim(davis_data_df, davis_drug_df, davis_target_df, davis_drug_dict, davis_graph_folderpath)
    valid_dataset = GraphDataset_withsim(davis_data_df, davis_drug_df, davis_target_df, davis_drug_dict, davis_graph_folderpath)
    weight = torch.tensor([1.0, 2.0]).to(torch.device(f'cuda:{args.local_rank}'))

    criterion2 = nn.CrossEntropyLoss(weight=weight)
    sim_loss_fn = SimilarityLoss()

    model = GraphDTI(bindingdb_drug_feature[0].shape[0], bindingdb_target_feature[0].shape[0], 2).to(torch.device(f'cuda:{args.local_rank}'))
    # state_dict_path = ""
    # state_dict = torch.load(state_dict_path)

   # Step 3: Load the state_dict into the model
    # model.load_state_dict(state_dict)

    # Train the model
    train_drug_graph(train_dataset, test_dataset, valid_dataset, model, criterion2, sim_loss_fn, args)
