from train_graph_bindingdb import *
davis_drug_path = '/home/jyjiang/drug_interaction/data/DAVIS_processed/davis_drugs/davis_drugs.csv'
davis_target_path = '/home/jyjiang/drug_interaction/data/DAVIS_processed/davis_targets/davis_targets_new.csv'
davis_drug_feature_path = '/home/jyjiang/drug_interaction/data/DAVIS_processed/davis_drugs/kpgt_base.npz'
davis_protein_feature_path = '/home/jyjiang/drug_interaction/data/DAVIS_processed/davis_targets/prot_repnew.pkl'
davis_data_path = '/home/jyjiang/drug_interaction/data/DAVIS_processed/df_new.csv'
davis_graph_folderpath = '/home/jyjiang/drug_interaction/data/DAVIS_processed/davis_targets/graph'
bindingdb_drug_path = 'data/BindingDB/drugs.csv'
bindingdb_target_path = 'data/BindingDB/targets.csv'
bindingdb_drug_feature_path = 'data/BindingDB/kpgt_base.npz'
bindingdb_protein_feature_path = 'data/BindingDB/prot_rep.pkl.gz'
bindingdb_data_path = 'data/BindingDB/df_less1000.csv'
bindingdb_graph_folderpath = '/home/jyjiang/drug_interaction/data/BindingDB/targets/graph'

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
# davis_target_dict = dict(zip(davis_target_ids, davis_target_feature))
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
# bindingdb_target_dict = dict(zip(bindingdb_target_ids, bindingdb_target_feature))
bindingdb_data_df = pd.read_csv(bindingdb_data_path)
device='cuda'
#train_dataset = GraphDataset_withsim(bindingdb_data_df, bindingdb_drug_df, bindingdb_target_df, bindingdb_drug_dict, bindingdb_graph_folderpath)
#val_dataset = GraphDataset_withsim(davis_data_df, davis_drug_df, davis_target_df, davis_drug_dict, davis_graph_folderpath)
fold1=pd.read_csv('/home/jyjiang/drug_interaction/data/BindingDB/fold_5/test.csv')
fold_dataset=GraphDataset_withsim(bindingdb_data_df, bindingdb_drug_df, bindingdb_target_df, bindingdb_drug_dict, bindingdb_graph_folderpath)
model3= GraphDTI(bindingdb_drug_feature[0].shape[0], 1280, 2)
model3.load_state_dict(torch.load('/home/jyjiang/drug_interaction/best_model-fold5.pth'))
model3=model3.to(device).eval()
val_loader=DataLoader(dataset=fold_dataset, batch_size=32, shuffle=False,collate_fn=custom_collate_fn_zero)
weight = torch.tensor([1.0, 2.0]).cuda()
criterion2=nn.CrossEntropyLoss(weight=weight)
val_metrics = evaluate(model3, val_loader, criterion2, device,log=False)
print(val_metrics)