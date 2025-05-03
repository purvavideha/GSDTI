# import esm
# import joblib
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
import pickle
import gzip
device = "cuda" if torch.cuda.is_available() else "cpu"

# df_davis = pd.read_csv('/home/hfcloudy/Program/drug_interaction/data/DAVIS/davis_targets.csv')
# prot_ids = df_davis['Target_ID'].values
# prot_seqs = df_davis['Target'].values
#df_bindingdb = pd.read_csv('drug-target/data/BindingDB/targets/targets.csv')
# prot_ids = df_bindingdb['Target_ID'].values
# prot_seqs = df_bindingdb['Target'].values
df_bind_data = pd.read_csv('/home/jyjiang/drug_interaction/data/BACE1/target.csv')
prot_ids = df_bind_data['Target_ID'].values
prot_seqs = df_bind_data['Target'].values

# id = prot_ids[0]
# data = prot_seqs[0]

# esm_format_data = [(id, data)]
esm_format_data = list(zip(prot_ids, prot_seqs))


esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
print('pretrain model downloded')
batch_converter = alphabet.get_batch_converter()
# esm_model = nn.DataParallel(esm_model, device_ids=[0])
esm_model = esm_model.to(device)
esm_model.eval()
# state_dict = torch.load("/home/hfcloudy/Program/drug_interaction/pretrained_models/esmc_600m_2024_12_v0.pth")
# model = ESMC(
#             d_model=1152,
#             n_heads=18,
#             n_layers=36,
#             tokenizer=get_esmc_model_tokenizers(),
#             use_flash_attn=True,
#         ).eval()                           #these are settings in the official repo
# model.load_state_dict(state_dict)
# model.cuda()
token_representations = []


for sample in esm_format_data:
   batch_labels, batch_strs, batch_tokens = batch_converter([sample])
   print(batch_tokens.shape)
   batch_lens = (batch_tokens != alphabet.padding_idx).sum(1).cuda()
   batch_tokens = batch_tokens.cuda()

   with torch.no_grad():
       results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
   token_representations.append(results["representations"][33].cpu().detach().numpy())
# for sample in esm_format_data:
#     protein=sample[1]
#     protein_tensor = model.encode(ESMProtein(sequence=protein))
#     with torch.no_grad():
#        logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
#     token_representations.append(logits_output.embeddings.cpu())


# token_representations = np.array(token_representations)

# with gzip.open('drug-target/data/BindingDB/targets/prot_rep_esmc.pkl.gz', 'wb') as f:
#     pickle.dump(token_representations, f)
with open('/home/jyjiang/drug_interaction/data/BACE1/prot_rep.pkl', 'wb') as f:
    pickle.dump(token_representations, f)
    
# np.savez_compressed('/data/yuqinze/project/drug-target/data/BindingDB/targets/prot_rep.npz', token_representations)
# np.save('/data/yuqinze/project/drug-target/data/BindingDB/targets/prot_id.npy', prot_ids)



# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
# sequence_representations = []
# for i, tokens_len in enumerate(batch_lens):
#     sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    
print(1)

