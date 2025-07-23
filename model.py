import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from pool import GraphMultisetTransformer
from torch_geometric.nn import global_max_pool as gmp

    
class GraphCNN(nn.Module):
    def __init__(self, channel_dims=[512, 512, 512], fc_dim=512, num_classes=256, pooling='MTP'):
        super(GraphCNN, self).__init__()

        # Define graph convolutional layers
        gcn_dims = [512] + channel_dims

        gcn_layers = [GCNConv(gcn_dims[i-1], gcn_dims[i], bias=True) for i in range(1, len(gcn_dims))]

        self.gcn = nn.ModuleList(gcn_layers)
        self.pooling = pooling
        if self.pooling == "MTP":
            self.pool = GraphMultisetTransformer(512, 256, 256, None, 10000, 0.25, ['GMPool_G', 'GMPool_G'], num_heads=8, layer_norm=True)
        else:
            self.pool = gmp
        # Define dropout
        self.drop1 = nn.Dropout(p=0.2)
    

    def forward(self, x, data):
        #x = data.x
        # Compute graph convolutional part
        x = self.drop1(x)
        for idx, gcn_layer in enumerate(self.gcn):
            if idx == 0:
                x = F.relu(gcn_layer(x, data.edge_index.long()))
            else:
                x = x + F.relu(gcn_layer(x, data.edge_index.long()))
                
        if self.pooling == 'MTP':
            # Apply GraphMultisetTransformer Pooling
            g_level_feat = self.pool(x, data.batch, data.edge_index.long())
        else:
            g_level_feat = self.pool(x, data.batch)

        n_level_feat = x


        return n_level_feat, g_level_feat
    
    
class GraphDTI_bi(nn.Module):
    def __init__(self, drug_dim, prot_dim, output_dim, surface_feature=True):
        super(GraphDTI_bi, self).__init__()
        self.esm_linear = nn.Linear(prot_dim, 512)
        self.esm_linear_sf = nn.Linear(prot_dim, 448)
        # self.gcn = torch.nn.DataParallel(GraphCNN(pooling='MTP'), device_ids=[0, 1])
        self.gcn = GraphCNN(pooling='MTP')
        self.surface_linear = nn.Linear(3, 16)
        self.surface_linear2 = nn.Linear(16, 64)
        
        self.drug_linear = nn.Linear(drug_dim, 1024)
        self.hidden2 = nn.Linear(1024, 256)
        self.drug_prot_linear = nn.Linear(512, 256)
        self.bilinear = nn.Bilinear(256, 256, 256)
        self.outlinear = nn.Linear(256, output_dim)
        self.surface_feature = surface_feature
        
    def forward(self, drug, prot_data):
        if self.surface_feature:
            x_esm = self.esm_linear_sf(prot_data.x.float())
            x_sf = self.surface_linear(prot_data.surface_feature)
            x_sf = self.surface_linear2(x_sf)
            x1 = torch.cat((x_esm, x_sf), dim=1)
        else:
            x1 = self.esm_linear(prot_data.x.float())
        gcn_n_feat1, gcn_g_feat1 = self.gcn(x1, prot_data)
       
        x2 = self.drug_linear(drug)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, 0.2)
        x2 = self.hidden2(x2)
        # x2 = F.relu(x2)
        # x2 = F.dropout(x2, 0.2)
            
        # x3 = torch.cat((x2, gcn_g_feat1), dim=1)
        # x3 = self.drug_prot_linear(x3)
        x3 = self.bilinear(x2, gcn_g_feat1)
        # x3 = torch.cat([x2, gcn_g_feat1, bilinear_out], dim=1) 
        x3 = F.relu(x3)
        x3 = F.dropout(x3, 0.2)
        out = self.outlinear(x3)
        
        return out, x2, gcn_g_feat1
        
    def get_pair_emb(self, drug, prot_data):
        x1 = self.esm_linear(prot_data.x.float())
        gcn_n_feat1, gcn_g_feat1 = self.gcn(x1, prot_data)
        x2 = self.drug_linear(drug)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, 0.2)
        x2 = self.hidden2(x2)

        x3 = self.bilinear(x2, gcn_g_feat1)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, 0.2)
        
        return x3, x2, gcn_g_feat1
    
    def predict_with_uncertainty(self, drug, prot_data, T=50, focus_class=None):
        """
        Compute mean prediction and uncertainty via MC Dropout, returning both pre-softmax (logits) and post-softmax (probs) versions.
        """
        original_mode = self.training
        self.train()  # Enable dropout
        
        logits_samples = []
        with torch.no_grad():
            for _ in range(T):
                out, _, _ = self.forward(drug, prot_data)
                logits_samples.append(out)
        
        logits_samples = torch.stack(logits_samples, dim=0)
        
        if focus_class is not None:
            logits_samples = logits_samples[:, :, focus_class].unsqueeze(-1) 
        
        # Compute mean and uncertainty for logits (pre-softmax)
        mean_logits = torch.mean(logits_samples, dim=0)
        uncertainty_logits = torch.var(logits_samples, dim=0)
        
        
        self.train(original_mode)
        
        return mean_logits, uncertainty_logits
    
