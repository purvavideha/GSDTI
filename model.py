import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from pool import GraphMultisetTransformer
from torch_geometric.nn import global_max_pool as gmp

class mlp_emb_extract(nn.Module):
    def __init__(self, drug_dim, prot_dim, output_dim):
        super(mlp_emb_extract, self).__init__()
        self.protlinear = nn.Linear(prot_dim, 512)
        self.hidden1 = nn.Linear(512, 256)
        self.drug_linear = nn.Linear(drug_dim, 1024)
        self.hidden2 = nn.Linear(1024, 256)
        self.threshold=0.5
        self.prot_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.drug_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.prot_query = nn.Parameter(torch.randn(1, 1, 256))  # (L=1, hidden_dim)
        self.drug_query = nn.Parameter(torch.randn(1, 1, 256))  # (L=1, hidden_dim)
    def forward(self, drug, prot):
        x1 = self.protlinear(prot)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, 0.2)
        x1 = self.hidden1(x1)
        x1 = F.relu(x1)
        x1 = x1.unsqueeze(1)  # Add sequence dimension for attention (B, L=1, D)
        prot_query = self.prot_query.repeat(x1.size(0), 1, 1)
        x1, _ = self.prot_attention(prot_query, x1, x1)  # Self-attention (query, key, value)
        x1 = x1.squeeze(1)
        x2 = self.drug_linear(drug)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, 0.2)
        x2 = self.hidden2(x2)
        x2 = F.relu(x2)
        x2 = x2.unsqueeze(1)  # Add sequence dimension for attention (B, L=1, D)
        drug_query = self.drug_query.repeat(x2.size(0), 1, 1)
        x2, _ = self.drug_attention(drug_query, x2, x2)  # Self-attention (query, key, value)
        x2 = x2.squeeze(1)  # Remove sequence dimension (B, D)
        embedding1 = F.normalize(x1, p=2, dim=-1)
        embedding2 = F.normalize(x2, p=2, dim=-1)
        return   embedding1,embedding2  
class MLP(nn.Module):
    def __init__(self, drug_dim, prot_dim, output_dim):
        super(MLP, self).__init__()
        self.protlinear = nn.Linear(prot_dim, 512)
        self.hidden1 = nn.Linear(512, 256)
        self.drug_linear = nn.Linear(drug_dim, 1024)
        self.hidden2 = nn.Linear(1024, 256)
        self.drug_prot_linear = nn.Linear(512, 256)
        self.outlinear = nn.Linear(256, output_dim)
      

    def forward(self, drug, prot):
        x1 = self.protlinear(prot)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, 0.2)
        x1 = self.hidden1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, 0.2)
        
        x2 = self.drug_linear(drug)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, 0.2)
        x2 = self.hidden2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, 0.2)

        x3 = torch.cat((x2, x1), dim=1)
        x3 = self.drug_prot_linear(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, 0.2)
        out = self.outlinear(x3)

        return out
class AttentionComparisonModule(nn.Module):
    def __init__(self, input_dim=256, output_dim=2, hidden_dim=512, num_heads=4):
        super(AttentionComparisonModule, self).__init__()
        
        # Linear layers to encode drug and protein features
        self.drug_encoder = nn.Linear(input_dim, hidden_dim)
        self.prot_encoder = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Fully connected layers after attention
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, drug, prot):
        x1, x2 = drug, prot

        # Encode drug and protein features
        drug_encoded = F.relu(self.drug_encoder(x1))  # Shape: [batch_size, hidden_dim]
        prot_encoded = F.relu(self.prot_encoder(x2))  # Shape: [batch_size, hidden_dim]

        # Combine drug and protein features for attention
        combined = torch.stack([drug_encoded, prot_encoded], dim=1)  # Shape: [batch_size, 2, hidden_dim]
        
        # Apply attention
        attended_features, _ = self.attention(combined, combined, combined)  # Self-attention
        attended_features = attended_features.contiguous().view(attended_features.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = F.relu(self.fc1(attended_features))
        logits = self.fc2(x)
        return logits
class CrossAttentionNetwork(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=2, num_heads=4):
        super(CrossAttentionNetwork, self).__init__()
        self.prot_to_drug_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.drug_to_prot_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        # Fully connected layers after attention
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, drug, prot):
        # Reshape drug and protein for attention
        drug = drug.unsqueeze(1)  # Add sequence dimension: [batch_size, 1, input_dim]
        prot = prot.unsqueeze(1)  # Add sequence dimension: [batch_size, 1, input_dim]

        # Cross-attention: drug attends to protein, and vice versa
        drug_attended, _ = self.prot_to_drug_attention(drug, prot, prot)
        prot_attended, _ = self.drug_to_prot_attention(prot, drug, drug)

        # Concatenate attended features
        combined = torch.cat((drug_attended.squeeze(1), prot_attended.squeeze(1)), dim=1)

        # Fully connected layers
        logits = self.fc(combined)
        return logits
class ResNetComparisonModule(nn.Module):
    def __init__(self, input_dim=256, output_dim=2, hidden_dim=512, num_residual_blocks=3):
        super(ResNetComparisonModule, self).__init__()
        self.input_layer = nn.Linear(input_dim * 2, hidden_dim)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            for _ in range(num_residual_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, drug, prot):
        x1, x2 = drug, prot
       # print(x1.shape, x2.shape)

        # Concatenate drug and protein features
        x = torch.cat((x1, x2), dim=1)

        # Input layer
        x = self.input_layer(x)
        x = F.relu(x)

        # Residual blocks
        for block in self.residual_blocks:
            residual = x
            x = block(x) + residual  # Add residual connection

        # Output layer
        logits = self.output_layer(x)
        return logits
class ResNetAttentionComparisonModule(nn.Module):
    def __init__(self, input_dim=256, output_dim=2, hidden_dim=512, num_residual_blocks=3, num_heads=4):
        super(ResNetAttentionComparisonModule, self).__init__()
        
        # Initial linear layer
        self.input_layer = nn.Linear(input_dim * 2, hidden_dim)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            for _ in range(num_residual_blocks)
        ])

        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, drug, prot):
        x1, x2 = drug, prot

        # Concatenate drug and protein features
        x = torch.cat((x1, x2), dim=1)
        x = self.input_layer(x)
        x = F.relu(x)

        # Pass through residual blocks
        for block in self.residual_blocks:
            residual = x
            x = block(x) + residual

        # Prepare for attention
        x1 = x.unsqueeze(1)  # Add sequence dimension for drug
        x2 = x.unsqueeze(1)  # Add sequence dimension for protein
        combined = torch.cat((x1, x2), dim=1)  # Combine into sequence [batch_size, 2, hidden_dim]

        # Apply attention
        attended_features, _ = self.attention(combined, combined, combined)
        attended_features = attended_features.contiguous().view(attended_features.size(0), -1)  # Flatten

        # Output logits
        logits = self.output_layer(attended_features)
        return logits
class comparsion_module(nn.Module):    
    def __init__(self,input_dim=256,output_dim=2):
        super(comparsion_module,self).__init__()
        self.outlinear=nn.Linear(input_dim, output_dim)
        self.drug_prot_linear=nn.Linear(input_dim*2, input_dim)
    def forward(self, drug, prot):
        x1,x2=drug,prot
        #print(x1.shape,x2.shape)
        x3 = torch.cat((x2, x1), dim=1)
        x3 = self.drug_prot_linear(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, 0.2)
        out = self.outlinear(x3)
        
        return out
class CombinedModel(nn.Module):
    def __init__(self, model, model2):
        super(CombinedModel, self).__init__()
        self.model = model  # Frozen model
        self.model2 = model2  # Trainable model

    def forward(self, x1,x2):
        emb1,emb2 = self.model(x1,x2)   # Pass input through the frozen model
        x = self.model2(emb1,emb2)  # Pass the output through the trainable model
        return x
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
            self.poollinear=nn.Linear(512, 256)
        # Define dropout
        self.drop1 = nn.Dropout(p=0.2)
    

    def forward(self, x, data, pertubed=False):
        #x = data.x
        # Compute graph convolutional part
        x = self.drop1(x)
        for idx, gcn_layer in enumerate(self.gcn):
            if idx == 0:
                x = F.relu(gcn_layer(x, data.edge_index.long()))
            else:
                x = x + F.relu(gcn_layer(x, data.edge_index.long()))
            
            if pertubed:
                random_noise = torch.rand_like(x).to(x.device)
                x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
        if self.pooling == 'MTP':
            # Apply GraphMultisetTransformer Pooling
            g_level_feat = self.pool(x, data.batch, data.edge_index.long())
        else:
            g_level_feat = self.pool(x, data.batch)
            g_level_feat = self.poollinear(g_level_feat)
        n_level_feat = x

        return n_level_feat, g_level_feat
    
    
class GraphDTI(nn.Module):
    def __init__(self, drug_dim, prot_dim, output_dim, perturb=False, surface_feature=False):
        super(GraphDTI, self).__init__()
        self.esm_linear = nn.Linear(prot_dim, 512)
        self.esm_linear_sf = nn.Linear(prot_dim, 448)
        # self.gcn = torch.nn.DataParallel(GraphCNN(pooling='MTP'), device_ids=[0, 1])
        #self.gcn = GraphCNN(pooling='MTP')
        self.gcn = GraphCNN(pooling='MTP')
        self.surface_linear = nn.Linear(3, 16)
        self.surface_linear2 = nn.Linear(16, 64)
        
        self.drug_linear = nn.Linear(drug_dim, 1024)
        self.hidden2 = nn.Linear(1024, 256)
        self.drug_prot_linear = nn.Linear(512, 256)
        self.outlinear = nn.Linear(256, output_dim)
        self.perturb = perturb
        self.surface_feature = surface_feature
        self.gradients = None
    
    def forward(self, drug, prot_data):
        if self.surface_feature:
            x_esm = self.esm_linear_sf(prot_data.x)
            x_sf = self.surface_linear(prot_data.surface_feature)
            x_sf = self.surface_linear2(x_sf)
            x1 = torch.cat((x_esm, x_sf), dim=1)
        else:
            x1 = self.esm_linear(prot_data.x)
        gcn_n_feat1, gcn_g_feat1 = self.gcn(x1, prot_data)
        if self.perturb:
            gcn_n_feat2, gcn_g_feat2 = self.gcn(x1, prot_data, pertubed=True)
            
            x2 = self.drug_linear(drug)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, 0.2)
            x2 = self.hidden2(x2)
            # x2 = F.relu(x2)
            # x2 = F.dropout(x2, 0.2)
            
            x3 = torch.cat((x2, gcn_g_feat1), dim=1)
            x3 = self.drug_prot_linear(x3)
            x3 = F.relu(x3)
            x3 = F.dropout(x3, 0.2)
            out = self.outlinear(x3)
            
            return out, gcn_g_feat1, gcn_g_feat2
        
        else:
            x2 = self.drug_linear(drug)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, 0.2)
            x2 = self.hidden2(x2)
            # x2 = F.relu(x2)
            # x2 = F.dropout(x2, 0.2)
            
            x3 = torch.cat((x2, gcn_g_feat1), dim=1)
            x3 = self.drug_prot_linear(x3)
            x3 = F.relu(x3)
            x3 = F.dropout(x3, 0.2)
            out = self.outlinear(x3)
            return out,x2, gcn_g_feat1
            #return out, x2, gcn_g_feat1
class Graphemb_extract(nn.Module):
    def __init__(self, drug_dim, prot_dim, output_dim, perturb=False, surface_feature=False):
        super( Graphemb_extract, self).__init__()
        self.esm_linear = nn.Linear(prot_dim, 512)
        self.esm_linear_sf = nn.Linear(prot_dim, 448)
        # self.gcn = torch.nn.DataParallel(GraphCNN(pooling='MTP'), device_ids=[0, 1])
        self.gcn = GraphCNN(pooling='MTP')
        self.surface_linear = nn.Linear(3, 16)
        self.surface_linear2 = nn.Linear(16, 64)
        self.threshold=0.5
        self.drug_linear = nn.Linear(drug_dim, 1024)
        self.hidden2 = nn.Linear(1024, 256)
        self.drug_prot_linear = nn.Linear(512, 256)
        #self.outlinear = nn.Linear(256, output_dim)
        self.perturb = perturb
        self.surface_feature = surface_feature
        self.prot_attention = nn.MultiheadAttention(embed_dim=256, num_heads=16, batch_first=True)
        self.drug_attention = nn.MultiheadAttention(embed_dim=256, num_heads=16, batch_first=True)
        self.prot_query = nn.Parameter(torch.randn(1, 1, 256))  # (L=1, hidden_dim)
        self.drug_query = nn.Parameter(torch.randn(1, 1, 256))  # (L=1, hidden_dim)
    def forward(self, drug, prot_data):
        if self.surface_feature:
            x_esm = self.esm_linear_sf(prot_data.x.float())
            x_sf = self.surface_linear(prot_data.surface_feature)
            x_sf = self.surface_linear2(x_sf)
            x1 = torch.cat((x_esm, x_sf), dim=1)
        else:
            x1 = self.esm_linear(prot_data.x.float())
        gcn_n_feat1, gcn_g_feat1 = self.gcn(x1, prot_data)
        if self.perturb:
            gcn_n_feat2, gcn_g_feat2 = self.gcn(x1, prot_data, pertubed=True)
            
            x2 = self.drug_linear(drug)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, 0.2)
            x2 = self.hidden2(x2)
            # x2 = F.relu(x2)
            # x2 = F.dropout(x2, 0.2)
            
            x3 = torch.cat((x2, gcn_g_feat1), dim=1)
            x3 = self.drug_prot_linear(x3)
            x3 = F.relu(x3)
            x3 = F.dropout(x3, 0.2)
            out = self.outlinear(x3)
            
            return gcn_g_feat1, gcn_g_feat2
        
        else:
            x1=gcn_g_feat1
            x1 = F.dropout(x1, 0.2)
            x1 = F.relu(x1)
            x1 = x1.unsqueeze(1)  # Add sequence dimension for attention (B, L=1, D)
            prot_query = self.prot_query.repeat(x1.size(0), 1, 1)
            x1, _ = self.prot_attention(prot_query, x1, x1)  # Self-attention (query, key, value)
            x1 = x1.squeeze(1)
            x2 = self.drug_linear(drug)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, 0.2)
            x2 = self.hidden2(x2)
            x2 = x2.unsqueeze(1)  # Add sequence dimension for attention (B, L=1, D)
            drug_query = self.drug_query.repeat(x2.size(0), 1, 1)
            x2, _ = self.drug_attention(drug_query, x2, x2)  # Self-attention (query, key, value)
            x2 = x2.squeeze(1)  # Remove sequence dimension (B, D)
            embedding1 = F.normalize(x1, p=2, dim=-1)
            embedding2 = F.normalize(x2, p=2, dim=-1)
            # x2 = F.relu(x2)
            # x2 = F.dropout(x2, 0.2)
            
            # x3 = torch.cat((x2, gcn_g_feat1), dim=1)
            # x3 = self.drug_prot_linear(x3)
            # x3 = F.relu(x3)
            # x3 = F.dropout(x3, 0.2)
            # out = self.outlinear(x3)
            return embedding1, embedding2
            #return out, x2, gcn_g_feat1    