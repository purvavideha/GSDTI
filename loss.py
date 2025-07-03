import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
    

    
    
class NTXentContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, sim_threshold=0.8):
        super(NTXentContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.sim_threshold = sim_threshold

    def forward(self, embeddings, sim_matrix):
        """
        embeddings: [batch_size, embed_dim]
        sim_matrix: [batch_size, batch_size] (Tanimoto or TM-score)
        """
        batch_size = embeddings.size(0)
        embeddings = F.normalize(embeddings, dim=1)
        logits = torch.matmul(embeddings, embeddings.t()) / self.temperature  # [B, B]
        positive_mask = (sim_matrix > self.sim_threshold).float()
        positive_mask.fill_diagonal_(0)
        # log-softmax
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        loss = - (positive_mask * log_prob).sum() / (positive_mask.sum() + 1e-8)
        return loss
    
