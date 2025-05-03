import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    '''
    Multi-class Focal Loss
    '''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        # self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss
    


# class DTIContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0, hard_negative_threshold=0.4, hard_negative_weight=2.0, positive_weight=2):
#         """
#         DTI contrastive learning loss function combined with hard negative sample mining
#         :param margin: Margin, minimum distance constraint for negative samples
#         :param hard_negative_threshold: threshold for hard samples
#         :param hard_negative_weight: weight for hard samples
#         """
#         super(DTIContrastiveLoss, self).__init__()
#         self.margin = margin
#         self.hard_negative_threshold = hard_negative_threshold
#         self.hard_negative_weight = hard_negative_weight
#         self.positive_weight = positive_weight

#     def forward(self, drug_embeddings, target_embeddings, labels):
#         """
#         Compute DTI contrastive learning loss
#         :param drug_embeddings: drug embedings
#         :param target_embeddings: target embeddings same shape with drug embeddings
#         :param labels: Labels, shape [batch_size]
#         :return: loss
#         """
#         # Eu distance between drugs and targets embeddings
#         distances = torch.norm(drug_embeddings - target_embeddings, p=2, dim=1)  # [batch_size]

#         # positives loss
#         positive_mask = labels == 1
#         positive_loss = self.positive_weight * (positive_mask * torch.pow(distances, 2))

#         # negatives loss
#         negative_mask = labels == 0
#         negative_loss = negative_mask * torch.pow(torch.relu(self.margin - distances), 2)

#         # select hard negatives
#         hard_negative_mask = negative_mask * (distances < self.hard_negative_threshold)
#         hard_negative_loss = hard_negative_mask * torch.pow(torch.relu(self.margin - distances), 2)

#         # high weight for hard negatives
#         total_negative_loss = torch.sum(negative_loss) + self.hard_negative_weight * torch.sum(hard_negative_loss)
        

#         total_loss = (torch.sum(positive_loss) + total_negative_loss) / drug_embeddings.size(0)

#         return total_loss
class DTIContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, hard_negative_threshold=0.4, hard_negative_weight=2.0, positive_weight=2):
        """
        DTI contrastive learning loss function combined with hard negative sample mining
        with dynamic hyperparameter adjustment.
        """
        super(DTIContrastiveLoss, self).__init__()
        self.margin = margin
        self.hard_negative_threshold = hard_negative_threshold
        self.hard_negative_weight = hard_negative_weight
        self.positive_weight = positive_weight

        # Track metrics for dynamic hyperparameter adjustment
        self.running_positive_loss = 0.0
        self.running_negative_loss = 0.0
        self.running_hard_negative_loss = 0.0
        self.num_batches = 0

    def forward(self, drug_embeddings, target_embeddings, labels):
        """
        Compute DTI contrastive learning loss.
        """
        # Euclidean distance between drug and target embeddings
        distances = torch.norm(drug_embeddings - target_embeddings, p=2, dim=1)  # [batch_size]

        # Positive loss: Penalize distances for positive pairs
        positive_mask = labels == 1
        positive_loss = self.positive_weight * (positive_mask * torch.pow(distances, 2))

        # Negative loss: Penalize small distances for negative pairs
        negative_mask = labels == 0
        negative_loss = negative_mask * torch.pow(torch.relu(self.margin - distances), 2)

        # Hard negative loss: Extra penalty for hard negatives
        hard_negative_mask = negative_mask * (distances < self.hard_negative_threshold)
        hard_negative_loss = hard_negative_mask * torch.pow(torch.relu(self.margin - distances), 2)

        # Weighted sum of negative losses
        total_negative_loss = torch.sum(negative_loss) + self.hard_negative_weight * torch.sum(hard_negative_loss)

        # Compute total loss
        total_loss = (torch.sum(positive_loss) + total_negative_loss) / drug_embeddings.size(0)

        # Track running losses for dynamic adjustment
        self.running_positive_loss += torch.sum(positive_loss).item()
        self.running_negative_loss += torch.sum(negative_loss).item()
        self.running_hard_negative_loss += torch.sum(hard_negative_loss).item()
        self.num_batches += 1

        return total_loss

    def adjust_hyperparameters(self):
        """
        Dynamically adjust the hyperparameters based on running loss statistics.
        Called at the end of each epoch.
        """
        # Compute average contributions of different loss components
        avg_positive_loss = self.running_positive_loss / self.num_batches
        avg_negative_loss = self.running_negative_loss / self.num_batches
        avg_hard_negative_loss = self.running_hard_negative_loss / self.num_batches

        # Adjust `positive_weight`:
        # Increase positive weight if positive loss is low compared to negatives
        if avg_positive_loss < avg_negative_loss:
            self.positive_weight *= 1.1  # Increase by 10%
        else:
            self.positive_weight *= 0.95  # Decrease by 5%

        # Adjust `hard_negative_weight`:
        # Increase hard negative weight if hard negatives contribute significantly
        if avg_hard_negative_loss > 0.5 * avg_negative_loss:
            self.hard_negative_weight *= 1.2  # Increase by 20%
        else:
            self.hard_negative_weight *= 0.9  # Decrease by 10%

        # Adjust `hard_negative_threshold`:
        # Gradually increase threshold to include more hard negatives as training progresses
        self.hard_negative_threshold = min(self.hard_negative_threshold + 0.01, self.margin)

        # Adjust `margin`:
        # Gradually shrink the margin if negative loss is low (model is separating well)
        if avg_negative_loss < avg_positive_loss:
            self.margin = max(self.margin * 0.95, 0.5)  # Shrink margin but keep it above 0.5

        # Reset running statistics
        self.running_positive_loss = 0.0
        self.running_negative_loss = 0.0
        self.running_hard_negative_loss = 0.0
        self.num_batches = 0


class SimilarityLoss(nn.Module):
    def __init__(self):
        """
        Structural similarity prediction loss
        """
        super(SimilarityLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, embeddings, similarity_matrix):
        """
        :param embeddings: drug or target embeddings, [num_entities, embed_dim]
        :param similarity_matrix: similarity matrix, [num_entities, num_entities]
        :return: loss
        """
        # cos similarity
        pred_similarity = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        loss = self.mse_loss(pred_similarity, similarity_matrix)
        return loss