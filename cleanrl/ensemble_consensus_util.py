import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def kl_div_logits(p, q, T):
    loss_func = nn.KLDivLoss(reduction = 'batchmean', log_target=True)
    loss = loss_func(F.log_softmax(p/T, dim=-1), F.log_softmax(q/T, dim=-1)) * T * T
    return loss

class ClassifierConsensusForthLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.models_num = config.num_agent
        self.alpha = config.alpha_values
        self.q = torch.nn.Parameter(torch.zeros(self.models_num))
        self.mask = torch.ones((self.models_num, self.models_num), requires_grad=False)
        for i in range(self.models_num):
            self.mask[i, i] = 0
        self.T = 1
        print("curren number of agent: ", self.models_num)
        print("curren alpha: ", self.alpha)
    def forward(self, models_logits, models_logits_multiple):
        assert self.models_num == len(models_logits), "number of agents mismatch during consensus forward"
        q_logits =  torch.stack([F.log_softmax(self.q*self.mask[i] - (1 - self.mask[i])*1e10, dim=0).cuda().view(self.models_num, 1, 1) for i in range(self.models_num)])
        models_pred = torch.stack([F.log_softmax(models_logits[i], dim=-1) for i in range(self.models_num)])
        models_pred_multi = [0 for _ in range(self.models_num)]
        for i in range(self.models_num):
            models_pred_multi[i] = torch.stack([F.log_softmax(models_logits_multiple[i][j], dim=-1) for j in range(self.models_num)]) + q_logits[i]
        ensemble_logits = torch.stack([torch.logsumexp(models_pred_multi[i], dim=0) for i in range(self.models_num)])
        ensemble_logits_normalized = F.log_softmax(ensemble_logits, dim=-1)
        kl_loss=0
        for k in range(self.models_num):
            kl_loss += kl_div_logits(models_pred[k], ensemble_logits_normalized[k], self.T)
        loss = self.alpha*kl_loss
        return loss


def reduce_ensemble_logits(teacher_logits_list):
    """Original teacher_logits_list each element format: [batch_size x num_class]"""
    
    teacher_logits =  torch.stack([logits for logits in teacher_logits_list], dim=1)
    assert teacher_logits.dim() == 3
    """teacher_logits shape: [batch_size x num_models x num_class]"""

    teacher_logits = F.log_softmax(teacher_logits, dim=-1)
    # n_teachers = len(teacher_logits)
    n_teachers = teacher_logits.shape[1]
    return torch.logsumexp(teacher_logits, dim=1) - math.log(n_teachers)

class ClassifierEnsemble(torch.nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.components = torch.nn.ModuleList(models)

    def forward(self, inputs):
        """[batch_size x num_components x ...]"""
        logits = torch.stack([model(inputs) for model in self.components], dim=1)
        logits = reduce_ensemble_logits(logits)
        return logits