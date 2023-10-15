import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MethyLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=None):
        super(MethyLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, weight):
        target = target.view(-1, 1)
        weight = weight.view(-1, 1)
        # compute pt
        logpt = F.log_softmax(input,-1)
        logpt = logpt.gather(1, target) # keep the real labels' logit value
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp()) # = exp(log(x)) = x
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1)) # keep the real labels' alpha value
            # log_pt * alpha
            logpt = logpt * Variable(at)
        active_weight = 1*(weight == target)
        methyl_weight = torch.where(active_weight>0, 0, 1)
        # focal loss 计算公式
        loss = -1 * logpt * (2 ** (methyl_weight*(1-pt)))
        if self.size_average:
            # 控制是否平均
            return loss.mean()
        else:
            return loss.sum()


