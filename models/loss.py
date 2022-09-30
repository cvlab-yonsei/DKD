import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class BCELoss(nn.Module):
    def __init__(self, ignore_index=255, ignore_bg=True, pos_weight=None, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.pos_weight = pos_weight
        self.reduction = reduction

        if ignore_bg is True:
            self.ignore_indexes = [0, self.ignore_index]
        else:
            self.ignore_indexes = [self.ignore_index]

    def forward(self, logit, label, logit_old=None):
        # logit:     [N, C_tot, H, W]
        # logit_old: [N, C_prev, H, W]
        # label:     [N, H, W] or [N, C, H, W]
        C = logit.shape[1]
        if logit_old is None:
            if len(label.shape) == 3:
                # target: [N, C, H, W]
                target = torch.zeros_like(logit).float().to(logit.device)
                for cls_idx in label.unique():
                    if cls_idx in self.ignore_indexes:
                        continue
                    target[:, int(cls_idx)] = (label == int(cls_idx)).float()
            elif len(label.shape) == 4:
                target = label
            else:
                raise NotImplementedError
            
            logit = logit.permute(0, 2, 3, 1).reshape(-1, C)
            target = target.permute(0, 2, 3, 1).reshape(-1, C)

            return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction=self.reduction)(logit, target)
        else:
            if len(label.shape) == 3:
                # target: [N, C, H, W]
                target = torch.zeros_like(logit).float().to(logit.device)
                target[:, 1:logit_old.shape[1]] = logit_old.sigmoid()[:, 1:]
                for cls_idx in label.unique():
                    if cls_idx in self.ignore_indexes:
                        continue
                    target[:, int(cls_idx)] = (label == int(cls_idx)).float()
            else:
                raise NotImplementedError
            
            loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction=self.reduction)(logit, target)
            del target

            return loss


class WBCELoss(nn.Module):
    def __init__(self, ignore_index=255, pos_weight=None, reduction='none', n_old_classes=0, n_new_classes=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.n_old_classes = n_old_classes  # |C0:t-1| + 1(bg), 19-1: 20 | 15-5: 16 | 15-1: 16...
        self.n_new_classes = n_new_classes  # |Ct|, 19-1: 1 | 15-5: 5 | 15-1: 1
        
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=self.reduction)
        
    def forward(self, logit, label):
        # logit:     [N, |Ct|, H, W]
        # label:     [N, H, W]

        N, C, H, W = logit.shape
        target = torch.zeros_like(logit, device=logit.device).float()
        for cls_idx in label.unique():
            if cls_idx in [0, self.ignore_index]:
                continue
            target[:, int(cls_idx) - self.n_old_classes] = (label == int(cls_idx)).float()
        
        loss = self.criterion(
            logit.permute(0, 2, 3, 1).reshape(-1, C),
            target.permute(0, 2, 3, 1).reshape(-1, C)
        )

        if self.reduction == 'none':
            return loss.reshape(N, H, W, C).permute(0, 3, 1, 2)  # [N, C, H, W]
        elif self.reduction == 'mean':
            return loss
        else:
            raise NotImplementedError


class KDLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logit, logit_old=None):
        # logit:     [N, |Ct|, H, W]
        # logit_old: [N, |Ct|, H, W]
        
        N, C, H, W = logit.shape
        loss = self.criterion(
            logit.permute(0, 2, 3, 1).reshape(-1, C),
            logit_old.permute(0, 2, 3, 1).reshape(-1, C)
        ).reshape(N, H, W, C).permute(0, 3, 1, 2)
        return loss


class ACLoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logit):
        # logit: [N, 1, H, W]
        
        return self.criterion(logit, torch.zeros_like(logit))
        # loss = -torch.log(1 - logit.sigmoid())
