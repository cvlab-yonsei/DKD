import numpy as np
import torch


class Evaluator:
    def __init__(self, num_class, old_classes_idx=None, new_classes_idx=None):
        self.num_class = num_class
        self.old_classes_idx = old_classes_idx
        self.new_classes_idx = new_classes_idx
        self.total_classes_idx = self.old_classes_idx + self.new_classes_idx
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum() * 100
        if self.old_classes_idx and self.new_classes_idx:
            Acc_old = (
                np.diag(self.confusion_matrix)[self.old_classes_idx].sum()
                / self.confusion_matrix[self.old_classes_idx, :].sum()
            ) * 100
            Acc_new = (
                np.diag(self.confusion_matrix)[self.new_classes_idx].sum()
                / self.confusion_matrix[self.new_classes_idx, :].sum()
            ) * 100
            return {'harmonic': 2 * Acc_old * Acc_new / (Acc_old + Acc_new),
                    'old': Acc_old, 'new': Acc_new, 'overall': Acc}
        else:
            return {'overall': Acc}

    def Pixel_Accuracy_Class(self):
        Acc_by_class = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1) * 100
        Acc = np.nanmean(np.nan_to_num(Acc_by_class)[self.total_classes_idx])
        if self.old_classes_idx and self.new_classes_idx:
            Acc_old = np.nanmean(np.nan_to_num(Acc_by_class[self.old_classes_idx]))
            Acc_new = np.nanmean(np.nan_to_num(Acc_by_class[self.new_classes_idx]))
            return {'harmonic': 2 * Acc_old * Acc_new / (Acc_old + Acc_new),
                    'by_class': Acc_by_class, 'old': Acc_old, 'new': Acc_new, 'overall': Acc}
        else:
            return {'overall': Acc, 'by_class': Acc_by_class}

    def Mean_Intersection_over_Union(self):
        MIoU_by_class = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix + 1e-6)
        ) * 100
        MIoU = np.nanmean(np.nan_to_num(MIoU_by_class)[self.total_classes_idx])
        if self.old_classes_idx and self.new_classes_idx:
            MIoU_old = np.nanmean(np.nan_to_num(MIoU_by_class[self.old_classes_idx]))
            MIoU_new = np.nanmean(np.nan_to_num(MIoU_by_class[self.new_classes_idx]))
            return {'harmonic': 2 * MIoU_old * MIoU_new / (MIoU_old + MIoU_new), 'by_class': MIoU_by_class, 'old': MIoU_old, 'new': MIoU_new, 'overall': MIoU}
        else:
            return {'overall': MIoU, 'by_class': MIoU_by_class}

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def sync(self, device):
        # Collect from multi-processes
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)

        torch.distributed.reduce(confusion_matrix, dst=0)

        if torch.distributed.get_rank() == 0:
            self.confusion_matrix = confusion_matrix.cpu().numpy()
