import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

'''
    Dice Loss function
'''    
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    input = nn.functional.softmax(input, dim=1)
    target = nn.functional.one_hot(target.long(),  num_classes = 8).permute(0, 3, 1, 2)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

'''
    Model evaluation metric
'''   

def pixelAccuracy(gen_mask, input_mask):
    pixel_labeled = torch.sum(input_mask > 0).float()
    pixel_corr = torch.sum((gen_mask == input_mask) * (input_mask > 0)).float()
    pixel_acc = pixel_corr / (pixel_labeled + 1e-10)

    return pixel_acc, pixel_corr, pixel_labeled

def MeanPixelAccuracy(gen_mask, input_mask):    
    pixel_acc = np.empty(input_mask.shape[0])
    pixel_corr = np.empty(input_mask.shape[0])
    pixel_labeled = np.empty(input_mask.shape[0])

    for i in range(input_mask.shape[0]):
        pixel_acc[i], pixel_corr[i], pixel_labeled[i] = \
        pixelAccuracy(gen_mask[i], input_mask[i])

    acc = 100.0 * np.sum(pixel_corr) / (np.spacing(1) + np.sum(pixel_labeled))

    return acc

def intersectionAndUnion(gen_mask, input_mask, numClass=8):
    gen_mask = gen_mask * (input_mask > 0).long()
    intersection = gen_mask * (gen_mask == input_mask).long()
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    (area_pred, _) = np.histogram(gen_mask, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(input_mask, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    IoU = area_intersection / (area_union + 1e-10)

    return IoU, area_intersection, area_union

def mIoU(gen_mask, input_mask):
    area_intersection = []
    area_union = []

    for i in range(input_mask.shape[0]):
        _, intersection, union = intersectionAndUnion(gen_mask[i], input_mask[i])
        area_intersection.append(intersection)
        area_union.append(union)

    IoU = 1.0 * np.sum(area_intersection, axis=0) / np.sum(np.spacing(1)+area_union, axis=0)

    return np.mean(IoU)

EPS = 1e-10
def nanmean(x):
    return torch.mean(x[x == x])

def _fast_hist(true, pred, num_classes):
    true = true.long()
    pred = pred.long()
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist

def per_class_pixel_accuracy(hist):
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc, per_class_acc

def per_class_PA (pred, label):
    hist = _fast_hist(pred, label,8)
    avg, per = per_class_pixel_accuracy(hist)
    return avg, per

def print_all_metrics(gen_mask, input_mask):
    acc,cor,lab = pixelAccuracy(gen_mask, input_mask)
    acc = acc.item()
    m_acc = MeanPixelAccuracy(gen_mask, input_mask)
    IOU,_,_ = intersectionAndUnion(gen_mask, input_mask)
    MIOU = mIoU(gen_mask, input_mask)
    avg, per = per_class_PA(gen_mask, input_mask)
    DICE = multiclass_dice_coeff(gen_mask, input_mask)
    DICE = DICE.item()
    
    print("===========>Evaluate by Metrics:")
    print(f"Pixel Accuracy: {acc},\t\
        class Pixel Accuracy： {per},\t\
            mPA: {m_acc},\t\
                Dice: {DICE},\t\
                    IoU: {IOU},\t\
                        mIoU: {MIOU}")