import matplotlib.pyplot as plt
import numpy as np
import torch


def decode_seg_map_sequence(label_masks, dataset="pascal"):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == "pascal":
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == "ade":
        n_classes = 151
        label_colours = get_ade_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
        
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def get_pascal_labels():
    """
    Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
    )


def get_ade_labels():
    return np.asarray(
        [
            [0, 0, 0],
            [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
            [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
            [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
            [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
            [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
            [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10],
            [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
            [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15],
            [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0],
            [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112],
            [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0],
            [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173],
            [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184],
            [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194],
            [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255],
            [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170],
            [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255],
            [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
            [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235],
            [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
            [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255],
            [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0],
            [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255]
        ]
    )
