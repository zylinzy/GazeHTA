"""
Utilities for bounding box manipulation and GIoU.
Copy-paste from DETR's official implementation with minor modifications.
"""
import torch
from torchvision.ops.boxes import box_area
import numpy as np
import cv2

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray

def points_inside_boxes(orig_points, bboxes):
    points = orig_points.unsqueeze(1).repeat(1, len(bboxes), 1)

    if points.shape[1] != bboxes.shape[0]:
        breakpoint()

    x_min_cond = points[:, :, 0] >= bboxes[:, 0]
    y_min_cond = points[:, :, 1] >= bboxes[:, 1]
    x_max_cond = points[:, :, 0] <= bboxes[:, 2]
    y_max_cond = points[:, :, 1] <= bboxes[:, 3]

    # If point is inside bbox, then sum of above conditions will be 4
    return (
        x_min_cond.int() + y_min_cond.int() + x_max_cond.int() + y_max_cond.int()
    ) == 4


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [n,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [n,M,2]

    wh = (rb - lt).clamp(min=0)  # [n,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [n,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The target should be in [x0, y0, x1, y1] format

    Returns a [n, M] pairwise matrix, where n = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate target gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [n,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def get_head_labelmap(img, pt, sigma, pdf='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma[0]), int(pt[1] - 3 * sigma[1])]
    br = [int(pt[0] + 3 * sigma[0] + 1), int(pt[1] + 3 * sigma[1] + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    sigma_x = sigma[0].numpy()
    sigma_y = sigma[1].numpy()
    
    size = 6 * sigma_x + 1
    x = np.arange(0, size, 1, float)[None]
    x0 = size // 2

    size = 6 * sigma_y + 1
    y = np.arange(0, size, 1, float)[..., None]
    y0 = size // 2

    # The gaussian is not normalized, we want the center value to equal 1
    if pdf == 'Gaussian':
        g = np.exp(-((x - x0)**2/(2*sigma_x**2)
             + (y - y0)**2/(2*sigma_y**2))) / (2*sigma_x*sigma_y)
    elif pdf == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    
    if np.max(img) != 0:
        img = img/np.max(img) # normalize heatmap so it has max value of 1
    return to_torch(img)


def get_box_from_heatmap(head_heatmap):
    # Grayscale then Otsu's threshold
    head_heatmap_ = (head_heatmap.cpu().data.numpy() * 255).astype(np.uint8)
    thresh = cv2.threshold(head_heatmap_, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    best_box = None
    best_sum = -999
    found = False
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        curr_sum = head_heatmap_[y:y+h, x:x+w].sum()
        if curr_sum > best_sum:
            best_box = np.array([[x, y, x+w, y+h]]).astype(np.float32)
            best_sum = curr_sum
            found = True

    if found == False:
        heatmap_size = head_heatmap_.shape[-1]
        best_box = np.array([[0, 0, heatmap_size, heatmap_size]]).astype(np.float32)

    return to_torch(best_box)

def get_box_from_heatmap_given_peak(head_heatmap, peak):
    # Grayscale then Otsu's threshold
    head_heatmap_ = (head_heatmap.cpu().data.numpy() * 255).astype(np.uint8)
    thresh = cv2.threshold(head_heatmap_, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    best_box = None
    best_sum = -999
    found = False
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        box = np.array([[x, y, x+w-1, y+h-1]]).astype(np.float32)
        is_inbox = points_inside_boxes(peak.unsqueeze(0), torch.from_numpy(box))
        if is_inbox.sum() != 0:
            curr_sum = head_heatmap_[y:y+h-1, x:x+w-1].sum()
            if curr_sum > best_sum:
                best_box = box
                best_sum = curr_sum
                found = True

    if found == False:
        heatmap_size = head_heatmap_.shape[-1]
        best_box = np.array([[0, 0, heatmap_size, heatmap_size]]).astype(np.float32)

    return to_torch(best_box)

from scipy.ndimage.filters import gaussian_filter
def find_peaks_from_heatmap(head_heatmaps, threshold=0.1, sigma = 1.0):
    thre1 = threshold # heatmap peak identifier threshold    
    all_peaks = []
    #peak_counter = 0 
    for map_ori in head_heatmaps:
        one_heatmap = gaussian_filter(map_ori, sigma=sigma) # smooth the data
        map_left = np.zeros(one_heatmap.shape)
        map_left[1:, :] = one_heatmap[:-1, :]
        map_right = np.zeros(one_heatmap.shape)
        map_right[:-1, :] = one_heatmap[1:, :]
        map_up = np.zeros(one_heatmap.shape)
        map_up[:, 1:] = one_heatmap[:, :-1]
        map_down = np.zeros(one_heatmap.shape)
        map_down[:, :-1] = one_heatmap[:, 1:]
    
        peaks_binary = np.logical_and.reduce(
            (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
        # find the peak of surrounding with window size = 1 and above threshold
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        all_peaks.append(peaks_with_score)
        
    return all_peaks


def get_boxes_from_heatmap(head_heatmap):
    # Grayscale then Otsu's threshold
    head_heatmap_ = (head_heatmap.cpu().data.numpy() * 255).astype(np.uint8)
    thresh = cv2.threshold(head_heatmap_, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        boxes.append(np.array([x, y, x+w, y+h]).astype(np.float32))

    boxes = to_torch(np.stack(boxes))

    return boxes

def get_box_from_heatmap_max(head_heatmap):
    # Grayscale then Otsu's threshold
    head_heatmap_ = (head_heatmap.cpu().data.numpy() * 255).astype(np.uint8)
    thresh = cv2.threshold(head_heatmap_, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    best_box = None
    best_sum = -999
    found = False
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if head_heatmap_[y:y+h-1, x:x+w-1].numel() != 0:
            curr_sum = head_heatmap_[y:y+h-1, x:x+w-1].max()
            if curr_sum > best_sum:
                best_box = np.array([[x, y, x+w-1, y+h-1]]).astype(np.float32)
                best_sum = curr_sum
                found = True

    if found == False:
        heatmap_size = head_heatmap_.shape[-1]
        best_box = np.array([[0, 0, heatmap_size, heatmap_size]]).astype(np.float32)

    return to_torch(best_box)
    