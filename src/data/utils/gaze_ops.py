import math
import numpy as np
import torch

#from utils import to_numpy, to_torch

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


def get_label_map(img, pt, sigma, pdf="Gaussian"):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [
        pt[0].round().int().item() - 3 * sigma,
        pt[1].round().int().item() - 3 * sigma,
    ]
    br = [
        pt[0].round().int().item() + 3 * sigma + 1,
        pt[1].round().int().item() + 3 * sigma + 1,
    ]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if pdf == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif pdf == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    if np.max(img) != 0:
        img = img / np.max(img)  # normalize heatmap so it has max value of 1

    return to_torch(img)

def get_label_line_map(img, pts, sigmas, scalars, pdf='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)
    
    for pt, sigma, scalar in zip(pts, sigmas, scalars):

        # Check that any part of the gaussian is in-bounds
        ul = [int(pt[0] - 3 * sigma[0]), int(pt[1] - 3 * sigma[1])]
        br = [ul[0] + int(6 * sigma[0] + 1), ul[1]+ int(6 * sigma[1] + 1)]
        if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
                br[0] < 0 or br[1] < 0):
            # If not, just return the image as is
            return to_torch(img)

        # Generate gaussian
        sigma_x = sigma[0].numpy()
        sigma_y = sigma[1].numpy()
        
        size = int(6 * sigma_x + 1)
        x = np.arange(0, size, 1, float)[None]
        x0 = size // 2
        
        size = int(6 * sigma_y + 1)
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
        
        
        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += scalar * g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    if np.max(img) != 0:
        img = img/np.max(img) # normalize heatmap so it has max value of 1
    return to_torch(img)

def get_multi_hot_map(gaze_pts, out_res, device=torch.device("cuda")):
    h, w = out_res
    target_map = torch.zeros((h, w), device=device).long()
    for p in gaze_pts:
        if p[0] >= 0:
            x, y = map(int, [p[0] * float(w), p[1] * float(h)])
            x = min(x, w - 1)
            y = min(y, h - 1)
            target_map[y, x] = 1

    return target_map


def get_heatmap_peak_coords(heatmap):
    np_heatmap = to_numpy(heatmap)
    idx = np.unravel_index(np_heatmap.argmax(), np_heatmap.shape)
    pred_y, pred_x = map(float, idx)

    return pred_x, pred_y


def get_l2_dist(p1, p2):
    return torch.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)


def rescale_heatmap(heatmap):
    heatmap -= heatmap.min(1, keepdim=True)[0]
    heatmap /= heatmap.max(1, keepdim=True)[0]

    return heatmap