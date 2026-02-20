import numpy as np
import torch
import colorsys

def get_n_different_colors(n: int) -> np.ndarray:
    np.random.seed(0)
    return np.random.randint(0, 256, (n, 3), dtype=np.uint8)


def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)     # [0, 1)
    s = 0.5 + (id % 2) * 0.5          # 0.5 / 1.0
    l = 0.5

    rgb = np.zeros((3,), dtype=np.uint8)
    if id == 0:   # invalid region / background
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r * 255), int(g * 255), int(b * 255)
    return rgb


def ndc2Pixel(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def transformPoint4x4(point, matrix):
    """Transform a point by a 4x4 matrix.

    :param point: [N, 3] 3D points.
    :param matrix: [4, 4] matrix.
    :return: [N, 4] transformed homogeneous points.
    """
    point = torch.cat([point, torch.ones_like(point[:, :1])], dim=1)
    transformed = torch.matmul(point, matrix)
    return transformed


def convert_matched_mask(labels, masks):
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)
    labels_np = labels_np.reshape(-1).astype(np.int64)

    num_local = labels_np.shape[0]
    max_local = int(np.max(masks))

    assert num_local == max_local, (
        f"convert_matched_mask: labels.shape[0]={num_local}, "
        f"but max(masks)={max_local}"
    )

    lut = np.zeros(num_local + 1, dtype=np.uint16)
    lut[1:] = labels_np + 1 

    matched_mask = lut[masks]
    return matched_mask


def mask_id_to_binary_mask(mask_id):
    num_masks = int(np.max(mask_id))
    h, w = mask_id.shape
    binary_mask = np.zeros((num_masks, h, w), dtype=bool)
    for m_idx in range(1, num_masks + 1):
        binary_mask[m_idx - 1] = (mask_id == m_idx)
    return binary_mask
