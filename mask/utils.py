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
    """
    将局部 mask 的 id（1..K）映射到全局 id（1..N）：
      - labels: shape [K]，元素是 0-based 的全局 id（Tensor 或 np.ndarray）
      - masks : H x W，像素值为 0..K（0 为背景，1..K 为局部实例 id）

    返回:
      - matched_mask: H x W, uint16
        像素值为 0..N，其中 0=背景，1..N=全局 mask id
    """
    # labels 统一成 numpy，一维数组
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)
    labels_np = labels_np.reshape(-1).astype(np.int64)

    num_local = labels_np.shape[0]
    max_local = int(np.max(masks))

    # 本地实例数量要和 mask 中的最大 id 对上
    assert num_local == max_local, (
        f"convert_matched_mask: labels.shape[0]={num_local}, "
        f"but max(masks)={max_local}"
    )

    # 构造一个查表：index = 局部 id (0..K)，value = 全局 id (0..N)
    # 0 保持 0（背景），1..K 映射到 labels_np + 1
    # 用 uint16 支持 >255 个类别（最多 65535）
    lut = np.zeros(num_local + 1, dtype=np.uint16)
    lut[1:] = labels_np + 1  # 保留 0 给背景

    # 直接查表：H x W -> H x W
    matched_mask = lut[masks]
    return matched_mask  # 不要再转成 uint8 了！！！


def mask_id_to_binary_mask(mask_id):
    """
    将 H x W 的整型 label 图（0=背景，1..N=实例）转为 [N, H, W] 的 bool mask。
    支持 uint8/uint16/int32 等各种类型。
    """
    num_masks = int(np.max(mask_id))
    h, w = mask_id.shape
    binary_mask = np.zeros((num_masks, h, w), dtype=bool)
    for m_idx in range(1, num_masks + 1):
        binary_mask[m_idx - 1] = (mask_id == m_idx)
    return binary_mask
