# This code is based on https://github.com/openai/guided-diffusion
"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def geodesic_distance(q1, q2):
    """
    Compute geodesic distance between two quaternions on SO(3).
    q1, q2: tensors of shape (batch_size, num_joints, 4)
    Returns: tensor of shape (batch_size, num_joints) with geodesic distances
    """
    # Normalize quaternions
    q1 = qnormalize(q1)
    q2 = qnormalize(q2)
    
    # Compute relative quaternion: q1 * qinv(q2)
    q_rel = qmul(q1, qinv(q2))
    
    # Extract scalar part (w component)
    w = q_rel[..., 0]
    
    # Geodesic distance = 2 * arccos(|w|)
    # Clamp w to [-1, 1] for numerical stability
    w = th.clamp(w, -1.0, 1.0)
    return 2 * th.acos(th.abs(w))


def rot6d_to_quaternion(rot6d):
    """
    Convert 6D rotation representation to quaternion.
    rot6d: tensor of shape (..., 6) where ... can be any batch dimensions
    Returns: tensor of shape (..., 4) quaternion
    """
    # Extract x and y vectors from 6D representation
    x_raw = rot6d[..., 0:3]
    y_raw = rot6d[..., 3:6]
    
    # Normalize x
    x = x_raw / th.norm(x_raw, dim=-1, keepdim=True)
    
    # Orthogonalize y with respect to x
    y = y_raw - th.sum(y_raw * x, dim=-1, keepdim=True) * x
    y = y / th.norm(y, dim=-1, keepdim=True)
    
    # Compute z as cross product of x and y
    z = th.cross(x, y, dim=-1)
    
    # Construct rotation matrix: [x, y, z] as columns
    # Stack to get [..., 3, 3]
    R = th.stack([x, y, z], dim=-1)
    
    # Convert rotation matrix to quaternion using Shepperd's method
    batch_shape = R.shape[:-2]  # All dimensions except last 2
    mat_flat = R.reshape(-1, 3, 3)  # [batch_product, 3, 3]
    
    trace = mat_flat[:, 0, 0] + mat_flat[:, 1, 1] + mat_flat[:, 2, 2]  # [batch_product]
    
    # Initialize quaternion tensor
    q = th.zeros(mat_flat.shape[0], 4, dtype=R.dtype, device=R.device)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = 0.5 / th.sqrt(trace[mask1] + 1.0)
    q[mask1, 0] = 0.25 / s1
    q[mask1, 1] = (mat_flat[mask1, 2, 1] - mat_flat[mask1, 1, 2]) * s1
    q[mask1, 2] = (mat_flat[mask1, 0, 2] - mat_flat[mask1, 2, 0]) * s1
    q[mask1, 3] = (mat_flat[mask1, 1, 0] - mat_flat[mask1, 0, 1]) * s1
    
    # Case 2: trace <= 0 (trace is smallest)
    mask2 = (trace <= 0) & (mat_flat[:, 0, 0] >= mat_flat[:, 1, 1]) & (mat_flat[:, 0, 0] >= mat_flat[:, 2, 2])
    s2 = 2.0 * th.sqrt(1.0 + mat_flat[mask2, 0, 0] - mat_flat[mask2, 1, 1] - mat_flat[mask2, 2, 2])
    q[mask2, 0] = (mat_flat[mask2, 2, 1] - mat_flat[mask2, 1, 2]) / s2
    q[mask2, 1] = 0.25 * s2
    q[mask2, 2] = (mat_flat[mask2, 0, 1] + mat_flat[mask2, 1, 0]) / s2
    q[mask2, 3] = (mat_flat[mask2, 0, 2] + mat_flat[mask2, 2, 0]) / s2
    
    # Case 3: m[1,1] is largest
    mask3 = (trace <= 0) & (mat_flat[:, 1, 1] >= mat_flat[:, 2, 2]) & ~mask2
    s3 = 2.0 * th.sqrt(1.0 + mat_flat[mask3, 1, 1] - mat_flat[mask3, 0, 0] - mat_flat[mask3, 2, 2])
    q[mask3, 0] = (mat_flat[mask3, 0, 2] - mat_flat[mask3, 2, 0]) / s3
    q[mask3, 1] = (mat_flat[mask3, 0, 1] + mat_flat[mask3, 1, 0]) / s3
    q[mask3, 2] = 0.25 * s3
    q[mask3, 3] = (mat_flat[mask3, 1, 2] + mat_flat[mask3, 2, 1]) / s3
    
    # Case 4: m[2,2] is largest
    mask4 = (trace <= 0) & ~mask2 & ~mask3
    s4 = 2.0 * th.sqrt(1.0 + mat_flat[mask4, 2, 2] - mat_flat[mask4, 0, 0] - mat_flat[mask4, 1, 1])
    q[mask4, 0] = (mat_flat[mask4, 1, 0] - mat_flat[mask4, 0, 1]) / s4
    q[mask4, 1] = (mat_flat[mask4, 0, 2] + mat_flat[mask4, 2, 0]) / s4
    q[mask4, 2] = (mat_flat[mask4, 1, 2] + mat_flat[mask4, 2, 1]) / s4
    q[mask4, 3] = 0.25 * s4
    
    # Reshape back to original batch shape
    q_shape = list(batch_shape) + [4]
    q = q.reshape(q_shape)
    
    return q


def qnormalize(q):
    """
    Normalize quaternion.
    q: tensor of shape (*, 4)
    Returns: normalized quaternion
    """
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    q[..., -1] += 1e-4  # Guy - for safety, avoid zero division
    return q / th.norm(q, dim=-1, keepdim=True)


def qinv(q):
    """
    Invert quaternion.
    q: tensor of shape (*, 4)
    Returns: inverted quaternion
    """
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = th.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Flatten all dimensions except the last (quaternion components)
    q_flat = q.reshape(-1, 4)  # [batch_product, 4]
    r_flat = r.reshape(-1, 4)  # [batch_product, 4]

    # Compute outer product via bmm: [batch, 4, 1] @ [batch, 1, 4] = [batch, 4, 4]
    terms = th.bmm(r_flat.unsqueeze(2), q_flat.unsqueeze(1))  # [batch_product, 4, 4]

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    
    result = th.stack((w, x, y, z), dim=1)  # [batch_product, 4]
    return result.view(original_shape)
