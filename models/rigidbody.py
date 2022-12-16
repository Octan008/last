# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=invalid-name
# pytype: disable=attribute-error
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#全体的に次元先頭でいきます

def skew(w: torch.tensor) -> torch.tensor:
    """Build a skew matrix ("cross product matrix") for vector w.

    Modern Robotics Eqn 3.30.

    Args:
    w: (3,) A 3-vector

    Returns:
    W: (3, 3) A skew matrix such that W @ v == w x v

    for pytorch
    (3, n) -> (3, 3, n)
    """
    #   w = jnp.reshape(w, (3))
    # return jnp.array([[0.0, -w[2], w[1]], \
    #                 [w[2], 0.0, -w[0]], \
    #                 [-w[1], w[0], 0.0]])
    return torch.stack([
        torch.stack([torch.zeros_like(w[0]), -w[2], w[1]], dim=0),
        torch.stack([w[2], torch.zeros_like(w[0]), -w[0]], dim=0),
        torch.stack([-w[1], w[0], torch.zeros_like(w[0])], dim=0)], dim=0)


def rp_to_se3(R: torch.tensor, p: torch.tensor) -> torch.tensor:
    """Rotation and translation to homogeneous transform.

    Args:
    R: (3, 3) An orthonormal rotation matrix.
    p: (3,) A 3-vector representing an offset.

    Returns:
    X: (4, 4) The homogeneous transformation matrix described by rotating by R
        and translating by p.

    for pytorch
    (3,3,n), (3,n) -> (4,4,n)
    """
    # p = jnp.reshape(p, (3, 1))
    # return jnp.block([[R, p], [jnp.array([[0.0, 0.0, 0.0, 1.0]])]])
    return torch.cat([
        torch.cat([R, p.unsqueeze(1)], dim=1),
        torch.cat([torch.zeros_like(R[:1, :3]), torch.ones_like(R[:1, :1])], dim=1)], dim=0)



def exp_so3(w: torch.tensor, theta: torch.tensor) -> torch.tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

    Args:
    w: (3,) An axis of rotation.
    theta: An angle of rotation.

    Returns:
    R: (3, 3) An orthonormal rotation matrix representing a rotation of
        magnitude theta about axis w.

    for pytorch
    (3, n), (1, n) -> (3,3,n)
    """
    W = skew(w)
    W2 =( W.permute(2,0,1) @ W.permute(2,0,1)).permute(1,2,0)
    tmp = torch.eye(3).to(w.device).unsqueeze(-1).expand( -1, -1, theta.shape[-1]) + torch.sin(theta) * W + (1.0 - torch.cos(theta)) * W2
    # tmp = tmp.permute(2,0,1) @ W.permute(2,0,1)
    return tmp
    # return torch.eye(3).to(w.device).unsqueeze(-1).expand( -1, -1, theta.shape[-1]) + torch.sin(theta) * W + (1.0 - torch.cos(theta)) * W @ W


def exp_se3(S: torch.tensor, theta: torch.tensor) -> torch.tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.88.

    Args:
    S: (6,) A screw axis of motion.
    theta: Magnitude of motion.

    Returns:
    a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
        motion of magnitude theta about S for one second.

    for pytorch
    (6, n), (1, n) -> (4,4, n)
    """
    w, v = S[:3], S[3:]
    W = skew(w)
    W2 = (W.permute(2,0,1) @ W.permute(2,0,1)).permute(1,2,0)
    R = exp_so3(w, theta)
    tmp = theta * torch.eye(3).to(w.device).unsqueeze(-1).expand(-1, -1, theta.shape[-1]) + (1.0 - torch.cos(theta)) * W + (theta - torch.sin(theta)) * W2
    # tmp = tmp.permute(2,0,1) @ W.permute(2,0,1)
    tmp = tmp.permute(2,0,1)
    p =  tmp @ v.permute(1,0).unsqueeze(-1)
    p = p.squeeze().permute(1,0)
    return rp_to_se3(R, p)


def to_homogenous(v):#合ってる？
  return torch.cat([v, torch.ones_like(v[..., :1])], axis=-1)


def from_homogenous(v):
  return v[..., :3] / v[..., -1:]