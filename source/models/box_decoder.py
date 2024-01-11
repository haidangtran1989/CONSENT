import torch
import torch.nn as nn
import torch.nn.functional as functional
from typing import Tuple
from utilities.box_wrapper import BoxTensor, log1mexp, CenterSigmoidBoxTensor
from utilities.config import *


def _compute_gumbel_min_max(box1: BoxTensor, box2: BoxTensor) -> Tuple[torch.Tensor, torch.Tensor]:
    min_point = torch.stack([box1.z, box2.z])
    min_point = torch.max(GUMBEL_BETA * torch.logsumexp(min_point / GUMBEL_BETA, 0), torch.max(min_point, 0)[0])
    max_point = torch.stack([box1.Z, box2.Z])
    max_point = torch.min(-GUMBEL_BETA * torch.logsumexp(-max_point / GUMBEL_BETA, 0), torch.min(max_point, 0)[0])
    return min_point, max_point


def _compute_hard_min_max(box1: BoxTensor, box2: BoxTensor) -> Tuple[torch.Tensor, torch.Tensor]:
    min_point = torch.max(box1.z, box2.z)
    max_point = torch.min(box1.Z, box2.Z)
    return min_point, max_point


class BoxDecoder(nn.Module):
    def __init__(self):
        super(BoxDecoder, self).__init__()
        self.box_embeddings = nn.Embedding(NUMBER_OF_TYPES, BOX_DIMENSION * 2, padding_idx=None,
                max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
        self.box_embeddings.to(DEVICE)
        self.type_box = None

    def init_type_box(self):
        inputs = torch.arange(0, self.box_embeddings.num_embeddings, dtype=torch.int64,
                device=self.box_embeddings.weight.device).to(DEVICE)
        emb = self.box_embeddings(inputs)
        self.type_box = CenterSigmoidBoxTensor.from_split(emb)

    def init_weights(self):
        torch.nn.init.uniform_(self.box_embeddings.weight[..., :BOX_DIMENSION], -INIT_INTERVAL_CENTER, INIT_INTERVAL_CENTER)
        torch.nn.init.uniform_(self.box_embeddings.weight[..., BOX_DIMENSION:], INIT_INTERVAL_DELTA, INIT_INTERVAL_DELTA)

    def log_soft_volume(self, point1: torch.Tensor, point2: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(point1.dtype).tiny
        softplus_scale = torch.tensor(SOFTPLUS_SCALE)
        if GUMBEL_BETA <= 0.:
            return torch.sum(torch.log(functional.softplus(point2 - point1, beta=INV_SOFTPLUS_TEMP).clamp_min(eps)),
                    dim=-1) + torch.log(softplus_scale)
        else:
            return torch.sum(torch.log(functional.softplus(point2 - point1 - 2 * EULER_GAMMA * GUMBEL_BETA,
                    beta=INV_SOFTPLUS_TEMP).clamp_min(eps)), dim=-1) + torch.log(softplus_scale)

    def type_box_volume(self) -> torch.Tensor:
        inputs = torch.arange(0, self.box_embeddings.num_embeddings, dtype=torch.int64,
                device=self.box_embeddings.weight.device)
        emb = self.box_embeddings(inputs)
        type_box = CenterSigmoidBoxTensor.from_split(emb)
        vol = self.log_soft_volume(type_box.z, type_box.Z)
        return vol

    def get_pairwise_conditional_prob(self, type_x_ids: torch.Tensor, type_y_ids: torch.Tensor) -> torch.Tensor:
        inputs = torch.arange(0, self.box_embeddings.num_embeddings, dtype=torch.int64,
                device=self.box_embeddings.weight.device)
        emb = self.box_embeddings(inputs)
        type_x = emb[type_x_ids]
        type_y = emb[type_y_ids]
        type_x_box = CenterSigmoidBoxTensor.from_split(type_x)
        type_y_box = CenterSigmoidBoxTensor.from_split(type_y)
        min_point, max_point = _compute_gumbel_min_max(type_x_box, type_y_box)
        intersection_vol = self.log_soft_volume(min_point, max_point)
        y_vol = self.log_soft_volume(type_y_box.z, type_y_box.Z)
        conditional_prob = intersection_vol - y_vol
        return torch.cat([conditional_prob.unsqueeze(-1), log1mexp(conditional_prob).unsqueeze(-1)], dim=-1)

    def get_mention_context_similarity(self, box1, box2):
        size1 = box1.data.size()[0]
        size2 = box2.data.size()[0]
        min_point = torch.max(torch.stack([box1.z.unsqueeze(1).expand(-1, size2, -1),
                                           box2.z.unsqueeze(0).expand(size1, -1, -1)]), 0)[0]
        max_point = torch.min(torch.stack([box1.Z.unsqueeze(1).expand(-1, size2, -1),
                                           box2.Z.unsqueeze(0).expand(size1, -1, -1)]), 0)[0]
        # Compute the volume of the intersection
        intersection_volume = self.log_soft_volume(min_point, max_point)
        # Compute  the volume of the first mention context box
        volume1 = self.log_soft_volume(box1.z, box1.Z)
        # Returns log probability list
        log_probs = intersection_volume - volume1.unsqueeze(-1)
        # Clip values > 1 for numerical stability
        if (log_probs > 0.0).any():
            log_probs[log_probs > 0.0] = 0.0
        return torch.exp(log_probs)

    def forward(self, mention_context_box: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.get_mention_context_similarity(mention_context_box, self.type_box)
