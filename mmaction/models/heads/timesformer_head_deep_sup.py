# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model.weight_init import trunc_normal_init
import torch
from torch import Tensor, nn
from typing import List, Tuple

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead


@MODELS.register_module()
class TimeSformerHeadDeep(BaseHead):
    """Classification head for TimeSformer.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        init_std (float): Std value for Initiation. Defaults to 0.02.
        dropout_ratio (float): Probability of dropout layer.
            Defaults to : 0.0.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 init_std: float = 0.02,
                 dropout_ratio: float = 0.0,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.dropout_ratio = dropout_ratio

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc_cls = nn.ModuleList([
            nn.Linear(in_channels, num_classes) for _ in range(4)
        ])

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        for fc in self.fc_cls:
            trunc_normal_init(fc, std=self.init_std)

    def forward(self, x: Tuple[Tensor, List[Tensor]], **kwargs) -> Tuple[Tensor, ...]:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
    
        last_feat, inter_feats = x
        if self.dropout is not None:
            last_feat = self.dropout(last_feat)
            inter_feats = [self.dropout(f) for f in inter_feats]

        scores = []
        for head, feat in zip(self.fc_cls, [last_feat] + inter_feats):
            scores.append(head(feat))
        return tuple(scores)
    
    def loss_by_feat(self, cls_scores, data_samples):
        # cls_scores: tuple of [main_score, aux1_score, aux2_score, aux3_score]
        device = cls_scores[0].device
        labels = torch.stack([s.gt_label for s in data_samples]).to(device)
        # labels = labels.squeeze()
        labels = labels.squeeze(-1)

        total_loss = 0
        aux_losses = 0
        losses = {}

        main_loss = self.loss_cls(cls_scores[0], labels)
        losses['loss_main'] = main_loss
        total_loss += (main_loss / 2)

        for i, score in enumerate(cls_scores[1:], start=1):
            key = f'loss_aux{i}'
            loss_i = self.loss_cls(score, labels)
            losses[key] = loss_i
            aux_losses += loss_i
        
        aux_losses = aux_losses / 3
        total_loss += (aux_losses / 2)

        losses['loss'] = total_loss
        return losses

    def predict_by_feat(self, cls_scores, data_samples):
        return super().predict_by_feat(cls_scores[0], data_samples)

