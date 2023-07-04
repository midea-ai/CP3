from abc import ABC
from collections import Callable
from typing import Iterable

import torch.nn as nn
import torch

from torch import Tensor
from typing import Dict

global feature_result
global entropy
global total


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            print('-------layer---------------')
            print(layer_id)
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))
            layer.register_forward_hook(self.print_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
            print('-----output-------')
            print(output)

        return fn

    def print_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            print(f"{layer_id}: {output.shape}")

        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        output = self.model(x)
        return output


class RankExtractor(nn.Module, ABC):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self.rank = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))
            layer.register_forward_hook(self.print_outputs_hook(layer_id))
            # self.save_outputs_hook(layer_id)

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            if len(output.shape) == 3:
                output = output.unsqueeze(-1)
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)
            a = output.shape[0]
            b = output.shape[1]
            c = torch.tensor([torch.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b)])

            c = c.view(a, -1).float()
            c = c.sum(0)
            feature_result = feature_result * total + c
            total = total + a
            feature_result = feature_result / total

            self.rank[layer_id] = feature_result

        return fn

    def print_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            print(f"{layer_id}: {output.shape}")

        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        output = self.model(x)
        return output
