from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Union, Tuple

import torch
import gpytorch
import warnings

from torch import Tensor
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing, global_mean_pool
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from ocpmodels.common.registry import registry

warnings.filterwarnings("ignore")


@registry.register_model("MLP")
class MLP(torch.nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 16, output_dim: int = 1):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)


@registry.register_model("GP_Exact")
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
@registry.register_model("GP_Tanimoto")
class TanimotoGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(TanimotoGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

@registry.register_model("GP_ExactMordred")
class ExactMordredGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactMordredGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar += torch.eye(x.size(0)) * 1e-3
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MolEncoder(metaclass=ABCMeta):
    _device = torch.device("cpu")

    def encode(self, x) -> Tensor:
        return x

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        raise NotImplementedError

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if not isinstance(device, torch.device):
            raise TypeError("Device must be a torch.device object")
        self._device = device
    
    def to(self, device: Union[str, torch.device]) -> MolEncoder:
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        return self


class LearnableMolEncoder(MolEncoder, torch.nn.Module, metaclass=ABCMeta):
    def forward(self, x) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        h = self._forward_pass(x)

        mu = self._mu(h)

        if self.variational:
            log_var = self._log_var(h)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, log_var

        else:
            return mu

    @abstractmethod
    def _forward_pass(self, x) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode(self, x) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        raise NotImplementedError

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise TypeError("Device must be a torch.device object")
        self._device = device
        torch.nn.Module.to(self, device)


@registry.register_model("MPNN_Benchmark")
class GNNEncoder(LearnableMolEncoder):
    def __init__(
            self,
            node_attributes: int = 5,
            edge_attributes: int = 4,
            hidden_size: int = 280,
            num_layers: int = 3,
            latent_dim: int = 64,
            variational: bool = False,
            pooling_type = global_mean_pool,
            learning_rate: float = 1e-3,
            **kwargs
    ):
        torch.nn.Module.__init__(self)

        self._node_dim = node_attributes
        self._edge_dim = edge_attributes

        # Settings regarding the GNN architecture

        self._hidden_size = hidden_size
        self._input_layer = torch.nn.Linear(node_attributes, hidden_size)
        self._message_passing = self._initialize_message_passing(num_layers)
        self._pooling = pooling_type
        self._latent_dim = latent_dim
        self._mu = torch.nn.Linear(hidden_size, latent_dim)
        self.variational = variational

        if variational:
            self._log_var = torch.nn.Linear(hidden_size, latent_dim)

        self.learning_rate = learning_rate

        for kwarg, val in kwargs.items():
            setattr(self, kwarg, val)

    def _forward_pass(self, x) -> Tensor:
        h = self._input_layer(x.x.float())

        for layer in self._message_passing:
            h = h + layer(h, x.edge_index, x.edge_attr)

        return self._pooling(h, x.batch)

    def _initialize_message_passing(
            self,
            num_layers: int
    ) -> torch.nn.ModuleList:
        layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            layers.append(MPNNLayer(hidden_dim=self._hidden_size, edge_dim=self._edge_dim))
        return layers

    def encode(self, x) -> Tensor:
        return self(x)

    @property
    def feature_dim(self) -> int:
        return self._latent_dim

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise TypeError("Device must be a torch.device object")

        self._device = device
        for layer in self._message_passing:
            layer.device = device
        torch.nn.Module.to(self, device)


class MPNNLayer(MessagePassing):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr="add")

        self._hidden_dim = hidden_dim
        self._edge_dim = edge_dim

        self._message_network = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU()
        )

        self._update_network = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU()
        )

        self._device = torch.device("cpu")

    def forward(self, h, edge_index, edge_attr):
        return self.propagate(edge_index, h=h, edge_attr=edge_attr)

    def message(self, h_i: Tensor, h_j: Tensor, edge_attr: Tensor) -> Tensor:
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self._message_network(msg)

    def aggregate(self, inputs: Tensor, index: Tensor) -> Tensor:
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out: Tensor, h: Tensor):
        upd = torch.cat([h, aggr_out], dim=-1)
        return self._update_network(upd)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise TypeError("Device must be a torch.device object")

        self._device = device
        self._message_network.to(device)
        self._update_network.to(device)
        torch.nn.Module.to(self, device)
