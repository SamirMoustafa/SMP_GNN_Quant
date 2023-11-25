from typing import Optional

import torch.nn.functional as F
from torch.nn import Parameter
from torch import (Tensor, arange, cat, ones, eye, tensor, zeros, nonzero, clamp, norm,
                   matmul as torch_matmul, abs as torch_abs, sum as torch_sum)

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul, fill_diag, sum as torch_sparse_sum, mul as torch_sparse_mul

from basic_message_passing_layer import MessagePassingQuant


def get_inc(edge_index):
    size = edge_index.sizes()[1]
    row_index = edge_index.storage.row()
    col_index = edge_index.storage.col()
    mask = row_index >= col_index  # remove duplicate edge and self loop
    row_index = row_index[mask]
    col_index = col_index[mask]
    edge_num = row_index.numel()
    row = cat([arange(edge_num), arange(edge_num)]).to(edge_index.device())
    col = cat([row_index, col_index]).to(edge_index.device())
    value = cat([ones(edge_num), -1 * ones(edge_num)]).to(edge_index.device())
    return SparseTensor(row=row, rowptr=None, col=col, value=value, sparse_sizes=(edge_num, size))


def inc_norm(inc, edge_index):
    edge_index = fill_diag(edge_index, 1.0)  ## add self loop to avoid 0 degree node
    deg = torch_sparse_sum(edge_index, dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    inc = torch_sparse_mul(inc, deg_inv_sqrt.view(1, -1))  ## col-wise
    return inc


def check_inc(edge_index, inc):
    nnz = edge_index.nnz()
    deg = eye(edge_index.sizes()[0]).to(edge_index.device())
    adj = edge_index.to_dense()
    lap = (inc.t() @ inc).to_dense()
    lap2 = deg - adj
    diff = torch_sum(torch_abs(lap2 - lap)) / nnz
    assert diff < 0.000001, f'error: {diff} need to make sure L=B^TB'


class prop_part_QUANT(MessagePassingQuant):
    _cached_adj_t: Optional[SparseTensor]
    _cached_inc = Optional[SparseTensor]

    def __init__(self,
                 number_of_layers: int,
                 lambda1: float = None,
                 lambda2: float = None,
                 L21: bool = True,
                 dropout: float = 0,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 original_edge_index=None,
                 prop_mode=None,
                 qtype=False,
                 message_group_quantizers=None,
                 eta_lambda=None,
                 alpha_threshold=None,
                 final_out_quantizer=None,
                 embedding_quant=False,
                 **kwargs):
        super(prop_part_QUANT, self).__init__(aggr='add', message_group_quantizers=message_group_quantizers, **kwargs)
        assert add_self_loops == True and normalize == True, "add_self_loops and normalize should be True"
        self.number_of_layers = number_of_layers
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.L21 = L21
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_adj_t = None
        self._cached_inc = None

        self.prop_mode = prop_mode
        self.original_edge_index = original_edge_index

        self.eta_lambda = eta_lambda
        self.alpha_threshold = alpha_threshold

        self.qtype = qtype

        self.out_quantizer = final_out_quantizer

        self.alpha_out = Parameter(tensor([1.0], requires_grad=True))
        self.embedding_quant = embedding_quant  # True: store quantized embedding; False: general quantization for node-classification

    def reset_parameters(self):
        self._cached_adj_t = None
        self._cached_inc = None

    def forward(self, x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                ) -> Tensor:
        """"""
        if self.number_of_layers <= 0:
            return x

        assert isinstance(edge_index, SparseTensor), "Only support SparseTensor now"
        assert edge_weight is None, "edge_weight is not supported yet, but it can be extended to weighted case"
        if self.normalize:
            cache = self._cached_inc
            if cache is None:
                inc_mat = get_inc(edge_index=edge_index)
                inc_mat = inc_norm(inc=inc_mat, edge_index=edge_index)
                if self.cached:
                    self._cached_inc = inc_mat
                    self.init_z = zeros((inc_mat.sizes()[0], x.size()[-1])).to(x.device)
            else:
                inc_mat = self._cached_inc

            cache = self._cached_adj_t
            if cache is None:
                edge_index = gcn_norm(edge_index,
                                      edge_weight,
                                      x.size(self.node_dim),
                                      False,
                                      add_self_loops=self.add_self_loops,
                                      dtype=x.dtype)

                if x.size()[0] < 30000:
                    check_inc(edge_index=edge_index, inc=inc_mat)

                if self.cached:
                    self._cached_adj_t = edge_index
            else:
                edge_index = cache

        dense_edge_index = edge_index.to_dense()
        self.edge_weight_transf = dense_edge_index[self.original_edge_index[0], self.original_edge_index[1]]

        hh = x
        if self.prop_mode == "EMP":
            x = self.EMP_forward(x=x, hh=hh, edge_index=edge_index, inc=inc_mat, number_of_layers=self.number_of_layers)

        if self.prop_mode == "SMP":
            x = self.SMP_forward(x=x, hh=hh, edge_index=edge_index, inc=inc_mat, number_of_layers=self.number_of_layers)
        return x

    def SMP_forward(self, x, hh, number_of_layers, edge_index, inc):
        lambda_list = ones([x.shape[0], 1], device=x.device) * 0.001
        s1_list = ones([x.shape[0], 1], device=x.device) * 0.001

        dense_edge_index = edge_index.to_dense()

        eta_H = 0.1
        eta_s1 = 0.00001
        for k in range(number_of_layers):
            layer_rep = (1 - (1 + self.lambda1) * eta_H) * x

            layer_rep_prop, _, _ = self.propagate(self.original_edge_index,
                                                  x=x,
                                                  edge_weight=self.edge_weight_transf,
                                                  size=None,
                                                  k=k,
                                                  name="prop")

            sub_layer_rep = layer_rep + self.lambda1 * eta_H * layer_rep_prop + eta_H * hh

            gap_rep = (sub_layer_rep - x)

            gap_rep_prop, _, _ = self.propagate(self.original_edge_index,
                                                x=gap_rep,
                                                edge_weight=self.edge_weight_transf,
                                                size=None,
                                                k=k,
                                                name="gap_prop")

            y = sub_layer_rep + 2 * eta_H * lambda_list * gap_rep_prop
            s1_list = s1_list + 2 * eta_s1 * lambda_list * s1_list

            smooth_cal = matmul(inc, gap_rep)
            smooth_cal_trans = smooth_cal.t()
            gap_sum = torch_matmul(smooth_cal_trans, smooth_cal)
            trace_val_g = gap_sum.trace()

            edge_NUM = nonzero(dense_edge_index).shape
            num_edge = edge_NUM[0]

            lambda_list = lambda_list + self.eta_lambda * (self.alpha_threshold * num_edge - s1_list * s1_list - trace_val_g)

            x = y
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.embedding_quant:
            x = self.out_quantizer['out'](x, custom_alpha=self.alpha_out, training=self.training)
            return x[0]
        else:
            return x

    def EMP_forward(self, x, hh, number_of_layers, edge_index, inc):
        lambda1 = self.lambda1
        lambda2 = self.lambda2

        gamma = 1 / (1 + lambda2)
        beta = 1 / (2 * gamma)
        lam_2_num = 0
        lam_1_num = 0
        escape_num = 0

        l21_num = 0
        l1_num = 0

        if lambda1 > 0:
            z = self.init_z.detach()

        for k in range(number_of_layers):

            if lambda2 > 0:
                dense_edge_index = edge_index.to_dense()
                edge_weight_transf = dense_edge_index[self.original_edge_index[0], self.original_edge_index[0]]

                temp_prop_val, _, _ = self.propagate(self.original_edge_index,
                                                     x=x, edge_weight=edge_weight_transf,
                                                     size=None,
                                                     k=k, name="prop")

                y = gamma * hh + (1 - gamma) * temp_prop_val


            else:
                y = gamma * hh + (1 - gamma) * x
                lam_2_num = lam_2_num + 1

            if lambda1 > 0:
                lam_1_num = lam_1_num + 1
                x_bar = y - gamma * (inc.t() @ z)

                z_bar = z + beta * (inc @ x_bar)
                if self.L21:
                    z = self.L21_projection(z_bar, lambda_=lambda1)
                    l21_num = l21_num + 1
                else:
                    z = self.L1_projection(z_bar, lambda_=lambda1)
                    l1_num = l1_num + 1
            else:
                escape_num = escape_num + 1
            x = y

            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def L1_projection(self, x: Tensor, lambda_):
        # component-wise projection onto the l∞ ball of radius λ1.
        return clamp(x, min=-lambda_, max=lambda_)

    def L21_projection(self, x: Tensor, lambda_):
        # row-wise projection on the l2 ball of radius λ1.
        row_norm = norm(x, p=2, dim=1)
        scale = clamp(row_norm, max=lambda_)
        index = row_norm > 0
        scale[index] = scale[index] / row_norm[index]  # avoid to be devided by 0
        temp_x = scale.unsqueeze(1) * x
        return temp_x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:

        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, lambda1={}, lambda2={}, L21={})'.format(
            self.__class__.__name__, self.number_of_layers, self.lambda1, self.lambda2, self.L21)
