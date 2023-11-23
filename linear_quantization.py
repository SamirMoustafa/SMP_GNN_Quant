import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearQuantized(nn.Linear):
    """A quantizable linear layer"""

    def __init__(
            self, in_features, out_features, layer_quantizers, initial_linear_gamma, bias=True):
        self.layer_quant = layer_quantizers
        super(LinearQuantized, self).__init__(in_features, out_features, bias)

        alpha_inputs = torch.tensor([initial_linear_gamma], requires_grad=True)
        self.alpha_inputs = torch.nn.Parameter(alpha_inputs)
        alpha_weights = torch.tensor([initial_linear_gamma], requires_grad=True)
        self.alpha_weights = torch.nn.Parameter(alpha_weights)
        alpha_features = torch.tensor([initial_linear_gamma], requires_grad=True)
        self.alpha_features = torch.nn.Parameter(alpha_features)

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, input, training=None):

        input_q, _, _ = self.layer_quant["inputs"](input, custom_alpha=self.alpha_inputs, training=training)
        w_q, _, _ = self.layer_quant["weights"](self.weight, custom_alpha=self.alpha_weights, training=training)

        if type(input_q) == tuple:
            input_q = input_q[0]
        if type(w_q) == tuple:
            w_q = w_q[0]

        out = F.linear(input_q, w_q, self.bias)
        out = self.layer_quant["features"](out, custom_alpha=self.alpha_features, training=training)

        return out
