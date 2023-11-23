import torch
import torch.nn as nn
from scipy.stats import kurtosis
from scipy.stats import skew
from torch.autograd.function import InplaceFunction


def get_qparams(max_val, min_val, num_bits, eps):
    max_val, min_val = float(max_val), float(min_val)
    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)

    qmin = 0
    qmax = 2.0 ** num_bits - 1
    if max_val == min_val:
        scale = 1.0
        zero_point = 0
    else:
        scale = (max_val - min_val) / float(qmax - qmin)
        scale = max(scale, eps)

        zero_point = qmin - round(min_val / scale)
        zero_point = max(qmin, zero_point)
        zero_point = min(qmax, zero_point)
        zero_point = zero_point

    return qmin, qmax, zero_point, scale


class LearnsQuantize(InplaceFunction):
    @classmethod
    def forward(
            cls, ctx, input, s_alpha, scale, zero_point, output_int, g, max_val, min_val, num_bits, eps, training,
            skew_val, BT_mode):

        output = input.clone()
        ctx.num_bits = num_bits
        if num_bits == 32:
            return output, None, None, None, None

        if s_alpha is not None:
            max_val = max_val * s_alpha
            min_val = min_val * s_alpha

        ctx.BT_mode = BT_mode

        if BT_mode != None:

            qmin, qmax, zero_point, scale = get_qparams(max_val, min_val, 8, eps)

            output_int = (output * (1 / scale) + zero_point).round().clip(qmin, qmax)

            if BT_mode == "BT":
                output_int = torch.round(output_int / 85) * 85
            elif BT_mode == "SBT":
                output_int = torch.round((output_int + skew_val) / 85) * 85
                ctx.skew_val = skew_val

            output = ((output_int.float()).add(-zero_point)).mul(scale)

        else:

            qmin, qmax, zero_point, scale = get_qparams(max_val, min_val, num_bits, eps)

            output_int = (output * (1 / scale) + zero_point).round().clip(qmin, qmax)
            output = ((output_int.float()).add(-zero_point)).mul(scale)

        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.num_bits = num_bits
        scale = torch.tensor([scale]).to(output.device)
        zero_point = torch.tensor(int(zero_point)).to(output.device)
        ctx.other = g, min_val, max_val, qmin, qmax
        ctx.scale = scale
        ctx.zp = zero_point

        ctx.output_int = output_int

        ctx.save_for_backward(output)
        ctx.s_alpha = s_alpha

        return output, scale, zero_point

    @staticmethod
    def backward(ctx, grad_output, grad_scale_val, grad_zero_point):

        if ctx.num_bits == 32:
            return None, None, None, None, None, None, None, None, None, None, None, None, None

        input = ctx.saved_tensors

        s_alpha = ctx.s_alpha
        if type(input) == tuple:
            input = input[0]

        input[torch.isinf(input)] = 0
        g, min_val, max_val, qmin, qmax = ctx.other
        indicate_small = (input < min_val).float()
        indicate_big = (input > max_val).float()

        inv_scale = 1.0 / ctx.scale
        num_bits = ctx.num_bits

        input = (input * (1 / ctx.scale) + ctx.zp)
        input_val = ctx.output_int

        if ctx.BT_mode == "BT":
            input = torch.round(input / 85) * 85

        if ctx.BT_mode == "SBT":
            input = torch.round((input + ctx.skew_val) / 85) * 85

        indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        grad_output = indicate_middle * grad_output
        grad_alpha = ((indicate_small * min_val + indicate_big * max_val + indicate_middle * (
                -input + input_val.round())) * grad_output).sum().unsqueeze(dim=0)

        return grad_output, grad_alpha, None, None, None, None, None, None, None, None, None, None, None


learns_quantize = LearnsQuantize.apply


def initial_alpha_range(input, num_bits, eps, datasetName, prop_mode, BT_mode):
    if "cuda" in input.device.type:
        input = input.cpu()
    init_min_val = input.min()
    init_max_val = input.max()
    candidate_alpa = 1.0
    minimal_error = 1e32

    kurt_val = kurtosis(input.data.reshape(-1), fisher=False)
    skew_val = round(skew(input.data.reshape(-1)))

    if num_bits == 2:
        range_num = 10
        interval_val = 0.1

    if num_bits == 4:
        range_num = 50
        interval_val = 0.02

    if num_bits == 8:
        range_num = 50
        interval_val = 0.02

    for i in range(range_num):
        test_alpha = (i + 1) * interval_val

        max_val = test_alpha * init_max_val
        min_val = test_alpha * init_min_val

        if BT_mode == "BT" or BT_mode == "SBT":
            qmin, qmax, zero_point, scale = get_qparams(max_val, min_val, 8, eps)
        else:
            qmin, qmax, zero_point, scale = get_qparams(max_val, min_val, num_bits, eps)
        inv_scale = 1.0 / scale
        quant_input = input * inv_scale + zero_point

        quant_input = quant_input.round().clamp(qmin, qmax)

        if BT_mode == "BT":
            quant_input = torch.round(quant_input / 85) * 85

        if BT_mode == "SBT":
            quant_input = torch.round((quant_input + skew_val) / 85) * 85

        dequant_input = (quant_input - zero_point) * scale

        find_kurt_val = kurtosis(dequant_input.data.reshape(-1), fisher=False)
        find_skew_val = round(skew(dequant_input.data.reshape(-1)))

        error_val = abs(dequant_input - input).mean()

        if error_val < minimal_error:
            minimal_error = error_val
            candidate_alpa = test_alpha
            find_kurt_val = kurt_val
            find_skew_val = skew_val

    if BT_mode == "SBT":
        if candidate_alpa < 0.5:
            candidate_alpa = candidate_alpa + 0.1
    else:
        if candidate_alpa < 0.1:
            candidate_alpa = candidate_alpa + 0.1

    return candidate_alpa, skew_val, kurt_val


class QLRQuantizer(nn.Module):

    def __init__(
            self,
            num_bits: int, datasetName: str, prop_mode: str, BT_mode: str,

    ):
        super(QLRQuantizer, self).__init__()

        self.num_bits = num_bits

        if num_bits != 32:
            self.register_buffer("min_val", torch.tensor([]))
            self.register_buffer("max_val", torch.tensor([]))
            self.num_bits = num_bits
            self.eps = torch.finfo(torch.float32).eps
            self.min_fn = torch.min
            self.max_fn = torch.max
            self.sample_fn = lambda x: x
            self.scale = torch.Tensor(1)
            self.zero_point = torch.Tensor(1)
            self.output_int = torch.Tensor(1)
            self.iter_count = 0
            self.min_val = None
            self.max_val = None
            self.custom_alpha_flag = False

            self.BT_mode = BT_mode

            self.skew_val = 0
            self.kurt_val = -1

            self.datasetName = datasetName
            self.prop_mode = prop_mode

    def update_ranges(self, input):

        input = self.sample_fn(input)
        self.min_val = self.min_fn(input)
        self.max_val = self.max_fn(input)

    def forward(self, input, custom_alpha=None, training=None):

        if self.num_bits == 32:
            return input, None, None

        self.iter_count = self.iter_count + 1
        if self.custom_alpha_flag == False:
            self.inital_alpha, self.skew_val, self.kurt_val = initial_alpha_range(input, self.num_bits, self.eps,
                                                                                  self.datasetName, self.prop_mode,
                                                                                  self.BT_mode)

            custom_alpha.data = torch.tensor([self.inital_alpha], device=input.device) * 1.0
            self.custom_alpha_flag = True

        if self.datasetName == "CiteSeer":
            if training and (self.iter_count - 1) % 100 == 0:
                self.inital_alpha, self.skew_val, self.kurt_val = initial_alpha_range(input, self.num_bits, self.eps,
                                                                                      self.datasetName, self.prop_mode,
                                                                                      self.BT_mode)
                custom_alpha.data = torch.tensor([self.inital_alpha], device=input.device) * 1.0

        self.custom_alpha = custom_alpha

        if training:
            self.update_ranges(input.detach())

        g = 0.5

        out, self.scale, self.zero_point = learns_quantize(input, custom_alpha, self.scale, self.zero_point,
                                                           self.output_int, g, self.max_val, self.min_val,
                                                           self.num_bits, self.eps, training, self.skew_val,
                                                           self.BT_mode)

        return out, self.scale, self.zero_point
