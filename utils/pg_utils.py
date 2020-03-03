import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##########
##  PACT
##########


class PactClip(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, upper_bound):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.

            upper_bound   if input > upper_bound
        y = input         if 0 <= input <= upper_bound
            0             if input < 0
        """
        ctx.save_for_backward(input, upper_bound)
        return torch.clamp(input, 0, upper_bound.data)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        input, upper_bound, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_upper_bound = grad_output.clone()
        grad_input[input<0] = 0
        grad_input[input>upper_bound] = 0
        grad_upper_bound[input<=upper_bound] = 0
        return grad_input, torch.sum(grad_upper_bound)

class PactReLU(nn.Module):
    def __init__(self, upper_bound=6.0):
        super(PactReLU, self).__init__()
        self.upper_bound = nn.Parameter(torch.tensor(upper_bound))

    def forward(self, input):
        return PactClip.apply(input, self.upper_bound)


##########
##  Mask
##########


class SparseGreaterThan(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, threshold):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input, torch.tensor(threshold))
        return torch.Tensor.float(torch.gt(input, threshold))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        input, threshold, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<threshold] = 0
        return grad_input, None

class GreaterThan(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, threshold):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.Tensor.float(torch.gt(input, threshold))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input, None


##########
##  Quant
##########


class Floor(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class Round(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the round function is defined as the identity function.
        """
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class Clamp(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, min, max):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        return torch.clamp(input, min, max)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the clamp function is defined as the identity function.
        """
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None

class TorchBinarize(nn.Module):
    """ Binarizes a value in the range [-1,+1] to {-1,+1} """
    def __init__(self):
        super(TorchBinarize, self).__init__()

    def forward(self, input):
        """  clip to [-1,1] """
        input = Clamp.apply(input, -1.0, 1.0)
        """ rescale to [0,1] """
        input = (input+1.0) / 2.0
        """ round to {0,1} """
        input = Round.apply(input)
        """ rescale back to {-1,1} """
        input = input*2.0 - 1.0
        return input

class TorchRoundToBits(nn.Module):
    """ Quantize a tensor to a bitwidth larger than 1 """
    def __init__(self, bits=2):
        super(TorchRoundToBits, self).__init__()
        assert bits > 1, "RoundToBits is only used with bitwidth larger than 1."
        self.bits = bits
        self.epsilon = 1e-7

    def forward(self, input):
        """ extract the sign of each element """
        sign = torch.sign(input).detach()
        """ get the mantessa bits """
        input = torch.abs(input)
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply( input/scaling ,0.0, 1.0 )
        """ round the mantessa bits to the required precision """
        input = Round.apply(input * (2.0**self.bits-1.0)) / (2.0**self.bits-1.0)
        return input * scaling * sign

class TorchTruncate(nn.Module):
    """ 
    Quantize an input tensor to a b-bit fixed-point representation, and
    remain the bh most-significant bits.
        Args:
        input: Input tensor
        b:  Number of bits in the fixed-point
        bh: Number of most-significant bits remained
    """
    def __init__(self, b=8, bh=4):
        super(TorchTruncate, self).__init__()
        assert b > 0, "Cannot truncate floating-point numbers (b=0)."
        assert bh > 0, "Cannot output floating-point numbers (bh=0)."
        assert b > bh, "The number of MSBs are larger than the total bitwidth."
        self.b = b
        self.bh = bh
        self.epsilon = 1e-7

    def forward(self, input):
        """ extract the sign of each element """
        sign = torch.sign(input).detach()
        """ get the mantessa bits """
        input = torch.abs(input)
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply( input/scaling ,0.0, 1.0 )
        """ round the mantessa bits to the required precision """
        input = Round.apply( input * (2.0**self.b-1.0) )
        """ truncate the mantessa bits """
        input = Floor.apply( input / (2**(self.b-self.bh) * 1.0) )
        """ rescale """
        input *= (2**(self.b-self.bh) * 1.0)
        input /= (2.0**self.b-1.0)
        return input * scaling * sign

class TorchQuantize(nn.Module):
    """ 
    Quantize an input tensor to the fixed-point representation. 
        Args:
        input: Input tensor
        bits:  Number of bits in the fixed-point
    """
    def __init__(self, bits=0):
        super(TorchQuantize, self).__init__()
        if bits == 0:
            self.quantize = nn.Identity()
        elif bits == 1:
            self.quantize = TorchBinarize()
        else:
            self.quantize = TorchRoundToBits(bits)

    def forward(self, input):
        return self.quantize(input)


##########
##  Layer
##########


class QuantizedConv2d(nn.Conv2d):
    """ 
    A convolutional layer with its weight tensor and input tensor quantized. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', wbits=0, abits=0):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, 
                                              kernel_size, stride, 
                                              padding, dilation, groups, 
                                              bias, padding_mode)
        self.quantize_w = TorchQuantize(wbits)
        self.quantize_a = TorchQuantize(abits)
        self.weight_rescale = \
            np.sqrt(1.0/(kernel_size**2 * in_channels)) if (wbits == 1) else 1.0

    def forward(self, input):
        """ 
        1. Quantize the input tensor
        2. Quantize the weight tensor
        3. Rescale via McDonnell 2018 (https://arxiv.org/abs/1802.08530)
        4. perform convolution
        """
        return F.conv2d(self.quantize_a(input),
                        self.quantize_w(self.weight) * self.weight_rescale,
                        self.bias, self.stride, self.padding, 
                        self.dilation, self.groups)

class QuantizedLinear(nn.Linear):
    """ 
    A fully connected layer with its weight tensor and input tensor quantized. 
    """
    def __init__(self, in_features, out_features, bias=True, wbits=0, abits=0):
        super(QuantizedLinear, self).__init__(in_features, out_features, bias)
        self.quantize_w = TorchQuantize(wbits)
        self.quantize_a = TorchQuantize(abits)
        self.weight_rescale = np.sqrt(1.0/in_features) if (wbits == 1) else 1.0

    def forward(self, input):
        """ 
        1. Quantize the input tensor
        2. Quantize the weight tensor
        3. Rescale via McDonnell 2018 (https://arxiv.org/abs/1802.08530)
        4. perform matrix multiplication 
        """
        return F.linear(self.quantize_a(input), 
                        self.quantize_w(self.weight) * self.weight_rescale, 
                        self.bias)
        
class PGConv2d(nn.Conv2d):
    """ 
    A convolutional layer computed as out = out_msb + mask . out_lsb
        - out_msb = I_msb * W
        - mask = (I_msb * W)  > Delta
        - out_lsb = I_lsb * W
    out_msb calculates the prediction results.
    out_lsb is only calculated where a prediction result exceeds the threshold.

    **Note**: 
        1. PG predicts with <activations>.
        2. bias must set to be False!
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', wbits=8, abits=8, pred_bits=4, 
                 sparse_bp=False, alpha=5):
        super(PGConv2d, self).__init__(in_channels, out_channels, 
                                       kernel_size, stride, 
                                       padding, dilation, groups, 
                                       bias, padding_mode)
        self.quantize_w = TorchQuantize(wbits)
        self.quantize_a = TorchQuantize(abits)
        self.trunc_a = TorchTruncate(b=abits, bh=pred_bits)
        self.gt = SparseGreaterThan.apply if sparse_bp else GreaterThan.apply
        self.weight_rescale = \
            np.sqrt(1.0/(kernel_size**2 * in_channels)) if (wbits == 1) else 1.0
        self.alpha = alpha

        """ 
        zero initialization
        nan loss while using torch.Tensor to initialize the thresholds 
        """
        self.threshold = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        """ number of output features """
        self.num_out = 0
        """ number of output features computed at high precision """
        self.num_high = 0

    def forward(self, input):
        """ 
        1. Truncate the input tensor
        2. Quantize the weight tensor
        3. Rescale via McDonnell 2018 (https://arxiv.org/abs/1802.08530)
        4. perform MSB convolution
        """
        out_msb = F.conv2d(self.trunc_a(input),
                           self.quantize_w(self.weight) * self.weight_rescale,
                           self.bias, self.stride, self.padding, 
                           self.dilation, self.groups)
        """ Calculate the mask """
        mask = self.gt(torch.sigmoid(self.alpha*(out_msb-self.threshold)), 0.5)
        """ update report """
        self.num_out = mask.cpu().numel()
        self.num_high = mask[mask>0].cpu().numel()
        """ perform LSB convolution """
        out_lsb = F.conv2d(self.quantize_a(input)-self.trunc_a(input),
                           self.quantize_w(self.weight) * self.weight_rescale, 
                           self.bias, self.stride, self.padding, 
                           self.dilation, self.groups)
        """ combine outputs """
        return out_msb + mask * out_lsb
