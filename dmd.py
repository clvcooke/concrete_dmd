import torch
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


class FixedDMDSpatial(nn.Module):
    def __init__(self, input_size=784, output_size=1, temperature=1, init_strategy='flat'):
        super(FixedDMDSpatial, self).__init__()
        assert temperature > 0
        self.input_size = input_size
        self.output_size = output_size
        # binary mask
        logit_shape = (self.input_size, self.output_size)
        if init_strategy == 'flat':
            logit_values = torch.ones(logit_shape, dtype=torch.float, requires_grad=True) / self.input_size
        elif init_strategy == 'uniform':
            logit_values = torch.rand(logit_shape, dtype=torch.float, requires_grad=True)
        elif init_strategy == 'normal':
            logit_values = torch.randn(logit_shape, dtype=torch.float, requires_grad=True)
        else:
            logit_values = None
        self.logits = nn.Parameter(logit_values)
        self.temperature = temperature
        self.sense_scale = nn.Parameter(
            torch.ones(self.output_size, dtype=torch.float, requires_grad=True) / self.input_size)
        self.sense_bias = nn.Parameter(torch.ones(self.output_size, dtype=torch.float, requires_grad=True) * -0.5)
        self.dmd = ParameterizableDMDSpatial(input_size, temperature)

    def forward(self, x, cold=False):
        # view x as a vector of (batch_size, input_size)
        batch_size = x.shape[0]
        logits_expanded = self.logits.expand((batch_size, self.input_size, self.output_size))
        sensed = self.dmd(x, logits_expanded, cold=cold)
        # now scale and bias
        sensed_scaled = sensed * self.sense_scale
        sensed_biased = sensed_scaled + self.sense_bias
        return sensed_biased


class ParameterizableDMDSpatial(nn.Module):

    def __init__(self, input_size, temperature=0.5, output_size=1):
        """
        A parameterizable DMD used for adaptive approaches and non-adaptive approaches
        :param input_size:
        :param temperature:
        """
        super(ParameterizableDMDSpatial, self).__init__()
        self.input_size = input_size
        self.temperature = temperature
        self.output_size = output_size

    def forward(self, x, logits, cold=False):
        x_local = x.view(-1, self.input_size, 1)
        # make a dist which has samples (batch_size, input_size, 2)
        if cold:
            temperature = 0.0
        else:
            temperature = self.temperature
        dist = RelaxedBernoulli(temperature=temperature, logits=logits)
        samples = dist.rsample()
        # now "mask" the pixels of the input spatially by elementwise multiply, we throw away the second value
        masked_x = samples * x_local
        # get sensor values, shape of (B)
        sensed = masked_x.sum(axis=1)
        return sensed


class ContinuousDMD(nn.Module):
    # I guess this is basically an SLM

    def __init__(self, input_size, output_size, *args):
        super(ContinuousDMD, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mask = nn.Parameter(
            torch.ones((self.input_size), dtype=torch.float, requires_grad=True) / 2)
        self.sense_scale = nn.Parameter(
            torch.ones(self.output_size, dtype=torch.float, requires_grad=True) / self.input_size)
        self.sense_bias = nn.Parameter(torch.ones(self.output_size, dtype=torch.float, requires_grad=True) * -0.5)

    def forward(self, x, binary=False):
        x_local = x.view(-1, self.input_size)
        # make a dist which has samples (batch_size, input_size, 2)
        batch_size = x.shape[0]
        mask = (self.mask.expand((batch_size, self.input_size)) + 1) / 2
        if binary:
            mask = torch.round(mask)
        # now "mask" the pixels of the input spatially by elementwise multiply, we throw away the second value
        masked_x = mask * x_local
        # get sensor values, shape of (B)
        sensed = masked_x.sum(axis=1)
        return sensed
