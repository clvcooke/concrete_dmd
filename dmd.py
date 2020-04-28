import torch
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import numpy as np


class FixedDMDAperture(nn.Module):
    def __init__(self, input_size, output_size=1, temperature=1, init_strategy='flat', noise=0.0):
        super(FixedDMDAperture, self).__init__()
        assert temperature > 0
        self.input_size = input_size
        self.output_size = output_size
        self.noise = noise
        self.temperature = temperature
        self.resolution = int(np.sqrt(self.input_size))
        # binary mask
        logit_shape = (self.output_size, self.input_size)
        if init_strategy == 'flat':
            logit_values = torch.ones(logit_shape, dtype=torch.float, requires_grad=True) / self.input_size
        elif init_strategy == 'uniform':
            logit_values = torch.rand(logit_shape, dtype=torch.float, requires_grad=True)
        elif init_strategy == 'normal':
            logit_values = torch.randn(logit_shape, dtype=torch.float, requires_grad=True)
        else:
            logit_values = None
        self.logits = nn.Parameter(logit_values, requires_grad=True)

    def forward(self, x, cold=False):
        batch_size = x.shape[0]
        # take fft (B, 1, width*height, 2)
        x_freq = torch.rfft(x, signal_ndim=2, onesided=False).view(batch_size, 1, self.input_size, 2)
        if cold:
            temperature = 0.0
        else:
            temperature = self.temperature
        logits = self.logits.expand((batch_size, self.output_size, self.input_size))
        dist = RelaxedBernoulli(temperature=temperature, logits=logits)
        if cold:
            samples = dist.sample()
        else:
            samples = dist.rsample()
        # reshape so broadcasting works properly
        samples = samples.view(batch_size, self.output_size, self.input_size, 1)
        # mask the frequencies (B, output_size, width*height, 2)
        sensed_freq = x_freq * samples
        # reshape for ifft
        sensed_freq = sensed_freq.view(-1, int(np.sqrt(self.input_size)))
        sensed_freq = sensed_freq.view(-1, self.resolution, self.resolution, 2)
        # (B*output_size, resolution, resolution)
        sensed_images = torch.irfft(sensed_freq, 2, normalized=False, onesided=False)
        sensed = torch.sum(sensed_images.view(batch_size, self.output_size, self.input_size), axis=-1)
        noise_scale = self.noise * torch.sqrt(sensed.detach())
        sensed += torch.randn_like(sensed) * noise_scale
        return sensed


class FixedDMDSpatial(nn.Module):
    def __init__(self, input_size=784, output_size=1, temperature=1, init_strategy='flat', noise=0.0):
        super(FixedDMDSpatial, self).__init__()
        assert temperature > 0
        self.input_size = input_size
        self.output_size = output_size
        self.noise = noise
        # binary mask
        logit_shape = (self.output_size, self.input_size)
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
        logits_expanded = self.logits.expand((batch_size, self.output_size, self.input_size))
        sensed = self.dmd(x / self.input_size, logits_expanded, cold=cold)
        # readout noise std
        noise_scale = self.noise * torch.sqrt(sensed.detach())
        sensed += torch.randn_like(sensed) * noise_scale
        # # now scale and bias
        # sensed_scaled = sensed * self.sense_scale
        # sensed_biased = sensed_scaled + self.sense_bias
        return sensed


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
        x_local = x.view(-1, 1, self.input_size)
        # make a dist which has samples (batch_size, input_size, 2)
        if cold:
            temperature = 0.0
        else:
            temperature = self.temperature
        dist = RelaxedBernoulli(temperature=temperature, logits=logits)
        if cold:
            samples = dist.sample()
        else:
            samples = dist.rsample()
        # now "mask" the pixels of the input spatially by elementwise multiply, we throw away the second value
        masked_x = samples * x_local
        # get sensor values, shape of (B)
        sensed = masked_x.sum(axis=-1)
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
