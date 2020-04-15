import torch
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


class FixedDMDSpatial(nn.Module):
    def __init__(self, input_size=784, output_size=1, temperature=1):
        super(FixedDMDSpatial, self).__init__()
        assert output_size == 1
        assert temperature > 0
        self.input_size = input_size
        self.output_size = output_size
        # binary mask
        self.logits = nn.Parameter(
            torch.ones((self.input_size), dtype=torch.float, requires_grad=True) / self.input_size)
        self.temperature = temperature
        self.sense_scale = nn.Parameter(
            torch.ones(self.output_size, dtype=torch.float, requires_grad=True) / self.input_size)
        self.sense_bias = nn.Parameter(torch.ones(self.output_size, dtype=torch.float, requires_grad=True) * -0.5)
        self.dmd = ParameterizableDMDSpatial(input_size, temperature)

    def forward(self, x):
        # view x as a vector of (batch_size, input_size)
        batch_size = x.shape[0]
        logits_expanded = self.logits.expand((batch_size, self.input_size))
        sensed = self.dmd(x, logits_expanded)
        # now scale and bias
        sensed_scaled = sensed * self.sense_scale
        sensed_biased = sensed_scaled + self.sense_bias
        return sensed_biased


class ParameterizableDMDSpatial(nn.Module):

    def __init__(self, input_size, temperature=0.5):
        """
        A parameterizable DMD used for adaptive approaches and non-adaptive approaches
        :param input_size:
        :param temperature:
        """
        super(ParameterizableDMDSpatial, self).__init__()
        self.input_size = input_size
        self.temperature = temperature

    def forward(self, x, logits):
        x_local = x.view(-1, self.input_size)
        # make a dist which has samples (batch_size, input_size, 2)
        dist = RelaxedBernoulli(temperature=self.temperature, logits=logits)
        samples = dist.rsample()
        # now "mask" the pixels of the input spatially by elementwise multiply, we throw away the second value
        masked_x = samples[:, :] * x_local
        # get sensor values, shape of (B)
        sensed = masked_x.sum(axis=1)
        return sensed

