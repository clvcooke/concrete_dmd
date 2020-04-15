import torch
import torch.nn as nn
from dmd import FixedDMDSpatial, ParameterizableDMDSpatial


class FixedDigitNet(nn.Module):
    def __init__(self, input_size=784, dmd_count=1, temperature=1, num_classes=10):
        super(FixedDigitNet, self).__init__()
        self.input_size = input_size
        self.dmd_count = dmd_count
        self.simple_mlp = nn.Sequential(
            nn.Linear(self.dmd_count, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=-1)
        )
        self.dmds = [FixedDMDSpatial(input_size, 1, temperature) for _ in range(self.dmd_count)]

    def forward(self, x):
        # sensed is (B, self.dmd_count)
        sensed = torch.stack([dmd(x) for dmd in self.dmds], dim=1)
        classified = self.simple_mlp(sensed)
        return classified


class AdaptiveDigitNet(nn.Module):
    def __init__(self, input_size=784, dmd_count=1, temperature=1, first_fixed=True, num_classes=10):
        super(AdaptiveDigitNet, self).__init__()
        self.input_size = input_size
        self.dmd_count = dmd_count
        if first_fixed:
            self.first_dmd = FixedDMDSpatial(input_size, 1, temperature)
        else:
            # TODO:
            self.first_dmd = None
            raise RuntimeError()
        self.parameterizable_dmd = ParameterizableDMDSpatial(input_size, temperature)
        self.measurement_count = 1
        encoder_output_size = 32
        self.sense_encoder = nn.Sequential(
            nn.Linear(self.measurement_count, encoder_output_size),
            nn.ReLU(),
        )
        # from encoded sensor reading to memory
        self.memory = nn.GRUCell(encoder_output_size, encoder_output_size)

        # from memory to next patttern
        pattern_generator_size = 32
        self.pattern_generator = nn.Sequential(
            nn.Linear(encoder_output_size, pattern_generator_size),
            nn.ReLU(),
            nn.Linear(pattern_generator_size, pattern_generator_size),
            nn.ReLU(),
            nn.Linear(pattern_generator_size, input_size)
        )

        classifier_hidden_size = 32
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Linear(classifier_hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Linear(classifier_hidden_size, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # TODO: maybe clear up
        # TODO: log the patterns
        hidden_state = None
        for i in range(self.dmd_count):
            if i == 0:
                sensor_reading = self.first_dmd(x)
            else:
                sensor_reading = self.parameterizable_dmd(x, logits)
            sensor_reading = sensor_reading.view(-1, self.measurement_count)
            encoded_reading = self.sense_encoder(sensor_reading)
            hidden_state = self.memory(encoded_reading, hidden_state)
            if i == (self.dmd_count - 1):
                classification = self.classifier(hidden_state)
            else:
                logits = self.pattern_generator(hidden_state)
        return classification
