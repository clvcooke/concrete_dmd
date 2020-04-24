import torch
import torch.nn as nn
from dmd import FixedDMDSpatial, ParameterizableDMDSpatial
from torchvision.models import segmentation


class FixedDigitNet(nn.Module):
    def __init__(self, input_size=784, dmd_count=1, temperature=1, num_classes=10, dmd_type=FixedDMDSpatial,
                 hidden_size=32, init_strategy="flat", **kwargs):
        super(FixedDigitNet, self).__init__()
        self.input_size = input_size
        self.dmd_count = dmd_count
        self.simple_mlp = nn.Sequential(
            nn.Linear(self.dmd_count, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=-1)
        )
        self.dmds = dmd_type(input_size, dmd_count, temperature, init_strategy)

    def forward(self, x, cold=False):
        # sensed is (B, self.trajectory_length)
        sensed = torch.stack([dmd(x, cold) for dmd in self.dmds], dim=1)
        classified = self.simple_mlp(sensed)
        return classified


class AdaptiveDigitNet(nn.Module):
    def __init__(self, input_size=784, dmd_count=1, temperature=1, first_fixed=True, num_classes=10, hidden_size=32,
                 init_strategy="flat", adaptive_multi=1, **kwargs):
        super(AdaptiveDigitNet, self).__init__()
        assert dmd_count // adaptive_multi == dmd_count / adaptive_multi
        self.input_size = input_size
        self.dmd_count = dmd_count // adaptive_multi
        self.adaptive_multi = adaptive_multi
        if first_fixed:
            self.first_dmd = FixedDMDSpatial(input_size, adaptive_multi, temperature, init_strategy=init_strategy)
        else:
            # TODO:
            self.first_dmd = None
            raise RuntimeError()
        self.parameterizable_dmd = ParameterizableDMDSpatial(input_size, temperature, adaptive_multi)
        self.measurement_count = adaptive_multi
        encoder_output_size = hidden_size
        self.sense_encoder = nn.Sequential(
            nn.Linear(self.measurement_count, encoder_output_size),
            nn.ReLU(),
        )
        # from encoded sensor reading to memory
        self.memory = nn.GRUCell(encoder_output_size, encoder_output_size)

        # from memory to next patttern
        pattern_generator_size = hidden_size
        self.pattern_generator = nn.Sequential(
            nn.Linear(encoder_output_size, pattern_generator_size),
            nn.ReLU(),
            nn.Linear(pattern_generator_size, pattern_generator_size),
            nn.ReLU(),
            nn.Linear(pattern_generator_size, input_size * adaptive_multi)
        )

        classifier_hidden_size = hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Linear(classifier_hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Linear(classifier_hidden_size, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, cold=False):
        # TODO: maybe clear up
        # TODO: log the patterns
        hidden_state = None
        for i in range(self.dmd_count):
            if i == 0:
                sensor_reading = self.first_dmd(x, cold=cold)
            else:
                sensor_reading = self.parameterizable_dmd(x, logits, cold=cold)
            sensor_reading = sensor_reading.view(-1, self.measurement_count)
            encoded_reading = self.sense_encoder(sensor_reading)
            hidden_state = self.memory(encoded_reading, hidden_state)
            if i == (self.dmd_count - 1):
                classification = self.classifier(hidden_state)
            else:
                logits = self.pattern_generator(hidden_state).view(-1, self.input_size, self.adaptive_multi)
        return classification


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ReconNetV2(nn.Module):
    def __init__(self, dmd_count=10, resolution=32, init_strategy='flat', temperature=1, noise=0.0, **kwargs):
        super().__init__()
        self.resolution = resolution
        self.input_size = resolution ** 2
        self.dmd_count = dmd_count
        self.dmds = FixedDMDSpatial(resolution * resolution, temperature=temperature, output_size=dmd_count,
                                    init_strategy=init_strategy, noise=noise)

        # self.signal_projection = resolution
        # self.signal_remapper = nn.Sequential(
        #     nn.BatchNorm1d(dmd_count, track_running_stats=False),
        #     nn.Linear(dmd_count, self.signal_projection * self.signal_projection),
        #     nn.ReLU(),
        #     # nn.BatchNorm1d(self.signal_projection * self.signal_projection, track_running_stats=False)
        # )
        k_size = 2
        padding = 0
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.dmd_count, 64, k_size, 2, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, k_size, 2, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, k_size, 2, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, k_size, 2, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, k_size, 2, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )


    def forward(self, x, cold=False):
        signal_map = self.dmds(x, cold)
        signals = signal_map.view(-1, self.dmd_count, 1,1)
        output = self.conv_decoder(signals)
        return output


class ReconNet(nn.Module):
    def __init__(self, dmd_count=10, resolution=32, init_strategy='flat', temperature=1, noise=0.0, **kwargs):
        super().__init__()
        self.resolution = resolution
        self.input_size = resolution ** 2
        self.dmds = FixedDMDSpatial(resolution * resolution, temperature=temperature, output_size=dmd_count,
                                    init_strategy=init_strategy, noise=noise)

        self.signal_remapper = nn.Sequential(
            nn.BatchNorm1d(dmd_count, track_running_stats=False),
            nn.Linear(dmd_count, resolution * resolution),
            # nn.ReLU(),
            nn.BatchNorm1d(resolution * resolution, track_running_stats=False)
        )

        self.conv_decoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            # nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
            nn.ReLU()
        )

    def forward(self, x, cold=False):
        signal_map = self.dmds(x, cold)
        feature_map = self.signal_remapper(signal_map).view(-1, 1, self.resolution, self.resolution)
        output = self.conv_decoder(feature_map)
        return output
