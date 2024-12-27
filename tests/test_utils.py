import torch as th

from torch.nn import functional as F

from weightslab.model_with_ops import NetworkWithOps
from weightslab.model_with_ops import DepType
from weightslab.modules_with_ops import BatchNorm2dWithNeuronOps
from weightslab.modules_with_ops import Conv2dWithNeuronOps
from weightslab.modules_with_ops import LinearWithNeuronOps
from weightslab.tracking import TrackingMode


class MNISTModel(NetworkWithOps):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.tracking_mode = TrackingMode.DISABLED

        self.conv0 = Conv2dWithNeuronOps(
            in_channels=1, out_channels=16, kernel_size=3)
        self.bnorm0 = BatchNorm2dWithNeuronOps(16)
        self.linear0 = LinearWithNeuronOps(in_features=16, out_features=10)

        self.register_dependencies({
            (self.conv0, self.bnorm0, DepType.SAME),
            (self.bnorm0, self.linear0, DepType.INCOMING),
        })

    def forward(self, input: th.tensor):
        self.maybe_update_age(input)
        x = self.conv0(input)
        x = self.bnorm0(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 16)
        x = x.view(x.size(0), -1)
        output = self.linear0(x)
        output = F.log_softmax(output, dim=1)
        return output