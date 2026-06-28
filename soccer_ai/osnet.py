"""OSNet (Omni-Scale Network) for person re-identification.

Ported from torchreid (MIT licence):
    Zhou et al. "Omni-Scale Feature Learning for Person Re-Identification." ICCV 2019.
    https://github.com/KaiyangZhou/deep-person-reid

Only osnet_x1_0 is kept (channels=[64,256,384,512], feature_dim=512).
The classifier head is retained so pretrained weights load cleanly; use
model.eval() and call forward() to get the 512-d embedding directly.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Sports-finetuned OSNet_x1_0 checkpoint (sportsreid, SoccerNet Re-ID, 83.4 mAP)
_SPORTS_GDRIVE_ID  = "1To0Ww6_HxU2ITAlb4kQEgYExV-orwit8"
_DEFAULT_SAVE_NAME = "osnet_x1_0_sportsreid.pth.tar"


# ---------------------------------------------------------------------------
# Basic layers
# ---------------------------------------------------------------------------

class _ConvBnRelu(nn.Module):
    def __init__(self, in_c, out_c, k, stride=1, pad=0, groups=1, instance_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=pad,
                              bias=False, groups=groups)
        self.bn = (nn.InstanceNorm2d(out_c, affine=True)
                   if instance_norm else nn.BatchNorm2d(out_c))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1(nn.Module):
    def __init__(self, in_c, out_c, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False, groups=groups)
        self.bn   = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1Linear(nn.Module):
    """1×1 conv + BN, no ReLU."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class _LightConv3x3(nn.Module):
    """Lightweight 3×3: 1×1 linear + depth-wise 3×3."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False, groups=out_c)
        self.bn    = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


# ---------------------------------------------------------------------------
# Channel-wise attention gate
# ---------------------------------------------------------------------------

class _ChannelGate(nn.Module):
    def __init__(self, in_c, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_c, in_c // reduction, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_c // reduction, in_c, 1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        s = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * s


# ---------------------------------------------------------------------------
# OSBlock
# ---------------------------------------------------------------------------

class _OSBlock(nn.Module):
    """Omni-scale feature learning block."""
    def __init__(self, in_c, out_c, instance_norm=False, bottleneck=4, **_):
        super().__init__()
        mid = out_c // bottleneck
        self.conv1  = _Conv1x1(in_c, mid)
        self.conv2a = _LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(_LightConv3x3(mid, mid), _LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(_LightConv3x3(mid, mid), _LightConv3x3(mid, mid),
                                    _LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(_LightConv3x3(mid, mid), _LightConv3x3(mid, mid),
                                    _LightConv3x3(mid, mid), _LightConv3x3(mid, mid))
        self.gate       = _ChannelGate(mid)
        self.conv3      = _Conv1x1Linear(mid, out_c)
        self.downsample = _Conv1x1Linear(in_c, out_c) if in_c != out_c else None
        self.IN         = nn.InstanceNorm2d(out_c, affine=True) if instance_norm else None

    def forward(self, x):
        identity = x
        x1  = self.conv1(x)
        x2  = (self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) +
               self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1)))
        x3  = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


# ---------------------------------------------------------------------------
# OSNet
# ---------------------------------------------------------------------------

class OSNet(nn.Module):
    """Omni-Scale Network backbone + 512-d embedding head."""

    def __init__(self, num_classes: int, channels=(64, 256, 384, 512),
                 feature_dim: int = 512, instance_norm: bool = False) -> None:
        super().__init__()
        IN = instance_norm

        self.conv1       = _ConvBnRelu(3, channels[0], 7, stride=2, pad=3, instance_norm=IN)
        self.maxpool     = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2       = self._make_layer(channels[0], channels[1], 2, reduce=True, IN=IN)
        self.conv3       = self._make_layer(channels[1], channels[2], 2, reduce=True)
        self.conv4       = self._make_layer(channels[2], channels[3], 2, reduce=False)
        self.conv5       = _Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc          = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
        )
        self.classifier  = nn.Linear(feature_dim, num_classes)
        self._init_params()

    @staticmethod
    def _make_layer(in_c, out_c, n, reduce=False, IN=False):
        layers = [_OSBlock(in_c, out_c, instance_norm=IN)]
        for _ in range(1, n):
            layers.append(_OSBlock(out_c, out_c, instance_norm=IN))
        if reduce:
            layers.append(nn.Sequential(_Conv1x1(out_c, out_c), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        v = self.global_avgpool(x).view(x.size(0), -1)
        return self.fc(v)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return 512-d feature vector regardless of train/eval mode."""
        return self._backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self._backbone(x)
        # In eval mode return the embedding; in train mode return logits
        if not self.training:
            return v
        return self.classifier(v)


# ---------------------------------------------------------------------------
# Factory + weight loader
# ---------------------------------------------------------------------------

def osnet_x1_0(num_classes: int = 1) -> OSNet:
    """Standard OSNet (width ×1.0, 512-d output)."""
    return OSNet(num_classes, channels=[64, 256, 384, 512], feature_dim=512)


def ensure_weights(save_dir: str | Path = "weights") -> Path:
    """Download sports-finetuned OSNet_x1_0 checkpoint if not present.

    Uses gdown (already a torchreid dependency).  Falls back to a manual
    instructions message if gdown is unavailable.

    Returns the path to the local checkpoint file.
    """
    save_dir  = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dest = save_dir / _DEFAULT_SAVE_NAME

    if dest.exists():
        log.info("ReID weights already present: %s", dest)
        return dest

    url = f"https://drive.google.com/uc?id={_SPORTS_GDRIVE_ID}"
    log.info("Downloading OSNet_x1_0 sportsreid weights → %s", dest)
    print(f"[osnet] Downloading sports ReID weights from Google Drive → {dest}")

    try:
        import gdown
        gdown.download(url, str(dest), quiet=False)
    except ImportError:
        raise RuntimeError(
            "gdown is required for auto-download.  "
            "Run:  pip install gdown\n"
            f"Or manually download from {url}\n"
            f"and save to {dest}"
        )

    if not dest.exists():
        raise RuntimeError(
            f"Download failed. Download manually from:\n"
            f"  https://drive.google.com/file/d/{_SPORTS_GDRIVE_ID}\n"
            f"and save to: {dest}"
        )

    print(f"[osnet] Download complete: {dest}")
    return dest


def load_weights(model: OSNet, weight_path: str, device: str = "cpu") -> None:
    """Load a sportsreid / torchreid checkpoint, ignoring mismatched layers.

    Handles:
    - Raw state_dict or {'state_dict': ...} wrapper
    - DataParallel 'module.' key prefix
    - Size mismatches (e.g. classifier with different num_classes)
    """
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    model_dict = model.state_dict()
    new_sd: OrderedDict = OrderedDict()
    skipped = []

    for k, v in state_dict.items():
        k = k[7:] if k.startswith("module.") else k          # strip DataParallel prefix
        if k in model_dict and model_dict[k].shape == v.shape:
            new_sd[k] = v
        else:
            skipped.append(k)

    model_dict.update(new_sd)
    model.load_state_dict(model_dict)

    matched = len(new_sd)
    total   = len(state_dict)
    print(f"[osnet] loaded {matched}/{total} layers from {weight_path}"
          + (f"  (skipped: {skipped})" if skipped else ""))
