"""Pure-PyTorch re-implementation of the MinkowskiEngine operations used by PUResNetV2.0 (no nvcc / ME build needed).

Only what PUResNetV2 needs is implemented, with MinkowskiEngine 0.5.x semantics (verified by reading the ME source):
  * coordinates are integer voxels; a SparseTensor has an integer `stride` (tensor stride, 1,2,4,8,16)
  * Convolution(kernel=3, stride s): output coordinates = floor(in_coord / (in_stride*s)) * (in_stride*s), unique;
    for each output coordinate o and kernel index k, input = o + off(k) * in_stride, with
    k = ox + 3*oy + 9*oz, off = (ox-1, oy-1, oz-1)   (x varies fastest, see ME kernel_region.hpp)
    out[o] += in[i] @ W[k]; W has shape [27, C_in, C_out] (ME parameter layout, bias=False)
  * ConvolutionTranspose(kernel=3, stride 2): output coordinates = the existing coordinate map at stride/2;
    pairs are those of the corresponding strided convolution reversed, out[a] += in[b] @ W[k].
  * BatchNorm (eval mode), ReLU, Dropout (identity in eval), cat (feature concat, same coordinates), residual add.
Weights of the original checkpoint load unchanged (same state_dict keys).
Batch size 1 (single structure per forward), eval only.
"""
import torch
import torch.nn as nn

OFFS = torch.tensor([[ox - 1, oy - 1, oz - 1] for oz in range(3) for oy in range(3) for ox in range(3)])  # k = ox+3oy+9oz


class ST:
    """sparse tensor: coords [N,3] int64 (>=0), feats [N,C], stride int, ctx dict stride->coords (shared)"""

    def __init__(self, coords, feats, stride, ctx):
        self.coords, self.feats, self.stride, self.ctx = coords, feats, stride, ctx

    def like(self, feats):
        return ST(self.coords, feats, self.stride, self.ctx)


M = 1 << 12  # key base; coordinates (in A) are < 4096 after shifting, padded below


def _key(c):  # c [...,3] int64 with small negative offsets allowed (pad 64)
    c = c + 64
    return (c[..., 0] * M + c[..., 1]) * M + c[..., 2]


def _kmap(in_coords, out_coords, step, device):
    """for each k: (in_idx, out_idx) for in = out + off_k*step, only where in exists."""
    kin = _key(in_coords)
    sk, order = torch.sort(kin)
    pairs = []
    for k in range(27):
        q = _key(out_coords + OFFS[k].to(device) * step)
        pos = torch.searchsorted(sk, q).clamp(max=len(sk) - 1)
        ok = sk[pos] == q
        pairs.append((order[pos[ok]], torch.nonzero(ok).squeeze(1)))
    return pairs


class Conv(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.stride = stride
        self.kernel = nn.Parameter(torch.zeros(27, cin, cout))

    def forward(self, x: ST):
        dev = x.feats.device
        if self.stride == 1:
            oc, ost = x.coords, x.stride
        else:
            ost = x.stride * self.stride
            q = torch.div(x.coords, ost, rounding_mode="floor") * ost
            oc = torch.unique(q, dim=0)  # sorted unique
            x.ctx[ost] = oc
        pairs = _kmap(x.coords, oc, x.stride, dev)
        out = torch.zeros(len(oc), self.kernel.shape[2], device=dev, dtype=x.feats.dtype)
        for k, (ii, oi) in enumerate(pairs):
            if len(ii):
                out.index_add_(0, oi, x.feats[ii] @ self.kernel[k])
        return ST(oc, out, ost, x.ctx)


class ConvTr(nn.Module):
    def __init__(self, cin, cout, stride=2):
        super().__init__()
        self.stride = stride
        self.kernel = nn.Parameter(torch.zeros(27, cin, cout))

    def forward(self, x: ST):
        dev = x.feats.device
        ost = x.stride // self.stride
        oc = x.ctx[ost]  # fine coordinate map registered earlier
        pairs = _kmap(oc, x.coords, ost, dev)  # (fine idx, coarse idx) with fine = coarse + off*ost
        out = torch.zeros(len(oc), self.kernel.shape[2], device=dev, dtype=x.feats.dtype)
        for k, (fi, ci) in enumerate(pairs):
            if len(fi):
                out.index_add_(0, fi, x.feats[ci] @ self.kernel[k])
        return ST(oc, out, ost, x.ctx)


class BN(nn.Module):
    def __init__(self, c, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(c, momentum=momentum)

    def forward(self, x):
        return x.like(self.bn(x.feats))


class ReLU(nn.Module):
    def forward(self, x):
        return x.like(torch.relu(x.feats))


class Identity(nn.Module):  # dropout in eval
    def forward(self, x):
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = Conv(inplanes, planes, stride)
        self.norm1 = BN(planes)
        self.conv2 = Conv(planes, planes, 1)
        self.norm2 = BN(planes)
        self.relu = ReLU()
        self.downsample = downsample

    def forward(self, x):
        res = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            res = self.downsample(x)
        return self.relu(out.like(out.feats + res.feats))


def cat(a: ST, b: ST):
    assert a.stride == b.stride and a.coords.shape == b.coords.shape
    return a.like(torch.cat([a.feats, b.feats], 1))


class PUResNetV2(nn.Module):
    """same layer names as the original so the checkpoint's state_dict loads unchanged"""

    def __init__(self, in_channels=18, out_channels=1, planes=(32, 48, 128, 128, 128, 128, 48, 32),
                 layers=(2, 3, 1, 3, 3, 2, 1, 3)):
        super().__init__()
        P, L = planes, layers
        self.PLANES = P
        self.inplanes = P[0]
        self.dropout = Identity()
        self.conv0p1s1 = Conv(in_channels, self.inplanes)
        self.bn0 = BN(self.inplanes)
        self.conv1p1s2 = Conv(self.inplanes, self.inplanes, 2)
        self.bn1 = BN(self.inplanes)
        self.block1 = self._make_layer(P[0], L[0])
        self.conv2p2s2 = Conv(self.inplanes, self.inplanes, 2)
        self.bn2 = BN(self.inplanes)
        self.block2 = self._make_layer(P[1], L[1])
        self.conv3p4s2 = Conv(self.inplanes, self.inplanes, 2)
        self.bn3 = BN(self.inplanes)
        self.block3 = self._make_layer(P[2], L[2])
        self.conv4p8s2 = Conv(self.inplanes, self.inplanes, 2)
        self.bn4 = BN(self.inplanes)
        self.block4 = self._make_layer(P[3], L[3])
        self.convtr4p16s2 = ConvTr(self.inplanes, P[4])
        self.bntr4 = BN(P[4])
        self.inplanes = P[4] + P[2]
        self.block5 = self._make_layer(P[4], L[4])
        self.convtr5p8s2 = ConvTr(self.inplanes, P[5])
        self.bntr5 = BN(P[5])
        self.inplanes = P[5] + P[1]
        self.block6 = self._make_layer(P[5], L[5])
        self.convtr6p4s2 = ConvTr(self.inplanes, P[6])
        self.bntr6 = BN(P[6])
        self.inplanes = P[6] + P[0]
        self.block7 = self._make_layer(P[6], L[6])
        self.convtr7p2s2 = ConvTr(self.inplanes, P[7])
        self.bntr7 = BN(P[7])
        self.inplanes = P[7] + P[0]
        self.block8 = self._make_layer(P[7], L[7])
        self.final = Conv1(P[7], out_channels)
        self.relu = ReLU()

    def _make_layer(self, planes, blocks):
        ds = None
        if self.inplanes != planes:
            ds = nn.Sequential(Conv1(self.inplanes, planes, bias=False), BN(planes))
        layers = [BasicBlock(self.inplanes, planes, 1, ds)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: ST):
        r, d = self.relu, self.dropout
        out = self.bn0(self.conv0p1s1(x))
        out_p1 = r(out)
        out = d(r(self.bn1(self.conv1p1s2(out_p1))))
        out_b1p2 = self.block1(out)
        out = d(r(self.bn2(self.conv2p2s2(out_b1p2))))
        out_b2p4 = self.block2(out)
        out = d(r(self.bn3(self.conv3p4s2(out_b2p4))))
        out_b3p8 = self.block3(out)
        out = d(r(self.bn4(self.conv4p8s2(out_b3p8))))
        out = self.block4(out)
        out = cat(d(r(self.bntr4(self.convtr4p16s2(out)))), out_b3p8)
        out = self.block5(out)
        out = cat(d(r(self.bntr5(self.convtr5p8s2(out)))), out_b2p4)
        out = self.block6(out)
        out = cat(d(r(self.bntr6(self.convtr6p4s2(out)))), out_b1p2)
        out = self.block7(out)
        out = cat(d(r(self.bntr7(self.convtr7p2s2(out)))), out_p1)
        out = self.block8(out)
        return self.final(out)


class Conv1(nn.Module):
    """1x1x1 sparse convolution (kernel [cin,cout]) with optional bias, as ME MinkowskiConvolution(kernel_size=1)"""

    def __init__(self, cin, cout, bias=True):
        super().__init__()
        self.kernel = nn.Parameter(torch.zeros(cin, cout))  # ME stores 1x1x1 kernels as [cin, cout]
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, cout))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        f = x.feats @ self.kernel
        if self.bias is not None:
            f = f + self.bias[0]
        return x.like(f)


def forward_atoms(model, coords_int, feats, device):
    """coords_int [N,3] int (non-negative, unique rows), feats [N,18] float -> logits [N]"""
    c = torch.as_tensor(coords_int, dtype=torch.long, device=device)
    f = torch.as_tensor(feats, dtype=torch.float32, device=device)
    ctx = {1: c}
    with torch.no_grad():
        out = model(ST(c, f, 1, ctx))
    assert out.coords.shape == c.shape and bool((out.coords == c).all()), "coordinate order changed"
    return out.feats[:, 0]
