import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from Vim import SS2D
from backbone.HRNet.config.default import _C as config
from backbone.HRNet.models.seg_hrnet import get_seg_model
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )

class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class CSDF(nn.Module):
    def __init__(self, dim, num_heads, atten_drop = 0., proj_drop = 0., dilation = [3, 5, 7], fc_ratio=4):
        super(CSDF, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.DMSC= ASPPModule(dim, dilation)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim//fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim//fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.spat_atten=SpatialAttention()
        self.kv = Conv(dim, 2 * dim, 1)
        self.fc2 = ConvBN(in_channels=dim, out_channels=dim, kernel_size=1)
        self.act = nn.ReLU6()

    def forward(self, x):
        u = x.clone()
        # Dense multi-scale connections
        attn = self.DMSC(x)

        # Channel Attention
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u

        # Spatial Attention
        s_attn=self.spat_atten(x)

        x=self.fc2(attn+c_attn+s_attn)
        x = self.act(x)

        return x

class GLSS(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=5, act_layer=nn.ReLU6, drop=0.,dim=512,atten_drop=0.):
        super(GLSS, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.conv1=DepthWiseConv(in_channel=hidden_features,out_channel=hidden_features,kernel=3)
        self.conv2=DepthWiseConv(in_channel=hidden_features,out_channel=hidden_features,kernel=5)
        self.fc2 = ConvBN(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        self.attn=SS2D(d_model=hidden_features,dropout=atten_drop,d_state=16,scan=8)

    def forward(self, x):
        x = self.fc1(x)
        #Local
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        #Global
        x3=self.attn(x)

        x=self.fc2(x1+x2+x3)
        x = self.act(x)

        return x

class Block(nn.Module):
    def __init__(self, dim=512, num_heads=16,  mlp_ratio=4, pool_ratio=16, drop=0., dilation=[3, 5, 7],
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = CSDF(dim, num_heads=num_heads, atten_drop=drop, proj_drop=drop, dilation=dilation, fc_ratio=mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim // mlp_ratio)
        self.mlp = GLSS(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                         drop=drop)
    def forward(self, x):
        u=x.clone()
        x=x+self.drop_path(self.attn(self.norm1(x)))
        x=x+self.drop_path(self.mlp(self.norm2(x)))

        return x+u

class Fusion(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(Fusion, self).__init__()
        self.dim=dim
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)
        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)

    def forward(self, x, res):
        if x.shape[2]!=res.shape[2]:
            x = self.upsample(x)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

class channelattention(nn.Module):
    def __init__(self, dim, fc_ratio=4):
        super(channelattention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim // fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        u=x.clone()
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u
        return c_attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.relu1 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x.clone()
        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)
        out1 = self.conv1(out)

        out2 = self.relu1(out1)

        out = self.sigmoid(out2)

        y = x * out.view(out.size(0), 1, out.size(-2), out.size(-1))
        return y

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout,dilation):

        super(depthwise_separable_conv, self).__init__()

        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, dilation=dilation,padding=(( dilation * 2) // 2), groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)


    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', depthwise_separable_conv(nin=inter_channels,nout=out_channels,dilation=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features

class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel,kernel):
        super(DepthWiseConv, self).__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=kernel,
                                    stride=1,
                                    padding= (kernel - 1) // 2,
                                    groups=in_channel)
        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels
        rate1, rate2, rate3 = atrous_rates

        self.b0 = _DenseASPPConv(in_channels,out_channels//2,out_channels,rate1,0.1)
        self.b1 = _DenseASPPConv(in_channels+out_channels*1,(in_channels+out_channels*1)//2,out_channels,rate2,0.1)
        self.b2 = _DenseASPPConv(in_channels+out_channels*2,(in_channels+out_channels*2)//2,out_channels,rate3,0.1)
        self.project = nn.Sequential(nn.Conv2d(in_channels+3 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        u=x.clone()
        aspp1=self.b0(x)
        x=torch.cat([aspp1,x],dim=1)

        aspp2=self.b1(x)
        x=torch.cat([aspp2,x],dim=1)

        aspp3=self.b2(x)
        x=torch.cat([aspp3,x],dim=1)
        return self.project(x)*u

class CAB(nn.Module):
    def __init__(self,eps=1e-8):
        super(CAB, self).__init__()
        channels=144
        #四层横向连接
        self.eps=eps
        #最底层
        self.weights0 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.attention0=channelattention(channels)
        self.conv4=ConvBNReLU(channels,channels,1)

        #倒数第二层

        self.weights1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.attention1=channelattention(channels)
        self.conv3=ConvBNReLU(channels,channels,1)

        #倒数第三层
        self.weights2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.attention2=channelattention(channels)
        self.conv2=ConvBNReLU(channels,channels,1)

        self.weights3 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.attention3=channelattention(channels)
        self.conv1=ConvBNReLU(channels,18,1)

    def forward(self, res1,res2,res3,res4):
        h, w = res4.size()[-2:]
        feature_1_4=F.interpolate(res1, size=(h, w), mode='bilinear')
        featture2_4=F.interpolate(res2, size=(h, w), mode='bilinear')
        feature_3_4=F.interpolate(res3, size=(h, w), mode='bilinear')
        out_feature1 = feature_1_4+ featture2_4+feature_3_4+res4
        out_feature1=self.attention0(out_feature1)

        #倒数第二层
        h, w = res3.size()[-2:]
        featture1_3 = F.interpolate(res1, size=(h, w), mode='bilinear')
        featture2_3=F.interpolate(res2, size=(h, w), mode='bilinear')
        feature4_3=F.interpolate(res4, size=(h, w), mode='bilinear')
        out_feature2 =featture1_3+ featture2_3+res3+feature4_3
        out_feature2=self.attention1(out_feature2)

        #倒数第三层
        h, w = res2.size()[-2:]
        feature4_2=F.interpolate(res4, size=(h, w), mode='bilinear')
        feature3_2=F.interpolate(res3, size=(h, w), mode='bilinear')
        feature1_2=F.interpolate(res1, size=(h, w), mode='bilinear')
        out_feature3 = feature1_2+res2+ feature3_2 + feature4_2
        out_feature3=self.attention2(out_feature3)

        #倒数第四层
        h, w = res1.size()[-2:]
        feature4_1=F.interpolate(res4, size=(h, w), mode='bilinear')
        feature3_1=F.interpolate(res3, size=(h, w), mode='bilinear')
        feature2_1=F.interpolate(res2, size=(h, w), mode='bilinear')
        out_feature4 =  res1+feature2_1+feature3_1 +  feature4_1
        out_feature4=self.conv1(self.attention3(out_feature4))

        return out_feature1,out_feature2,out_feature3,out_feature4

class Segmentation_Head(nn.Module):
    def __init__(self, dim, atten_drop=0., proj_drop=0., dilation=[3, 5, 7], fc_ratio=4, pool_ratio=16,dropout=0.,num_classes=6):
        super(Segmentation_Head, self).__init__()
        self.dim = dim
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.MSC = ASPPModule(dim, dilation)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim // fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.spat_atten = SpatialAttention()
        self.kv = Conv(dim, 2 * dim, 1)
        self.head = nn.Sequential(SeparableConvBNReLU(dim, dim, kernel_size=3),
                                  nn.Dropout2d(p=dropout, inplace=True),
                                  Conv(18, num_classes, kernel_size=1))

    def forward(self, x):
        # print(x.shape)
        u = x.clone()
        B, C, H, W = x.shape
        attn = self.MSC(x)

        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u

        s_attn = self.spat_atten(x)
        out = self.head(attn + c_attn + s_attn)
        return out

class Decoder(nn.Module):
    def __init__(self,
                 encode_channels=[32,64,128,256],
                 decode_channels=128,
                 dilation = [[6, 12, 18], [6, 12, 18], [6, 12, 18], [6, 12, 18]],
                 fc_ratio=4,
                 dropout=0.1,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.Conv1 = ConvBNReLU(encode_channels[-2], decode_channels, 1)
        self.Conv2 = ConvBNReLU(encode_channels[-3], decode_channels, 1)

        self.Conv4 = ConvBNReLU(encode_channels[-4], decode_channels, 1)
        self.b4 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[0])

        self.p3 = Fusion(decode_channels)
        self.b3 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[1])

        self.p2 = Fusion(decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[2])

        self.Conv3 = ConvBN(decode_channels, encode_channels[-4], 1)

        self.p1 = Fusion(encode_channels[-4])
        self.seg_head = Segmentation_Head(encode_channels[-4], fc_ratio=fc_ratio, dilation=dilation[3], dropout=dropout, num_classes=num_classes)

        self.CAB=CAB()
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):

        res3 = self.Conv1(res3)
        res2 = self.Conv2(res2)

        res1 = self.Conv4(res1)

        decoder_feature = []
        outfeature1, outfeature2, outfeature3, outfeature4 = self.CAB(res1, res2, res3, res4)
        x = self.b4(outfeature1)

        decoder_feature.append(x)
        x = self.p3(x, outfeature2)
        x = self.b3(x)
        decoder_feature.append(x)
        x = self.p2(x, outfeature3)
        x = self.b2(x)
        decoder_feature.append(x)
        x = self.Conv3(x)
        x = self.p1(x, outfeature4)
        decoder_feature.append(x)
        x = self.seg_head(x)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class MF_Mamba(nn.Module):
    def __init__(self,
                 encode_channels=[18,36,72,144],
                 decode_channels=144,
                 dropout=0.1,
                 num_classes=6,
                 ):
        super().__init__()

        self.backbone = get_seg_model(config)
        self.decoder = Decoder(encode_channels, decode_channels, dropout=dropout, num_classes=num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        x = self.decoder(res1, res2, res3, res4, h, w)
        return x

if __name__ == '__main__':
    from thop import profile
    from thop import clever_format
    DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    x = torch.randn(1, 3, 512,512).to(DEVICE)
    net = MF_Mamba().to(DEVICE)
    out = net(x)
    print(out.shape)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
    flops, params = clever_format([flops, params], '%.3f')
    print(f"运算量：{flops}, 参数量：{params}")
