import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from math import pi, sqrt, ceil
import torch.nn.functional as F
import numpy as np
from matplotlib.path import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms.functional import center_crop, resize
from torchvision.io import read_image
from torch.nn.functional import interpolate
from hnerv_utils import eval_quantize_per_tensor

# Video dataset
class VideoDataSet(Dataset):
    def __init__(self, args):
        if os.path.isfile(args.data_path):
            video = EncodedVideo.from_path(args.data_path)
            video_data = video.get_clip(0, video.duration)['video'] / 255. # (C,T,H,W)
        else:
            img_list = [read_image(os.path.join(args.data_path, x)) for x in sorted(os.listdir(args.data_path))]
            video_data = torch.stack(img_list, 1) / 255. # (C,T,H,W)
        # Resize the input video and center crop
        if args.crop_list != '-1': 
            crop_h, crop_w = [int(x) for x in args.crop_list.split('_')[:2]]
            if 'last' not in args.crop_list:
                video_data = center_crop(video_data, (crop_h, crop_w))
        if args.resize_list != '-1':
            if '_' in args.resize_list:
                resize_h, resize_w = [int(x) for x in args.resize_list.split('_')]
                video_data = interpolate(video_data, (resize_h, resize_w), 'bicubic')
            else:
                resize_hw = int(args.resize_list)
                video_data = resize(video_data, resize_hw,  'bicubic')
        if 'last' in args.crop_list:
            video_data = center_crop(video_data, (crop_h, crop_w))
        self.video_data =video_data        
        self.final_size = video_data.size(-2) * video_data.size(-1)

    def __len__(self):
        return self.video_data.size(1)

    def __getitem__(self, idx):
        tensor_image = self.video_data[:,idx]
        norm_idx = float(idx) / self.video_data.size(1)
        sample = {'img': tensor_image, 'idx': idx, 'norm_idx': norm_idx}
        
        return sample


class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        conv = UpConv if kargs['dec_block'] else DownConv
        self.conv = conv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], strd=kargs['strd'], ks=kargs['ks'], 
            conv_type=kargs['conv_type'], bias=kargs['bias'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def Quantize_tensor(img_embed, quant_bit):
    out_min = img_embed.min(dim=1, keepdim=True)[0]
    out_max = img_embed.max(dim=1, keepdim=True)[0]
    scale = (out_max - out_min) / 2 ** quant_bit
    img_embed = ((img_embed - out_min) / scale).round()
    img_embed = out_min + scale * img_embed  
    return img_embed


def OutImg(x, out_bias='tanh'):
    if out_bias == 'sigmoid':
        return torch.sigmoid(x)
    elif out_bias == 'tanh':
        return (torch.tanh(x) * 0.5) + 0.5
    else:
        return x + float(out_bias)


class HNeRV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = args.embed
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split('_')]
        enc_blks, dec_blks = [int(x) for x in args.num_blks.split('_')]

        # BUILD Encoder LAYERS
        if len(args.enc_strds):         #HNeRV
            enc_dim1, enc_dim2 = [int(x) for x in args.enc_dim.split('_')]
            c_in_list, c_out_list = [enc_dim1] * len(args.enc_strds), [enc_dim1] * len(args.enc_strds)
            c_out_list[-1] = enc_dim2
            if args.conv_type[0] == 'convnext':
                self.encoder = ConvNeXt(stage_blocks=enc_blks, strds=args.enc_strds, dims=c_out_list,
                    drop_path_rate=0)
            else:
                c_in_list[0] = 3
                encoder_layers = []
                for c_in, c_out, strd in zip(c_in_list, c_out_list, args.enc_strds):
                    encoder_layers.append(NeRVBlock(dec_block=False, conv_type=args.conv_type[0], ngf=c_in,
                     new_ngf=c_out, ks=ks_enc, strd=strd, bias=True, norm=args.norm, act=args.act))
                self.encoder = nn.Sequential(*encoder_layers)
            hnerv_hw = np.prod(args.enc_strds) // np.prod(args.dec_strds)
            self.fc_h, self.fc_w = hnerv_hw, hnerv_hw
            ch_in = enc_dim2
        else:
            ch_in = 2 * int(args.embed.split('_')[-1])
            self.pe_embed = PositionEncoding(args.embed)  
            self.encoder = nn.Identity()
            self.fc_h, self.fc_w = [int(x) for x in args.fc_hw.split('_')]

        # BUILD Decoder LAYERS  
        decoder_layers = []        
        ngf = args.fc_dim
        out_f = int(ngf * self.fc_h * self.fc_w)
        decoder_layer1 = NeRVBlock(dec_block=False, conv_type='conv', ngf=ch_in, new_ngf=out_f, ks=0, strd=1, 
            bias=True, norm=args.norm, act=args.act)
        decoder_layers.append(decoder_layer1)
        for i, strd in enumerate(args.dec_strds):                         
            reduction = sqrt(strd) if args.reduce==-1 else args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            for j in range(dec_blks):
                cur_blk = NeRVBlock(dec_block=True, conv_type=args.conv_type[1], ngf=ngf, new_ngf=new_ngf, 
                    ks=min(ks_dec1+2*i, ks_dec2), strd=1 if j else strd, bias=True, norm=args.norm, act=args.act)
                decoder_layers.append(cur_blk)
                ngf = new_ngf
        
        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv2d(ngf, 3, 3, 1, 1) 
        self.out_bias = args.out_bias

    def forward(self, input, input_embed=None, encode_only=False):
        if input_embed != None:
            img_embed = input_embed
        else:
            if 'pe' in self.embed:
                input = self.pe_embed(input[:,None]).float()
            img_embed = self.encoder(input)

        # import pdb; pdb.set_trace; from IPython import embed; embed()     
        embed_list = [img_embed]
        dec_start = time.time()
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        embed_list.append(output)
        for layer in self.decoder[1:]:
            output = layer(output) 
            embed_list.append(output)

        img_out = OutImg(self.head_layer(output), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start

        return  img_out, embed_list, dec_time


class HNeRVDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fc_h, self.fc_w = [torch.tensor(x) for x in [model.fc_h, model.fc_w]]
        self.out_bias = model.out_bias
        self.decoder = model.decoder

    def forward(self, img_embed):
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        for layer in self.decoder[1:]:
            output = layer(output) 

        return  OutImg(output, self.out_bias)


###################################  Basic layers like position encoding/ downsample layers/ upscale blocks   ###################################
class PositionEncoding(nn.Module):
    def __init__(self, pe_embed):
        super(PositionEncoding, self).__init__()
        self.pe_embed = pe_embed
        if 'pe' in pe_embed:
            lbase, levels = [float(x) for x in pe_embed.split('_')[-2:]]
            self.pe_bases = lbase ** torch.arange(int(levels)) * pi

    def forward(self, pos):
        if 'pe' in self.pe_embed:
            value_list = pos * self.pe_bases.to(pos.device)
            pe_embed = torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)
            return pe_embed.view(pos.size(0), -1, 1, 1)
        else:
            return pos


class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)


def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = Sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):    
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


class DownConv(nn.Module):
    def __init__(self, **kargs):
        super(DownConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        if kargs['conv_type'] == 'pshuffel':
            self.downconv = nn.Sequential(
                nn.PixelUnshuffle(strd) if strd !=1 else nn.Identity(),
                nn.Conv2d(ngf * strd**2, new_ngf, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias'])
            )
        elif kargs['conv_type'] == 'conv':
            self.downconv = nn.Conv2d(ngf, new_ngf, ks+strd, strd, ceil(ks / 2), bias=kargs['bias'])
        elif kargs['conv_type'] == 'interpolate':
            self.downconv = nn.Sequential(
                nn.Upsample(scale_factor=1. / strd, mode='bilinear',),
                nn.Conv2d(ngf, new_ngf, ks+strd, 1, ceil((ks + strd -1) / 2), bias=kargs['bias'])
            )
        
    def forward(self, x):
        return self.downconv(x)


class UpConv(nn.Module):
    def __init__(self, **kargs):
        super(UpConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        if  kargs['conv_type']  == 'pshuffel':
            self.upconv = nn.Sequential(
                nn.Conv2d(ngf, new_ngf * strd * strd, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias']),
                nn.PixelShuffle(strd) if strd !=1 else nn.Identity(),
            )
        elif  kargs['conv_type']  == 'conv':
            self.upconv = nn.ConvTranspose2d(ngf, new_ngf, ks+strd, strd, ceil(ks / 2))
        elif  kargs['conv_type']  == 'interpolate':
            self.upconv = nn.Sequential(
                nn.Upsample(scale_factor=strd, mode='bilinear',),
                nn.Conv2d(ngf, new_ngf, strd + ks, 1, ceil((ks + strd -1) / 2), bias=kargs['bias'])
            )

    def forward(self, x):
        return self.upconv(x)


class ModConv(nn.Module):
    def __init__(self, **kargs):
        super(ModConv, self).__init__()
        mod_ks, mod_groups, ngf = kargs['mod_ks'], kargs['mod_groups'], kargs['ngf']
        self.mod_conv_multi = nn.Conv2d(ngf, ngf, mod_ks, 1, (mod_ks - 1)//2, groups=(ngf if mod_groups==-1 else mod_groups))
        self.mod_conv_sum = nn.Conv2d(ngf, ngf, mod_ks, 1, (mod_ks - 1)//2, groups=(ngf if mod_groups==-1 else mod_groups))

    def forward(self, x):
        sum_att = self.mod_conv_sum(x)
        multi_att = self.mod_conv_multi(x)
        return torch.sigmoid(multi_att) * x + sum_att


###################################  Tranform input for denoising or inpainting   ###################################
def RandomMask(height, width, points_num, scale=(0, 1)):
    polygon = [(x, y) for x,y in zip(np.random.randint(height * scale[0], height * scale[1], size=points_num), 
                             np.random.randint(width * scale[0], width * scale[1], size=points_num))]
    poly_path=Path(polygon)

    x, y = np.mgrid[:height, :width]
    coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)
    mask = poly_path.contains_points(coors).reshape(height, width)
    return 1 - torch.from_numpy(mask).float()


class TransformInput(nn.Module):
    def __init__(self, args):
        super(TransformInput, self).__init__()
        self.vid = args.vid
        if 'inpaint' in self.vid:
            self.inpaint_size = int(self.vid.split('_')[-1]) // 2

    def forward(self, img):
        inpaint_mask = torch.ones_like(img)
        if 'inpaint' in self.vid:
            gt = img.clone()
            h,w = img.shape[-2:]
            inpaint_mask = torch.ones((h,w)).to(img.device)
            for ctr_x, ctr_y in [(1/2, 1/2), (1/4, 1/4), (1/4, 3/4), (3/4, 1/4), (3/4, 3/4)]:
                ctr_x, ctr_y = int(ctr_x * h), int(ctr_y * w)
                inpaint_mask[ctr_x - self.inpaint_size: ctr_x + self.inpaint_size, ctr_y - self.inpaint_size: ctr_y + self.inpaint_size] = 0
            input = (img * inpaint_mask).clamp(min=0,max=1)
        else:
            input, gt = img, img

        return input, gt, inpaint_mask.detach()


###################################  Code for ConvNeXt   ###################################
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, stage_blocks=0, strds=[2,2,2,2], dims=[96, 192, 384, 768], 
            in_chans=3, drop_path_rate=0., layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stage_num = len(dims)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, stage_blocks*self.stage_num)] 
        cur = 0
        for i in range(self.stage_num):
            # Build downsample layers
            if i > 0:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i-1], dims[i], kernel_size=strds[i], stride=strds[i]),
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=strds[i], stride=strds[i]),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                )                
            self.downsample_layers.append(downsample_layer)

            # Build more blocks
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(stage_blocks)]
            )
            self.stages.append(stage)
            cur += stage_blocks

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_list = []
        for i in range(self.stage_num):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out_list.append(x)
        return out_list[-1]


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
