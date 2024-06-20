import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from arch_util import LayerNorm2d

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x


        
######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q,k,v    
    



class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v



#########################################
########### window-based self-attention #############
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
            
        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            raise Exception("Projection error!") 
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
    
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0]*self.win_size[1]
        nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H*W, H*W)
        
        # attn = (q @ k.transpose(-2, -1))

        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)
        
        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}"%(flops/1e9))
        return flops

########### self-attention #############
class Attention(nn.Module):
    def __init__(self, dim,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))


        if mask is not None:
            nW = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'






#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out



# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out




class Upsample_scale(nn.Module):
    def __init__(self, in_channel, out_channel, scale):
        super(Upsample_scale, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=scale, stride=scale),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out



class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x



# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x




class WHAformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',
                 modulator=False,cross_modulator=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.channel_Attn = MDTA(dim, num_heads)

        self.cab = CABlock(dim * 2)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim * 2)

        self.downsample_0 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)
        self.downsample_1 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)
        self.upsample_0 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.dw_conv1 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1, stride=1,
                                  groups=dim,
                                  bias=True)

        self.ffn = MFN(dim, 2.66, False)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
        
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}"

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1)
            input_mask_windows = window_partition(input_mask, self.win_size) # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask


        if self.cross_modulator is not None:
            shortcut = x
            x_cross = self.norm_cross(x)
            x_cross = self.cross_attn(x, self.cross_modulator.weight)
            x = shortcut + x_cross

        shortcut = x # shortcut: B, H * W, C

        x = self.drop_path(self.norm1(x))

        x = x.view(B, H, W, C)


        # cyclic shift
        if self.shift_size > 0:
            att_out = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            att_out = x # att_out: B, H, W, C



        # Part : gcap way (input: B, H, W, C; output: B, C, H, W)
        mdta_in = self.downsample_0(att_out.permute(0, 3, 1, 2))  # mdta_in : B, C*2, H//2, W//2
        cab_first = self.drop_path(self.cab(mdta_in)) + mdta_in

        # partition windows (input: B, H, W, C; output: nW*B, win_size, win_size, C)
        x_windows = window_partition(att_out, self.win_size)  # nW*B, win_size, win_size, C  N*C->C

        # Part : Window Channel MSA (input: nW*B, win_size, win_size, C; output: nW*B, win_size, win_size, C)
        chn_att_in = x_windows.permute(0, 3, 1, 2) # x_windows: nW*B, C, win_size, win_size
        bw, cw, hw, ww = chn_att_in.shape
        chn_att = chn_att_in + self.drop_path(self.channel_Attn(self.norm4(chn_att_in.reshape(bw, cw, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(bw, cw, hw, ww)))
        chn_att_out = chn_att.permute(0, 2, 3, 1) # chn_att_out: nW*B, win_size, win_size, C

        # Part gcap way (output: B, C, H, W)
        from_whtb = window_reverse(chn_att_out, self.win_size, H, W)  # B H' W' C
        from_whtb = self.downsample_1(from_whtb.permute(0, 3, 1, 2)) # B, C*2, H//2, W//2
        hybridAtt = from_whtb + cab_first
        cab_second = self.drop_path(self.cab(hybridAtt)) + hybridAtt

        gcap_out = self.upsample_0(cab_second)  # B, C, H, W

        chn_att_out = self.norm3(chn_att_out)

        spa_att_in = chn_att_out.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        spa_att = self.attn(spa_att_in, mask=attn_mask)  # nW*B, win_size*win_size, C
        spa_att_out = self.drop_path(spa_att)
        window_hybrid_att_out = spa_att_out.view(-1, self.win_size, self.win_size, C)

        # merge windows (intput: nW*B, win_size, win_size, C; output: B, H * W, C)
        att_out = window_reverse(window_hybrid_att_out, self.win_size, H, W)  # B H' W' C
        # reverse cyclic shift
        if self.shift_size > 0:
            whap_out = torch.roll(att_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            whap_out = att_out


        whap_out = whap_out.permute(0, 3, 1, 2) # (B, C, H, W)
        whtb_msa = gcap_out + whap_out


        msa = shortcut.view(B, H, W, C).permute(0, 3, 1, 2) + self.drop_path(whtb_msa)
        whtb_out = msa + self.drop_path(self.ffn(self.norm2(msa.reshape(B, C, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(B, C, H, W)))
        whtb_out = whtb_out.permute(0, 2, 3, 1).view(B, H * W, C)

        del attn_mask

        return whtb_out

class BasicWHAformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='ffn', shift_flag=True,
                 modulator=False,cross_modulator=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                WHAformerBlock(dim=dim, input_resolution=input_resolution,
                               num_heads=num_heads, win_size=win_size,
                               shift_size=0 if (i % 2 == 0) else win_size // 2,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop, attn_drop=attn_drop,
                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                               norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                               modulator=modulator, cross_modulator=cross_modulator)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                WHAformerBlock(dim=dim, input_resolution=input_resolution,
                               num_heads=num_heads, win_size=win_size,
                               shift_size=0,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop, attn_drop=attn_drop,
                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                               norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                               modulator=modulator, cross_modulator=cross_modulator)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"    

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x,mask)
        return x


class WHAformer(nn.Module):
    def __init__(self, img_size=256, in_chans=1, dd_in=1,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                 dowsample=Downsample, upsample=Upsample, shift_flag=False, modulator=False,
                 cross_modulator=False, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        conv_dpr = [drop_path_rate]*depths[3]
        dec_dpr = enc_dpr[::-1]

        # shallow layers
        self.conv_in = nn.Conv2d(dd_in, dd_in, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1, padding=0)

        # Input/Output
        self.input_proj = InputProj(in_channel=dd_in, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicWHAformerLayer(dim=embed_dim,
                                                  output_dim=embed_dim,
                                                  input_resolution=(img_size,
                                                img_size),
                                                  depth=depths[0],
                                                  num_heads=num_heads[0],
                                                  win_size=win_size,
                                                  mlp_ratio=self.mlp_ratio,
                                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  drop=drop_rate, attn_drop=attn_drop_rate,
                                                  drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                  norm_layer=norm_layer,
                                                  use_checkpoint=use_checkpoint,
                                                  token_projection=token_projection, token_mlp=token_mlp, shift_flag=shift_flag)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = BasicWHAformerLayer(dim=embed_dim * 2,
                                                  output_dim=embed_dim*2,
                                                  input_resolution=(img_size // 2,
                                                img_size // 2),
                                                  depth=depths[1],
                                                  num_heads=num_heads[1],
                                                  win_size=win_size,
                                                  mlp_ratio=self.mlp_ratio,
                                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  drop=drop_rate, attn_drop=attn_drop_rate,
                                                  drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                  norm_layer=norm_layer,
                                                  use_checkpoint=use_checkpoint,
                                                  token_projection=token_projection, token_mlp=token_mlp, shift_flag=shift_flag)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)
        self.encoderlayer_2 = BasicWHAformerLayer(dim=embed_dim * 4,
                                                  output_dim=embed_dim*4,
                                                  input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                                                  depth=depths[2],
                                                  num_heads=num_heads[2],
                                                  win_size=win_size,
                                                  mlp_ratio=self.mlp_ratio,
                                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  drop=drop_rate, attn_drop=attn_drop_rate,
                                                  drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                  norm_layer=norm_layer,
                                                  use_checkpoint=use_checkpoint,
                                                  token_projection=token_projection, token_mlp=token_mlp, shift_flag=shift_flag)
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)


        self.drop_path = DropPath(0.3)

        # Bottleneck
        self.conv = BasicWHAformerLayer(dim=embed_dim * 8,
                                        output_dim=embed_dim*8,
                                        input_resolution=(img_size // (2 ** 4),
                                                img_size // (2 ** 4)),
                                        depth=depths[3],
                                        num_heads=num_heads[3],
                                        win_size=win_size,
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=conv_dpr,
                                        norm_layer=norm_layer,
                                        use_checkpoint=use_checkpoint,
                                        token_projection=token_projection, token_mlp=token_mlp, shift_flag=shift_flag)



        # Decoder
        self.upsample_0 = upsample(embed_dim*8, embed_dim*4)
        self.decoderlayer_0 = BasicWHAformerLayer(dim=embed_dim * 8,
                                                  output_dim=embed_dim*8,
                                                  input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                                                  depth=depths[4],
                                                  num_heads=num_heads[4],
                                                  win_size=win_size,
                                                  mlp_ratio=self.mlp_ratio,
                                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  drop=drop_rate, attn_drop=attn_drop_rate,
                                                  drop_path=dec_dpr[:depths[4]],
                                                  norm_layer=norm_layer,
                                                  use_checkpoint=use_checkpoint,
                                                  token_projection=token_projection, token_mlp=token_mlp, shift_flag=shift_flag,
                                                  modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_1 = upsample(embed_dim*8, embed_dim*2)
        self.decoderlayer_1 = BasicWHAformerLayer(dim=embed_dim * 4,
                                                  output_dim=embed_dim*4,
                                                  input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                                                  depth=depths[5],
                                                  num_heads=num_heads[5],
                                                  win_size=win_size,
                                                  mlp_ratio=self.mlp_ratio,
                                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  drop=drop_rate, attn_drop=attn_drop_rate,
                                                  drop_path=dec_dpr[sum(depths[4:5]):sum(depths[4:6])],
                                                  norm_layer=norm_layer,
                                                  use_checkpoint=use_checkpoint,
                                                  token_projection=token_projection, token_mlp=token_mlp, shift_flag=shift_flag,
                                                  modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_2 = upsample(embed_dim*4, embed_dim*1)
        self.decoderlayer_2 = BasicWHAformerLayer(dim=embed_dim * 2,
                                                  output_dim=embed_dim*2,
                                                  input_resolution=(img_size // 2,
                                                img_size // 2),
                                                  depth=depths[6],
                                                  num_heads=num_heads[6],
                                                  win_size=win_size,
                                                  mlp_ratio=self.mlp_ratio,
                                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  drop=drop_rate, attn_drop=attn_drop_rate,
                                                  drop_path=dec_dpr[sum(depths[4:6]):sum(depths[4:7])],
                                                  norm_layer=norm_layer,
                                                  use_checkpoint=use_checkpoint,
                                                  token_projection=token_projection, token_mlp=token_mlp, shift_flag=shift_flag,
                                                  modulator=modulator, cross_modulator=cross_modulator)

        self.refinement = BasicWHAformerLayer(dim=embed_dim * 2,
                                              output_dim=embed_dim * 2,
                                              input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                              depth=1,
                                              num_heads=1,
                                              win_size=win_size,
                                              mlp_ratio=self.mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              drop_path=dec_dpr[sum(depths[4:6]):sum(depths[4:7])],
                                              norm_layer=norm_layer,
                                              use_checkpoint=use_checkpoint,
                                              token_projection=token_projection, token_mlp=token_mlp,
                                              shift_flag=shift_flag,
                                              modulator=modulator, cross_modulator=cross_modulator)

        self.apply(self._init_weights)




    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x, mask=None):


        # Input Projection
        y = self.input_proj(x)
        y = self.pos_drop(y)

        #Encoder
        conv0 = self.encoderlayer_0(y,mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0,mask=mask)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1,mask=mask)
        pool2 = self.dowsample_2(conv2)


        # Bottleneck
        conv4 = self.conv(pool2, mask=mask)




        #Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0,conv2],-1)
        deconv0 = self.decoderlayer_0(deconv0, mask=mask)



        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1,conv1],-1)
        deconv1 = self.decoderlayer_1(deconv1,mask=mask)




        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2,conv0],-1)
        deconv2 = self.decoderlayer_2(deconv2,mask=mask)

        refine = self.refinement(deconv2, mask=mask)

        # Output Projection
        y = self.output_proj(refine)

        return x + y

    def flops(self):
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso,self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops()+self.dowsample_0.flops(self.reso,self.reso)
        flops += self.encoderlayer_1.flops()+self.dowsample_1.flops(self.reso//2,self.reso//2)
        flops += self.encoderlayer_2.flops()+self.dowsample_2.flops(self.reso//2**2,self.reso//2**2)
        flops += self.encoderlayer_3.flops()+self.dowsample_3.flops(self.reso//2**3,self.reso//2**3)

        # Bottleneck
        flops += self.conv.flops()

        # Decoder
        flops += self.upsample_0.flops(self.reso//2**4,self.reso//2**4)+self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso//2**3,self.reso//2**3)+self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso//2**2,self.reso//2**2)+self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso//2,self.reso//2)+self.decoderlayer_3.flops()
        
        # Output Projection
        flops += self.output_proj.flops(self.reso,self.reso)
        return flops

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class CABlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # SimpleGate
        self.sg = SimpleGate()
        self.relu = nn.ReLU()

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)

        x = self.dropout1(x)
        return x



class MFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(MFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 3, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 3, hidden_features * 3, kernel_size=3, stride=1, padding=1, groups=hidden_features * 3, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 3, hidden_features * 3, kernel_size=5, stride=1, padding=2, groups=hidden_features * 3, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features * 3, hidden_features * 3, kernel_size=7, stride=1, padding=3,
                                   groups=hidden_features * 3, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu7 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 3, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 3, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)
        self.dwconv7x7_1 = nn.Conv2d(hidden_features * 3, hidden_features, kernel_size=7, stride=1, padding=3,
                                     groups=hidden_features, bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()
        self.relu7_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 3, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3, x3_3 = self.relu3(self.dwconv3x3(x)).chunk(3, dim=1)
        x1_5, x2_5, x3_5 = self.relu5(self.dwconv5x5(x)).chunk(3, dim=1)
        x1_7, x2_7, x3_7 = self.relu7(self.dwconv7x7(x)).chunk(3, dim=1)

        x1 = torch.cat([x1_3, x1_5, x1_7], dim=1)
        x2 = torch.cat([x2_3, x2_5, x2_7], dim=1)
        x3 = torch.cat([x3_3, x3_5, x3_7], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))
        x3 = self.relu7_1(self.dwconv7x7_1(x3))

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.project_out(x)

        return x


