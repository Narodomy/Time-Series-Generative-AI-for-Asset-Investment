import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Input Channels (c_in) = Input Features from original is RGB so c_in = 3 but we have 2 features Close price and Volume therefore c_in = 2
# Output Channels (c_out) = Output Features same as Input Channels (c_in)
# Time Embedding Dimension (time_dim) = timestep in Diffusion Model
# note __init__ = blueprint define the nn first and not yet active until call -> forward function in the class
# channels = nums of feature
# shape of x = [B,C,H,W] in image case but our project is [B,C,L]
class SelfAttention(nn.Module):
    def __init__(self, channels):
        # super(SelfAttention, self).__init__() # -> old version
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True) # paramters (channels, num_heads) when output will be [B, L, C] = [Batch, length, Channel]
        self.ln = nn.LayerNorm([channels]) # ln = (L)ayer(N)orm likes GroupNorm but better than because MHA is very sensitive with input scale
        # ff_self = Feed-Forward (Network)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2) # for image processing
        x_swapped = x.swapaxes(1, 2) # [B, C, L] such as [4, 128, 1024] -> swapaxes(1, 2) -> [B, L, C] = [4, 1024, 128]
        x_ln = self.ln(x_swapped)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln) # mha(Query, Key, Value), _ = Attention Weights
        attention_value = attention_value + x_swapped # this is Residual Connection No.1
        attention_value = self.ff_self(attention_value) + attention_value # Residual Connection No.2
        return attention_value.swapaxes(1, 2) # Swap them from [B, L, C] to [B, C, L]

class DoubleConv(nn.Module):
    # residual = switch for shortcut connection (skip connection that idea comes from ResNet)
    # c_in = input_channels, c_out = output_channels, c_mid = mid_channels
    # c_in -> c_mid -> c_out
    def __init__(self, c_in, c_out, c_mid=None, residual=False):
        super().__init__()
        self.residual = residual
        if not c_mid:
            c_mid = c_out
        self.double_conv = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, c_mid),
            nn.GELU(),
            nn.Conv1d(c_mid, c_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, c_out),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    # SiLU comes from Sigmoid Linear Unit
    def __init__(self, c_in, c_out, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(c_in, c_in, residual=True),
            DoubleConv(c_in, c_out),
        )

        # covert from the time_dim = emb_dim
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                c_out
            )
        )
        
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        # shape of emb_layer(t) = [B, C, L] or [Batch, Channel, length]
        emb = self.emb_layer(t)[:, :, None].repeat(1,1, x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, c_in, c_out, emb_dim=256):
        super().__init__()
        # Upsample to 1D
        # mode meaning: bilinear = 2D, linear = 1D
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False) # if Maxpool (Down Side) means Shrink, Upsample (Up Side) means Expand
        self.conv = nn.Sequential(
            DoubleConv(c_in, c_in, residual=True),
            DoubleConv(c_in, c_out, c_mid=c_in // 2),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                c_out
            ),
        )

    def forward(self, x, skip_x, t):
        # 1. Expand x (from lower level) to double its length
        x = self.up(x)

        # 2. Calculate the length difference between skip_x (from Down path) and x (from Up path)
        #    This is crucial because Conv1d padding might cause a 1-pixel (or 1-step) mismatch.
        #    e.g., skip_x.size(L=60) - x.size(L=59) = 1
        diff_len = skip_x.size()[2] - x.size()[2]

        # 3. Crop the skip_x (the longer one) to match the length of x (the shorter one)
        #    We crop symmetrically by taking 'diff_len // 2' from the start and the rest from the end.
        #    e.g., if diff_len=1, crop [:, :, 0:59] (0 from start, 1 from end)
        skip_x_cropped = skip_x[:, :, diff_len // 2 : skip_x.size()[2] - (diff_len - diff_len // 2)]
        
        # 4. Concatenate (stack) the cropped skip_x and x along the Channel (feature) dimension
        #    e.g., [B, 256, 59] + [B, 256, 59] -> [B, 512, 59]
        x = torch.cat([skip_x_cropped, x], dim=1) # x and skip_x = (B, C, L) and dim=0 means (B) Batch, dim=1 means (C) Channels (Features), dim=2 means (L) Length 
        
        # 5. Run the concatenated tensor through the convolution block
        x = self.conv(x)
        
        # 6. Add the time embedding
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb

# Input Channels (c_in) = Input Features from original is RGB so c_in = 3 but we have 2 features Close price and Volume therefore c_in = 2
# Output Channels (c_out) = Output Features same as Input Channels (c_in)
# Time Embedding Dimension (time_dim) = timestep in Diffusion Model
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        # self.sa1 = SelfAttention(128, 32)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        # self.sa2 = SelfAttention(256, 16)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        # self.sa3 = SelfAttention(256, 8)
        self.sa3 = SelfAttention(256)


        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        # self.sa4 = SelfAttention(128, 16)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        # self.sa5 = SelfAttention(64, 32)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        # self.sa6 = SelfAttention(64, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv1d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        # t arrives as a 1D tensor (a "vector") of timesteps, one for each sample in the batch.
        # It is NOT a single number (scalar) and NOT the final 256-dim vector yet.
        # e.g., if Batch Size (B) = 4, t might be: 
        #       tensor([150, 50, 800, 150])
        #       Shape: [B] or [4]
        
        # 1. "Prep" the timestep tensor 't'.
        #    .type(torch.float): PE algorithms (sin/cos) require floats, not ints (e.g., 150 -> 150.0).
        #    .unsqueeze(-1):  The PE function expects a 2D input [B, 1]. This adds a dummy dimension
        #                     at the end.
        # e.g., t shape is now: [B, 1] or [4, 1]
        #       t is now: tensor([[150.0],
        #                        [ 50.0],
        #                        [800.0],
        #                        [150.0]])
        t = t.unsqueeze(-1).type(torch.float)
        # 2. "Process" t into the final 't_vector' (the "smart" signature vector).
        #    self.pos_encoding expands the last dimension (from 1) to time_dim (e.g., 256).
        # e.g., Final t shape: [B, time_dim] or [4, 256]
        #       t is now: tensor([[-0.8, 0.1, ..., 0.5],  # Signature for t=150
        #                       [ 0.2, 0.9, ..., -0.3], # Signature for t=50
        #                       [ 0.7, -0.4, ..., 0.1], # Signature for t=800
        #                       [-0.8, 0.1, ..., 0.5]]) # Signature for t=150 (again)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
    
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)

        x = self.up2(x, x2, t)
        x = self.sa5(x)

        x = self.up3(x, x1, t)
        x = self.sa6(x)

        output = self.outc(x)
        return output