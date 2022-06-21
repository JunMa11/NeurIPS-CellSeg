#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:23:19 2022
Author: MONAI

"""

from typing import Tuple, Union
import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets import ViT

class UNETR2D(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        debug: bool = False
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.debug = debug
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=2
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size): # x: (B, 256, 768)
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size) # (B, 16, 16, 768)
        x = x.permute(0, 3, 1, 2).contiguous() # (B, 768, 16, 16)
        return x

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in) # x: (B, 256,768), hidden_states_out: list, 12 elements, (B,256,768)
        enc1 = self.encoder1(x_in) # (1, 16, 256, 256)
        x2 = hidden_states_out[3] # (B, 256, 768)
        # self.proj_feat(x2, self.hidden_size, self.feat_size): (B, 768, 16,16) -> enc2: (B,32,128,128)
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size)) # hidden_size=768, self.feat_size=16
        x3 = hidden_states_out[6] # (B, 256, 768)
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size)) #(B, 768, 16,16) -> (B, 64, 64, 64)
        x4 = hidden_states_out[9] # (B, 256, 768)
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size)) # (B, 768, 16, 16) -> (B, 128, 32, 32)
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size) # (B, 768, 16, 16)
        dec3 = self.decoder5(dec4, enc4) # up -> cat -> ResConv; (B, 128, 32, 32)
        dec2 = self.decoder4(dec3, enc3) # (B, 64, 64, 64)
        dec1 = self.decoder3(dec2, enc2) # (B, 32, 128, 128)
        out = self.decoder2(dec1, enc1) # (B, 16, 256, 256)
        logits = self.out(out)
        
        if self.debug:
            return x, x2, x3,x4, hidden_states_out, enc1, enc2, enc3, enc4, dec4, dec3, dec2, dec1, logits 
        else:
            return logits
    

# model = UNETR2D(
#     in_channels=3, # 3 channels, R,G,B
#     out_channels=3,
#     img_size=(256, 256),
#     feature_size=16,
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=12,
#     pos_embed="perceptron",
#     norm_name="instance",
#     res_block=True,
#     dropout_rate=0.0,
#     debug=True
# ).cuda()

# from torchinfo import summary

# batch_size = 1
# summary(model, input_size=(batch_size, 3, 256, 256))

# x = torch.rand((1,3,256,256)).cuda()
# x, x2, x3,x4, hidden_states_out, enc1, enc2, enc3, enc4, dec4, dec3, dec2, dec1, logits  = model(x)
# print(logits.shape) # torch.Size([1, 3, 256, 256]) 




