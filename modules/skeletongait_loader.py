"""
Custom SkeletonGait++ loader that handles the import issues
by manually resolving dependencies and creating the model class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from einops import rearrange
import copy

# Add OpenGait to path
opengait_path = os.path.join(os.path.dirname(__file__), "../OpenGait")
if opengait_path not in sys.path:
    sys.path.insert(0, opengait_path)

# Import OpenGait dependencies
from opengait.modeling.base_model import BaseModel
from opengait.modeling.modules import (
    HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, 
    SeparateBNNecks, SetBlockWrapper, conv3x3, conv1x1, 
    BasicBlock2D, BasicBlockP3D
)

class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super(AttentionFusion, self).__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, pose, sil):
        concat = torch.cat([pose, sil], dim=1)
        att = self.att(concat)
        att_pose, att_sil = torch.chunk(att, 2, dim=1)
        fused = att_pose * pose + att_sil * sil
        return fused

class CatFusion(nn.Module):
    def __init__(self, channels):
        super(CatFusion, self).__init__()
        self.conv = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, pose, sil):
        concat = torch.cat([pose, sil], dim=1)
        return self.conv(concat)

class PlusFusion(nn.Module):
    def __init__(self, channels):
        super(PlusFusion, self).__init__()

    def forward(self, pose, sil):
        return pose + sil

class SkeletonGaitPP(BaseModel):
    """
    SkeletonGait++ implementation with resolved dependencies
    """

    def build_network(self, model_cfg):
        in_C, B, C = model_cfg['Backbone']['in_channels'], model_cfg['Backbone']['blocks'], model_cfg['Backbone']['C']
        self.inference_use_emb = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        self.inplanes = 32 * C
        self.sil_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(1, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))

        self.map_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(2, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))

        self.sil_layer1 = self.make_layer(BasicBlock2D, 32 * C, 1, B[0], mode='2d')
        self.map_layer1 = self.make_layer(BasicBlock2D, 32 * C, 1, B[0], mode='2d')

        # Fusion layer - using simple plus fusion for now
        self.fusion = PlusFusion(32 * C)

        # Use 2D blocks for simplicity and compatibility
        self.layer2 = self.make_layer(BasicBlock2D, 64 * C, 2, B[1], mode='2d')
        self.layer3 = self.make_layer(BasicBlock2D, 128 * C, 2, B[2], mode='2d')
        self.layer4 = self.make_layer(BasicBlock2D, 256 * C, 1, B[3], mode='2d')

        self.FCs = SeparateFCs(16, 256*C, 128*C)
        self.BNNecks = SeparateBNNecks(16, 128*C, class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):
        # Handle stride as int or list
        if isinstance(stride, int):
            stride_val = stride
            stride_3d = [1, stride, stride] if mode == 'p3d' else stride
        else:
            stride_val = max(stride)
            stride_3d = [1] + stride if mode == 'p3d' else stride[0]
            
        if stride_val != 1 or self.inplanes != planes:
            if mode == '2d':
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes, stride_val),
                    nn.BatchNorm2d(planes)
                )
            elif mode == 'p3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride_3d, bias=False),
                    nn.BatchNorm3d(planes)
                )
        else:
            downsample = lambda x: x
        
        # Fix: Use correct parameter order for BasicBlock2D/P3D
        if mode == '2d':
            layers = [block(self.inplanes, planes, stride_val, downsample)]
            self.inplanes = planes
            for i in range(1, blocks_num):
                layers.append(block(self.inplanes, planes))
        else:  # p3d mode
            layers = [block(self.inplanes, planes, stride_3d, downsample)]
            self.inplanes = planes
            for i in range(1, blocks_num):
                layers.append(block(self.inplanes, planes))
                
        return SetBlockWrapper(nn.Sequential(*layers))

    def inputs_pretreament(self, inputs):
        """Ensure the same data augmentation for heatmap and silhouette"""
        pose_sils = inputs[0]
        new_data_list = []
        for pose, sil in zip(pose_sils[0], pose_sils[1]):
            sil = sil[:, np.newaxis, ...] # [T, 1, H, W]
            pose_h, pose_w = pose.shape[-2], pose.shape[-1]
            sil_h, sil_w = sil.shape[-2], sil.shape[-1]
            if sil_h != sil_w and pose_h == pose_w:
                cutting = (sil_h - sil_w) // 2
                pose = pose[..., cutting:-cutting]
            cat_data = np.concatenate([pose, sil], axis=1) # [T, 3, H, W]
            new_data_list.append(cat_data)
        new_inputs = [[new_data_list], inputs[1], inputs[2], inputs[3], inputs[4]]
        return super().inputs_pretreament(new_inputs)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        pose = ipts[0]
        pose = pose.transpose(1, 2).contiguous()
        assert pose.size(-1) in [44, 48, 88, 96]
        maps = pose[:, :2, ...]
        sils = pose[:, -1, ...].unsqueeze(1)

        del ipts
        map0 = self.map_layer0(maps)
        map1 = self.map_layer1(map0)
        
        sil0 = self.sil_layer0(sils)
        sil1 = self.sil_layer1(sil0)

        out1 = self.fusion(sil1, map1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3) # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        n, c, h, w = outs.size()

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        
        if self.inference_use_emb:
             embed = embed_2
        else:
             embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(pose * 255., 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval

def create_skeletongait_pp():
    """Factory function to create SkeletonGait++ instance"""
    return SkeletonGaitPP
