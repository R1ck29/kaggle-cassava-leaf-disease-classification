from os.path import join, dirname
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Any

sys.path.append(join(dirname(__file__), "../../../.."))
from src.models.backbone.pytorch.hrnet import BasicBlock, HighResolutionNet

BN_MOMENTUM = 0.1

class PoseHigherResolutionNet(HighResolutionNet):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        extra = cfg.MODEL.EXTRA
        self.num_deconvs = extra.DECONV.NUM_DECONVS
        self.cat_deconv = list(cfg.MODEL.EXTRA.DECONV.CAT_OUTPUT)
        self.loss_config = cfg.LOSS
        self.final_layers = self._make_final_layers(cfg, self.pre_stage_channels[0])
        self.deconv_layers = self._make_deconv_layers(cfg, self.pre_stage_channels[0])

        self.pretrained_layers = cfg.MODEL.EXTRA.PRETRAINED_LAYERS
        
    def _make_final_layers(self, cfg, input_channels):
        dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1
        extra = cfg.MODEL.EXTRA

        final_layers = []
        output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
            if cfg.LOSS.WITH_AE_LOSS[0] else cfg.MODEL.NUM_JOINTS
        final_layers.append(nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        ))

        deconv_cfg = extra.DECONV
        for i in range(deconv_cfg.NUM_DECONVS):
            input_channels = deconv_cfg.NUM_CHANNELS[i]
            output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
                if cfg.LOSS.WITH_AE_LOSS[i+1] else cfg.MODEL.NUM_JOINTS
            final_layers.append(nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            ))

        return nn.ModuleList(final_layers)
    
    def _make_deconv_layers(self, cfg, input_channels):
        dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1
        extra = cfg.MODEL.EXTRA
        deconv_cfg = extra.DECONV

        deconv_layers = []
        for i in range(deconv_cfg.NUM_DECONVS):
            if deconv_cfg.CAT_OUTPUT[i]:
                final_output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
                    if cfg.LOSS.WITH_AE_LOSS[i] else cfg.MODEL.NUM_JOINTS
                input_channels += final_output_channels
            output_channels = deconv_cfg.NUM_CHANNELS[i]
            deconv_kernel, padding, output_padding = \
                self._get_deconv_cfg(deconv_cfg.KERNEL_SIZE[i])

            layers = []
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ))
            for _ in range(cfg.MODEL.EXTRA.DECONV.NUM_BASIC_BLOCKS):
                layers.append(nn.Sequential(
                    BasicBlock(output_channels, output_channels),
                ))
            deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        return nn.ModuleList(deconv_layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        
        x_list = []
        i = 0
        for trans in self.transition1:
            if trans is not None:
                x_list.append(trans(x))
            else:
                x_list.append(x)
            i +=1
        y_list = self.stage2(x_list)
        
        y_list.append(y_list[-1])
        x_list = []
        i = 0
        for trans in self.transition2:
            #if trans is not None:
            x_list.append(trans(y_list[i]))
            i +=1
        
        x_list = self.stage3(x_list)
        
        y_list = x_list
        
        y_list.append(y_list[-1])
        x_list = []
        i = 0
        for trans in self.transition3:
            #if trans is not None:
            x_list.append(trans(y_list[i]))
            i +=1
        y_list = self.stage4(x_list)
        
        final_outputs = []
        x = y_list[0]
        y = self.final_layers[0](x)
        final_outputs.append(y)

        #for i in range(self.num_deconvs):
        if self.cat_deconv[0]:
            x = torch.cat((x, y), 1)

        x = self.deconv_layers[0](x)
        y = self.final_layers[1](x)
        final_outputs.append(y)
        
        return final_outputs
    
    def init_weights(self, pretrained='', wo_head=False):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if pretrained:
            
            if '.ckpt' in pretrained:
                ckpt = torch.load(pretrained, map_location=device)
                state_dict = ckpt['state_dict']
                for key in list(state_dict.keys()):
                    new_key = key.replace('model.','')
                    state_dict[new_key] = state_dict[key]
                    state_dict.pop(key)
                pretrained_state_dict = state_dict
            else:
                pretrained_state_dict = torch.load(pretrained, map_location=device)
            
            if wo_head:
                pretrained_state_dict.pop('final_layers.0.weight')
                pretrained_state_dict.pop('final_layers.0.bias')
                pretrained_state_dict.pop('final_layers.1.weight')
                pretrained_state_dict.pop('final_layers.1.bias')
                pretrained_state_dict.pop('deconv_layers.0.0.0.weight')
                state = self.state_dict()
                state.update(pretrained_dict)
                self.load_state_dict(state)
                
            else:
                need_init_state_dict = {}
                for name, m in pretrained_state_dict.items():
                    if name.split('.')[0] in self.pretrained_layers \
                       or self.pretrained_layers[0] is '*':
                        if name in parameters_names or name in buffers_names:
                            need_init_state_dict[name] = m
                self.load_state_dict(need_init_state_dict, strict=False)
                

def get_pose_net(cfg, dir_path, **kwargs):
    model = PoseHigherResolutionNet(cfg, **kwargs)
    
    if cfg.MODEL.WEIGHT_PATH:
        cfg.MODEL.WEIGHT_PATH = join(dir_path, cfg.MODEL.WEIGHT_PATH)
    
    model.init_weights(cfg.MODEL.WEIGHT_PATH, cfg.MODEL.WO_HEAD)
    
    return model