import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .attn import DAC_structure, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .RevIN import RevIN
from tkinter import _flatten
class layer_block(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(layer_block, self).__init__()
        self.conv_output = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 2))

        self.conv_output1 = nn.Conv2d(c_in, c_out, kernel_size=(1, k_size), stride=(1, 1), padding=(0, int( (k_size-1)/2 ) ) )
        self.output = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

        self.conv_output1 = nn.Conv2d(c_in, c_out, kernel_size=(1, k_size), stride=(1, 1) )
        self.output = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2))
        self.relu = nn.ReLU()
        
        
    def forward(self, input):
        conv_output = self.conv_output(input) # shape (B, D, N, T)

        conv_output1 = self.conv_output1(input)
        
        output = self.output(conv_output1)

        return self.relu( output+conv_output[...,-output.shape[3]:] )
    
class multi_scale_block(nn.Module):
    def __init__(self, c_in, c_out, seq_length, kernel_set):
        super(multi_scale_block, self).__init__()

        self.seq_length = seq_length
        self.layer_num = len(kernel_set)
        self.norm = nn.ModuleList()
        self.scale = nn.ModuleList()

        for i in range(self.layer_num):
            self.norm.append(nn.BatchNorm2d(c_out, affine=False))
        
        self.start_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1))

        self.scale.append(nn.Conv2d(c_out, c_out, kernel_size=(1, kernel_set[0]), stride=(1, 1)))

        for i in range(1, self.layer_num):
            
            self.scale.append(layer_block(c_out, c_out, kernel_set[i]))

        
    def forward(self, input): # input shape: B D N T

        scale = []
        scale_temp = input
        
        scale_temp = self.start_conv(scale_temp)
        
        for i in range(self.layer_num):
            scale_temp = self.scale[i](scale_temp)
            scale.append(scale_temp)
        return scale


class DCdetector(nn.Module):
    def __init__(self, win_size, enc_in, c_out, conv_channels, layer_num):
        super(DCdetector, self).__init__()
        
        self.win_size = win_size
        self.layer_num = layer_num
        
        self.kernel_set = [3, 2]
        self.enc_in = enc_in
        self.multi_scale_block = multi_scale_block(1, conv_channels, self.win_size, self.kernel_set)
        

    def forward(self, x):
        # torch.Size([64, 90, 55])Batch win_size channel
        B, L, M = x.shape
        
        revin_layer = RevIN(num_features=M)
        x = revin_layer(x, 'norm')
        # Batch, dim, win_size channel
        x = torch.unsqueeze(x, dim=1)
        # Batch, dim, channel, win_size 
        x = x.transpose(2,3)


        scale = self.multi_scale_block(x)
        print(scale[0].shape)
        exit()
        return
        

