import torch
import torch.nn as nn
import torch.nn.functional as F

# def Autoencoder3D():
#     encoder = nn.Sequential(
#             nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, stride=2, padding=1), # Adjust the out_channels as needed
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), # Further reduce dimensions
#             nn.LeakyReLU(0.1, inplace=True),
#     )

#     decoder = nn.Sequential(
#             nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.ConvTranspose3d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1), # Restore to original channel size
#             nn.LeakyReLU(0.1, inplace=True),
#     )



def conv(input_channels, output_channels, kernel_size, stride, dropout_rate):
    layer = nn.Sequential(
        nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding=(kernel_size - 1) // 2),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout_rate)
    )
    return layer

def deconv(input_channels, output_channels):
    layer = nn.Sequential(
        nn.ConvTranspose3d(input_channels, output_channels, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.1, inplace=True)
    )
    return layer

    
class Encoder(nn.Module):
    def __init__(self, input_channels, kernel_size, dropout_rate):
        super(Encoder, self).__init__()
        self.conv1 = conv(input_channels, 64, kernel_size, 2, dropout_rate)
        self.conv2 = conv(64, 128, kernel_size, 2, dropout_rate)
        self.conv3 = conv(128, 256, kernel_size, 2, dropout_rate)
        # self.conv4 = conv(256, 512, kernel_size, 2, dropout_rate)
                
    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        # out_conv4 = self.conv4(out_conv3)
        return out_conv1, out_conv2, out_conv3#, out_conv4 

# class Decoder(nn.Module):
#     def __init__(self, input_channels, kernel_size, dropout_rate):
#         super(Decoder, self).__init__()
#         self.deconv = nn.Sequential(nn.ConvTranspose3d(input_channels, output_channels, kernel_size=3, stride=2, padding=1),
#                                     nn.LeakyReLU(0.1, inplace=True)
#                                     )
                        
#     def forward(self, x):
#         out_deconv = self.deconv(x)
#         return out_deconv

class OutputFinalPrediction(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_spatial_size):
        super(OutputFinalPrediction, self).__init__()
        
        # 3D Convolution to process and reduce the time dimension
        # Assuming the time dimension is reduced to 1, adjust kernel_size and stride as needed
        spatial_padding = (kernel_spatial_size - 1) // 2
        self.conv3d = nn.Conv3d(in_channels=input_channels, out_channels=64, kernel_size=(31, kernel_spatial_size, kernel_spatial_size), stride=(1, 1, 1), padding=(0,spatial_padding,spatial_padding))
        
        # 2D Convolutional layers
        self.conv2d_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_spatial_size, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=128, out_channels=output_channels, kernel_size=kernel_spatial_size, padding=1)

    def forward(self, x):
        # Apply 3D convolution
        x = self.conv3d(x)
        
        # Remove the reduced time dimension
        x = x.squeeze(2)  # Assuming the time dimension is reduced to size 1

        # Apply 2D convolutions
        x = F.relu(self.conv2d_1(x))
        x = self.conv2d_2(x)
        return x


class spacetime_kernel_net(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels, 
                 kernel_size, 
                 dropout_rate, 
                 time_range,
                 h_dim = 64, 
                 w_dim = 64,
                 inp_dim = 2):
        super(spacetime_kernel_net, self).__init__()
        
        self.spatial_filter = nn.Conv2d(1, 1, kernel_size = 3, padding = 1, bias = False)   
        self.temporal_weights =  nn.Parameter(torch.ones(time_range), requires_grad=True)
        
        self.input_channels = input_channels
        self.input_length = input_channels//2
        self.time_range = time_range
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.inp_dim = inp_dim
        
        # Encoders
        self.encoder1 = Encoder(input_channels, kernel_size, dropout_rate)
        self.encoder2 = Encoder(input_channels, kernel_size, dropout_rate)
        self.encoder3 = Encoder(input_channels, kernel_size, dropout_rate)
        
        # Decoders
        # self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(256, 128)
        self.deconv1 = deconv(128, 64)
        self.deconv0 = deconv(64, 32)
        self.output_layer = OutputFinalPrediction(32 + input_channels, output_channels, kernel_spatial_size = 3)
        
    def forward(self, xx):
        #TODO: Follow creation of the u quants through st_mix and TFNet
        # u = u_mean + u_tilde + u_prime
        batch_size = xx.shape[0]
        time_samples = int(xx.shape[1]/self.inp_dim)
        # h_size = xx.shape[2]
        # w_size = xx.shape[3]
        print(f"Input dim:{self.inp_dim}")
        print(f"xx.shape:{xx.shape}")
        # apply spatial filter equally to velocity x and y
        u_intermediate = self.spatial_filter(xx.reshape(xx.shape[0]*xx.shape[1], 1, self.h_dim, self.w_dim))
        u_intermediate = u_intermediate.reshape(xx.shape[0], xx.shape[1], self.h_dim, self.w_dim) #(time, full_batch, h, w)
        
        # u_prime
        u_prime = (xx - u_intermediate)[:, -self.input_channels:] #(time, self.input_channels, h, w)
        
        # u_mean
        u_intermediate = u_intermediate.reshape(u_intermediate.shape[0], 
                                                u_intermediate.shape[1]//self.inp_dim, 
                                                self.inp_dim, self.h_dim, self.w_dim) # (time, full_batch//inp_dim, inp_dim, h, w)
        
        u_mean = self.temporal_weights[0] * u_intermediate[:,:self.input_length] # (time, self.input_length, inp_dim, h, w)
        for i in range(1, self.time_range):
            u_mean += self.temporal_weights[i] * u_intermediate[:, i:i + self.input_length] #(time, self.input_length, inp_dim, h, w)
        # u_mean = u_mean.reshape(u_mean.shape[0], -1, time_samples, self.h_dim, self.w_dim) #(time, self.input_length*inp_dim, h, w) which is (time, self.input_channels, h, w)
        
        # print(u_intermediate.shape, u_mean.shape)
        # u_tilde
        # u_intermediate = u_intermediate.reshape(u_intermediate.shape[0], -1, self.h_dim, self.w_dim) # (time, full_batch, h, w)
        u_intermediate = u_intermediate.reshape(u_intermediate.shape[0], -1, self.h_dim, self.w_dim) # (time, full_batch, h, w)
        print(f"Shape of u_intermediate:{u_intermediate.shape}")
        print(f"Shape of u_mean:{u_mean.shape}")

        u_tilde = u_intermediate[:,(self.time_range-1)*self.inp_dim:] - u_mean # (time, self.input_channels, h, w)

        #reshape for compatibility with 3D convolution (Need shape Batch, Channel, Time, h, w):
        
        out_conv1_mean, out_conv2_mean, out_conv3_mean = self.encoder1(u_mean)
        out_conv1_tilde, out_conv2_tilde, out_conv3_tilde = self.encoder2(u_tilde)
        out_conv1_prime, out_conv2_prime, out_conv3_prime = self.encoder3(u_prime)
        
        # out_deconv3 = self.deconv3(out_conv4_mean + out_conv4_tilde + out_conv4_prime)
        out_deconv2 = self.deconv2(out_conv3_mean + out_conv3_tilde + out_conv3_prime)
        out_deconv1 = self.deconv1(out_conv2_mean + out_conv2_tilde + out_conv2_prime + out_deconv2)
        out_deconv0 = self.deconv0(out_conv1_mean + out_conv1_tilde + out_conv1_prime + out_deconv1)        
        
        concat0 = torch.cat((xx[:, -self.input_channels:], out_deconv0), 1)
        print(f"shape of concat0:{concat0.shape}")
        out = self.output_layer(concat0)
        return out