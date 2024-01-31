import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(input_channels, output_channels, kernel_size, stride, dropout_rate):
    layer = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=(kernel_size - 1) // 2),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout_rate)
    )
    return layer

def deconv(input_channels, output_channels):
    layer = nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.1, inplace=True)
    )
    return layer

    
class Encoder(nn.Module):
    def __init__(self, input_channels, kernel_size, dropout_rate):
        super(Encoder, self).__init__()
        # self.conv1 = conv(input_channels, 64, kernel_size, 2, dropout_rate)
        # self.conv2 = conv(64, 128, kernel_size, 2, dropout_rate)
        # self.conv3 = conv(128, 256, kernel_size, 2, dropout_rate)
        # self.conv4 = conv(256, 512, kernel_size, 2, dropout_rate)        
        self.conv1 = conv(input_channels, 64, kernel_size, 2, dropout_rate)
        self.conv2 = conv(64, 128, kernel_size, 2, dropout_rate)
        self.conv3 = conv(128, 128, kernel_size, 2, dropout_rate)
        
                
    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        # out_conv4 = self.conv4(out_conv3)
        return out_conv1, out_conv2, out_conv3#, out_conv4 


class TFNet_multiscale(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels, 
                 max_kernel_size, 
                 dropout_rate, 
                 max_time_range,
                 h_dim = 64, 
                 w_dim = 64,
                 inp_dim = 2):
        super(TFNet_multiscale, self).__init__()
        # print("This runs init")
        self.kernel_size_large = max_kernel_size
        self.kernel_size_medium = max_kernel_size - 2
        self.kernel_size_small = max_kernel_size - 4

        self.time_range_large = max_time_range
        self.time_range_medium = max_time_range-2
        self.time_range_small = max_time_range-4

        # self.xx_time_range_large = xx
        # self.xx_time_range_medium = 
        # self.xx_time_range_small = 
        
        self.spatial_filter_1 = nn.Conv2d(1, 1, kernel_size = self.kernel_size_large, padding = int((self.kernel_size_large-1)//2), bias = False)   
        print(self.spatial_filter_1)
        self.spatial_filter_2 = nn.Conv2d(1, 1, kernel_size = self.kernel_size_medium, padding = int((self.kernel_size_medium-1)//2), bias = False)   
        self.spatial_filter_3 = nn.Conv2d(1, 1, kernel_size = self.kernel_size_small, padding = int((self.kernel_size_small-1)//2), bias = False)   
        self.temporal_weights_1 =  nn.Parameter(torch.ones(self.time_range_large), requires_grad=True)
        self.temporal_weights_2 =  nn.Parameter(torch.ones(self.time_range_medium), requires_grad=True)
        self.temporal_weights_3 =  nn.Parameter(torch.ones(self.time_range_small), requires_grad=True)

        self.input_channels = input_channels
        self.input_length = input_channels//2
        # self.time_range = max_time_range
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.inp_dim = inp_dim
        
        # for spatial kernel_size_large
        self.encoder1_kernel_large = Encoder(input_channels, self.kernel_size_large, dropout_rate)
        self.encoder2_kernel_large = Encoder(input_channels, self.kernel_size_large, dropout_rate)
        self.encoder3_kernel_large = Encoder(input_channels, self.kernel_size_large, dropout_rate)

        # for spatial kernel_size_medium
        self.encoder1_kernel_medium = Encoder(input_channels, self.kernel_size_medium, dropout_rate)
        self.encoder2_kernel_medium = Encoder(input_channels, self.kernel_size_medium, dropout_rate)
        self.encoder3_kernel_medium = Encoder(input_channels, self.kernel_size_medium, dropout_rate)

        # for spatial kernel_size_small
        self.encoder1_kernel_small = Encoder(input_channels, self.kernel_size_small, dropout_rate)
        self.encoder2_kernel_small = Encoder(input_channels, self.kernel_size_small, dropout_rate)
        self.encoder3_kernel_small = Encoder(input_channels, self.kernel_size_small, dropout_rate)
        
        # self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(128, 128)
        self.deconv1 = deconv(128, 64)
        self.deconv0 = deconv(64, 32)
        # self.output_layer = nn.Conv2d(32 + input_channels, output_channels, max_kernel_size, padding=(kernel_size - 1) // 2)

        self.output_layer_kernel_large = nn.Conv2d(32 + input_channels, output_channels, kernel_size=self.kernel_size_large, padding=int((self.kernel_size_large - 1)// 2))
        self.output_layer_kernel_medium = nn.Conv2d(32 + input_channels, output_channels, kernel_size=self.kernel_size_medium, padding=int((self.kernel_size_medium - 1)// 2))
        self.output_layer_kernel_small = nn.Conv2d(32 + input_channels, output_channels, kernel_size=self.kernel_size_small, padding=int((self.kernel_size_small - 1)// 2))

        # Initialize the combining convolutional layer
        # Assuming each stream output has 'output_channels' number of channels
        # self.combining_conv = nn.Conv2d(3 * output_channels, output_channels, kernel_size=1, padding=0)
        # self.combining_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        # Define the 3D convolutional layer
        # The number of input channels is 3 (one for each model prediction)
        # We use 2 output channels for the v_x and v_y components
        # The kernel size, stride, and padding can be adjusted based on your specific needs
        # self.combining_conv = nn.Conv3d(in_channels=3, out_channels=2, kernel_size=(1, 1, 1), stride=1, padding=0)

        # Define a 1D convolutional layer
        # The number of input channels is 3 (one for each model prediction)
        # The number of output channels is set to 1
        self.combining_conv = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)



    def forward(self, xx):
        # Stream 1 - Larger scale
        stream_1_out = self.process_stream(xx, self.spatial_filter_1, self.temporal_weights_1, self.time_range_large,
                                           encoders_list = [self.encoder1_kernel_large, self.encoder2_kernel_large, self.encoder3_kernel_large], 
                                           deconvs_list = [self.deconv2, self.deconv1, self.deconv0], 
                                           output_layer = self.output_layer_kernel_large)

        # Stream 2 - Medium scale
        stream_2_out = self.process_stream(xx, self.spatial_filter_2, self.temporal_weights_2, self.time_range_medium,
                                           encoders_list = [self.encoder1_kernel_medium, self.encoder2_kernel_medium, self.encoder3_kernel_medium], 
                                           deconvs_list = [self.deconv2, self.deconv1, self.deconv0], 
                                           output_layer = self.output_layer_kernel_medium)

        # Stream 3 - Smaller scale
        stream_3_out = self.process_stream(xx, self.spatial_filter_3, self.temporal_weights_3, self.time_range_small,
                                           encoders_list = [self.encoder1_kernel_small, self.encoder2_kernel_small, self.encoder3_kernel_small], 
                                           deconvs_list = [self.deconv2, self.deconv1, self.deconv0], 
                                           output_layer = self.output_layer_kernel_small)

        # Combine outputs from all streams
        # combined = torch.cat((stream_1_out, stream_2_out, stream_3_out), 1)
        # Final trainable feedforward layer
        # Assuming each stream output has shape [batch_size, channels, height, width]
        # out = self.combining_conv(combined)  # combining_conv is an nn.Conv2d layer


        combined = torch.stack((stream_1_out, stream_2_out, stream_3_out), dim=0)

        # combined is the input tensor of shape (3, 64, 2, 64, 64)
        # Transpose to bring the model_prediction dimension to the last position
        combined = combined.permute(1, 0, 2, 3, 4)  # New shape: (64, 2, 64, 64, 3)
        
        # Flatten the dimensions after the model_prediction dimension
        batch_size, predictions, channels, height, width = combined.shape
        combined = combined.reshape(batch_size, predictions, -1)  # New shape: (64, 3, 2*64*64)

        # Apply 1D convolution
        combined = self.combining_conv(combined)

        # Reshape back to (64, 2, 64, 64)
        out = combined.reshape(batch_size, channels, height, width)



        # # Reshape to merge the last two dimensions for 1D convolution
        # batch_size, channels, height, width = combined.shape[:4]
        # combined = combined.reshape(batch_size * channels * height * width, 3)
        # print(f"Combined has shape: {combined.shape}")
        # # Apply 1D convolution
        # combined = self.combining_conv(combined)
        
        # # Reshape back to the original dimensions (64, 2, 64, 64)
        # out = combined.squeeze(0).reshape(batch_size, channels, height, width)

        return out

    def process_stream(self, xx, spatial_filter, temporal_weights, time_range, encoders_list, deconvs_list, output_layer):
        encoder1, encoder2, encoder3 = encoders_list
        deconv2, deconv1, deconv0 = deconvs_list
        # print("==================")
        # print(f"time_range:{time_range}")
        # print(f"xx.shape:{xx.shape}")
        # apply spatial filter equally to velocity x and y
        u_intermediate = spatial_filter(xx.reshape(xx.shape[0]*xx.shape[1], 1, self.h_dim, self.w_dim))
        # print(f"u_intermediate.shape:{u_intermediate.shape}")
        u_intermediate = u_intermediate.reshape(xx.shape[0], xx.shape[1], self.h_dim, self.w_dim) #(time, full_batch, h, w)
        
        # u_prime
        u_prime = (xx - u_intermediate)[:, -self.input_channels:] #(time, self.input_channels, h, w)
        
        # u_mean
        u_intermediate = u_intermediate.reshape(u_intermediate.shape[0], 
                                                u_intermediate.shape[1]//self.inp_dim, 
                                                self.inp_dim, self.h_dim, self.w_dim) # (time, full_batch//inp_dim, inp_dim, h, w)
        # print("This runs 1")
        # print("Before reshaping u_mean:")
        # print(f"temporal_weights.shape:{temporal_weights.shape}")
        # print(f"u_intermediate.shape:{u_intermediate.shape}")
        # print(f"self.inp_dim:{self.inp_dim}")
        # print(f"self.input_length:{self.input_length}") # 26
        u_mean = temporal_weights[0] * u_intermediate[:,:self.input_length] # (time, self.input_length, inp_dim, h, w)
        for i in range(1, time_range):
            u_mean += temporal_weights[i] * u_intermediate[:, i:i + self.input_length] #(time, self.input_length, inp_dim, h, w)
        # print("After reshaping u_mean for good:")
        u_mean = u_mean.reshape(u_mean.shape[0], -1, self.h_dim, self.w_dim) #(time, self.input_length*inp_dim, h, w) which is (time, self.input_channels, h, w), time is length 52.
        
        # print(u_intermediate.shape, u_mean.shape)
        # u_tilde
        u_intermediate = u_intermediate.reshape(u_intermediate.shape[0], -1, self.h_dim, self.w_dim) # (time, full_batch, h, w)
        # print(f"Shape of u_intermediate:{u_intermediate.shape}")
        # print(f"Shape of u_mean:{u_mean.shape}")
        # print(f"Shape of u_intermediate[:,(time_range-1)*self.inp_dim:]:{u_intermediate[:,(self.time_range_large-1)*self.inp_dim:].shape}")
        # print(f"Shape of u_mean:{u_mean.shape}")
        # print(f"Using u_intermediate from time {(time_range-1)*self.inp_dim} to end")
        u_tilde = u_intermediate[:,(self.time_range_large-1)*self.inp_dim:] - u_mean # (time, self.input_channels, h, w)
        # print("This runs 2")
        out_conv1_mean, out_conv2_mean, out_conv3_mean = encoder1(u_mean)
        out_conv1_tilde, out_conv2_tilde, out_conv3_tilde = encoder2(u_tilde)
        out_conv1_prime, out_conv2_prime, out_conv3_prime = encoder3(u_prime)
        # print("Shapes of the u quantities:")
        # print(f"Shape of u_mean:{u_mean.shape}")
        # print(f"Shape of u_tilde:{u_tilde.shape}")
        # print(f"Shape of u_prime:{u_prime.shape}")


        # out_deconv3 = deconv3(out_conv4_mean + out_conv4_tilde + out_conv4_prime)
        out_deconv2 = deconv2(out_conv3_mean + out_conv3_tilde + out_conv3_prime)
        out_deconv1 = deconv1(out_conv2_mean + out_conv2_tilde + out_conv2_prime + out_deconv2)
        out_deconv0 = deconv0(out_conv1_mean + out_conv1_tilde + out_conv1_prime + out_deconv1)
        concat0 = torch.cat((xx[:, -self.input_channels:], out_deconv0), 1)

        stream_out = output_layer(concat0)
        # print(f"stream_out.shape:{stream_out.shape}")
        return stream_out


