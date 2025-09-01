import torch  
import torch.nn as nn  

# Import the building blocks for the U-Net architecture
from unet_parts import DoubleConv, DownSample, UpSample

# Defines the U-Net architecture, a popular convolutional neural network for image segmentation.
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        Initializes the U-Net class.
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB images or 1 for grayscale images).
            num_classes (int): Number of output classes for segmentation.
        """
        super().__init__()  # Calls the constructor of the nn.Module base class to initialize the U-Net properly.
        
        # Downsampling path: Each step reduces spatial dimensions while increasing the number of feature channels.
        self.down_convolution_1 = DownSample(in_channels, 64)  # First downsampling block (input -> 64 channels)
        self.down_convolution_2 = DownSample(64, 128)         # Second downsampling block (64 -> 128 channels)
        self.down_convolution_3 = DownSample(128, 256)        # Third downsampling block (128 -> 256 channels)
        self.down_convolution_4 = DownSample(256, 512)        # Fourth downsampling block (256 -> 512 channels)

        # Bottleneck: Captures deep, high-level features with the largest receptive field.
        self.bottle_neck = DoubleConv(512, 1024)  # Double convolution at the bottleneck (512 -> 1024 channels)

        # Upsampling path: Each step increases spatial dimensions and decreases the number of feature channels.
        self.up_convolution_1 = UpSample(1024, 512)  # First upsampling block (1024 -> 512 channels)
        self.up_convolution_2 = UpSample(512, 256)   # Second upsampling block (512 -> 256 channels)
        self.up_convolution_3 = UpSample(256, 128)   # Third upsampling block (256 -> 128 channels)
        self.up_convolution_4 = UpSample(128, 64)    # Fourth upsampling block (128 -> 64 channels)
        
        # Final output layer: Reduces the number of channels to the number of classes.
        # kernel_size=1 ensures that no spatial dimensions are changed during this operation.
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the U-Net architecture.
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes, height, width), representing segmentation masks.
        """
        # Downsampling path
        down_1, p1 = self.down_convolution_1(x)  # Apply first downsampling block
        down_2, p2 = self.down_convolution_2(p1)  # Apply second downsampling block
        down_3, p3 = self.down_convolution_3(p2)  # Apply third downsampling block
        down_4, p4 = self.down_convolution_4(p3)  # Apply fourth downsampling block

        # Bottleneck
        b = self.bottle_neck(p4)  # Extract deep features at the bottleneck

        # Upsampling path (mirrors the downsampling path)
        up_1 = self.up_convolution_1(b, down_4)  # Apply first upsampling block and concatenate with down_4 features
        up_2 = self.up_convolution_2(up_1, down_3)  # Apply second upsampling block and concatenate with down_3 features
        up_3 = self.up_convolution_3(up_2, down_2)  # Apply third upsampling block and concatenate with down_2 features
        up_4 = self.up_convolution_4(up_3, down_1)  # Apply fourth upsampling block and concatenate with down_1 features

        # Final output
        out = self.out(up_4)  # Generate the final segmentation map using a 1x1 convolution
        return out  # Return the output segmentation map
