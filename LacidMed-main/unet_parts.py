import torch  
import torch.nn as nn 

# This class defines a module that performs two consecutive convolutional operations, each followed by a ReLU activation.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initializes the DoubleConv class.
        Args:
            in_channels (int): Number of input channels (e.g., RGB images have 3 input channels).
            out_channels (int): Number of output channels after the convolution operations.
        """
        # Calls the constructor of the parent class nn.Module.
        # This ensures that the module is properly initialized as part of the PyTorch module framework.
        super().__init__()
        
        # Define a sequential block with two convolutional layers and ReLU activations.
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # First 2D convolutional layer
            # kernel_size=3 means the filter size is 3x3 pixels. Padding=1 ensures the output dimensions remain the same.
            nn.ReLU(inplace=True),  # Apply ReLU activation to introduce non-linearity (inplace saves memory).
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Second 2D convolutional layer
            nn.ReLU(inplace=True)  # Apply ReLU activation again
        )

    def forward(self, x):
        """
        Forward pass through the DoubleConv module.
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.conv_op(x)  # Pass the input through the sequential block


# This class defines a module that performs downsampling using a double convolution followed by max-pooling.
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initializes the DownSample class.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()  # Calls the constructor of the nn.Module base class.
        self.conv = DoubleConv(in_channels, out_channels)  # Double convolution to extract features.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max-pooling reduces the spatial dimensions by a factor of 2.

    def forward(self, x):
        """
        Forward pass through the DownSample module.
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Tuple[Tensor, Tensor]: 
                - down (Tensor): Features after the double convolution.
                - p (Tensor): Features after max-pooling (reduced spatial size).
        """
        down = self.conv(x)  # Extract features using the double convolution.
        p = self.pool(down)  # Downsample the feature map using max-pooling.
        return down, p  # Return both the extracted features and the downsampled output.


# This class defines a module for upsampling, combining an upsample operation with a double convolution.
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initializes the UpSample class.
        Args:
            in_channels (int): Number of input channels (typically from the downsampling path).
            out_channels (int): Number of output channels after the double convolution.
        """
        super().__init__()  # Calls the constructor of the nn.Module base class.
        
        # Transposed convolution layer for upsampling.
        # kernel_size=2 and stride=2 increase the spatial dimensions by a factor of 2.
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Double convolution to process the concatenated feature maps after upsampling.
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass through the UpSample module.
        Args:
            x1 (Tensor): Feature map from the previous layer (to be upsampled).
            x2 (Tensor): Feature map from the corresponding downsampling layer.
        Returns:
            Tensor: Output tensor after upsampling and double convolution.
        """
        x1 = self.up(x1)  # Upsample x1 using transposed convolution.
        
        # Concatenate the upsampled feature map (x1) with the corresponding downsampled feature map (x2).
        # Concatenation is done along the channel dimension (dim=1).
        x = torch.cat([x1, x2], dim=1)
        
        return self.conv(x)  # Pass the concatenated feature map through the double convolution.
