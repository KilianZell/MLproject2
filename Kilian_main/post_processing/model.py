import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):                                                   
    """
    Class that represents a double convolution (happenning at each step of the Unet)
    """
    def __init__(self, in_channels, out_channels):
        """
        Function that initiates DoubleConv.
        Input:
            - self: the DoubleConv that is initiated
            - in_channels: number of channels in the input image
            - out_channels: number of channels produced by the convolution 
        """                              
        super(DoubleConv, self).__init__()                                      #Avoid referring to the base class explicitly
        
        self.conv = nn.Sequential(                                              #Sequential container.
        #----1st Convulution-------    
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),          #Applies a 2D convolution over an input signal composed of several input planes.
            nn.BatchNorm2d(out_channels),                                       # Normalization
            nn.ReLU(inplace=True),                                              #Applies the rectified linear unit function element-wise (max(0,input))
        
        #----2nd Convulution-------
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Modelize one step of convolution.
        Input:
            - self: the DoubleConv in question
            - x: data to be convoluted
        Output:
            - The convulation result of x
        """                                                       
        return self.conv(x)

class UNET(nn.Module): 
    """
    Class that represents a UNET model (called one time per training)
    """                                                  
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],  #Single binary output (single output channel)
    ):
        """
        Function that initializes the UNET.
        Input:
            - self: The UNET in question
            - in_channels: number of channels in the input image 
                (defoult 3 because images are RGB)
            - out_channels: number of channels produced by the UNET
                (defoult 1 because out put masks are B&W)
            - Features: list of image feature at given steps 
                (at each step the number of features is divede/multiplied by 2)
        """
        super(UNET, self).__init__()                                            #Avoid referring to the base class explicitly

        self.ups = nn.ModuleList()                                              #Declare list of ups
        self.downs = nn.ModuleList()                                            #Declare list of downs
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #----Down part------- 
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))                 #Add double conv layer to downs
            in_channels = feature                                               #Looping

        #----Up part------- 
        for feature in reversed(features):                                      #Bottom-up approach
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,                #Adding skip connection (feature*2)
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))                     #2 conv. per up steps

        #----Lowest point------- 
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)              
        
        #----Final conv------- 
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)   #set number of channeles to 1

    def forward(self, x):
        """
        Function that modelizes an entire step of the UNET. Note that here skip connections are 
        used in assending steps to avoid the degradation problem.
        Input:
            - self: the UNET in question
            - x: the initial data
        Output:
            - self.final_conv(x): the result of the final convolution
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)                                          #will be used for skip connections
            x = self.pool(x)

        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]                               #reverse skip connections (bottom -> up)

        for idx in range(0, len(self.ups), 2):                                  #step of two as we want 2conv per up steps
            x = self.ups[idx](x)

            skip_connection = skip_connections[idx//2]                          #set step tp 1 for skip connections

            if x.shape != skip_connection.shape:                                #make sure we can compute even if input is not dividle by 16
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)                #add skip connections
            x = self.ups[idx+1](concat_skip)                                    #double conv

        return self.final_conv(x)                                               #returns the last conv of x after the unet


        """
        References:
            - https://arxiv.org/abs/1505.04597
            - https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
            - https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5 
        """