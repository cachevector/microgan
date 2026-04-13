import torch
import torch.nn as nn

class MicroDCGANGenerator(nn.Module):
    def __init__(self, latent_dim=32, channels=1):
        super(MicroDCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        
        # Initial FC layer to 4x4 feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Convolutional layers for upsampling
        self.conv_blocks = nn.Sequential(
            # [128, 4, 4] -> [64, 8, 8]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # [64, 8, 8] -> [32, 16, 16]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # [32, 16, 16] -> [channels, 32, 32]
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        img = self.conv_blocks(x)
        return img

class MicroDCGANDiscriminator(nn.Module):
    def __init__(self, channels=1):
        super(MicroDCGANDiscriminator, self).__init__()
        self.channels = channels
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 32, bn=False), # [1, 32, 32] -> [32, 16, 16]
            *discriminator_block(32, 64),               # [32, 16, 16] -> [64, 8, 8]
            *discriminator_block(64, 128),              # [64, 8, 8] -> [128, 4, 4]
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity
