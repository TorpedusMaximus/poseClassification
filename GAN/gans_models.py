import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_classes):
        super(Generator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            # latent in 128x1x1
            nn.ConvTranspose2d(128 + num_classes, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            # out 1024x4x4

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            # out 512x8x8

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            # out 256x16x16

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            # out 128x32x32

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # out 64x64x64

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            # out 32x128x128

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out 3x256x256
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), 128, 1, 1)
        c = self.label_embedding(labels).view(labels.size(0), -1, 1, 1)
        z = torch.cat((z, c), dim=1)

        img = self.model(z)
        return img.view(img.size(0), 3, 256, 256)


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, 3 * 256 * 256)

        self.model = nn.Sequential(
            # input size being of 3 channels, 256x256
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            # output size being of 32 channels, 128x128

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # out 64x64x64

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            # out 128x32x32

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            # out 256x16x16

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            # out 512x8x8

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            # out 1024x4x4

            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out 1x1x1

            nn.Flatten(),
            nn.Sigmoid(),
            # final activation for T/F
        )

        self.auxiliary_layer = nn.Sequential(
            nn.Linear(1024, num_classes),  # Assuming 1024 is the size of the feature map before the final layer
            nn.Softmax(dim=1)
        )

    def forward(self, img, labels):
        c = self.label_embedding(labels).view(labels.size(0), 3, 256, 256)
        img = torch.cat((img, c), dim=1)

        validity = self.model(img)
        return validity

    def get_auxiliary_output(self, img):
        return self.auxiliary_layer(img.view(img.size(0), -1))
