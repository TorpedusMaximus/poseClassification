import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from GAN.gans_models import Discriminator, Generator


def get_class_labels(dataset):
    return [cls for cls, idx in sorted(dataset.class_to_idx.items(), key=lambda x: x[1])]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device} as a device")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = ImageFolder(root='./prepared', transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    latent_dim = 128
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    num_epochs = 100

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    adversarial_loss = nn.BCELoss()

    optimizer_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1,beta2))

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader):
            # Getting images from batch
            real_images = images.to(device)
            # Creating adversarial ground truths
            valid = torch.ones(real_images.shape[0], 1, device=device)
            fake = torch.zeros(real_images.shape[0], 1, device=device)
            # Configuring input
            real_images = real_images.to(device)

            """
            Training the discriminator
            """
            optimizer_discriminator.zero_grad()
            # Sample noise as generator input
            z = torch.randn(real_images.shape[0], latent_dim, 1, 1, device=device)
            # Generating a batch of images
            fake_images = generator(z)

            # Measuring discriminator's ability to classify real and fake images
            real_loss = adversarial_loss(discriminator(real_images), valid)
            fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)
            discriminator_loss = (real_loss + fake_loss) / 2
            # Backward pass nad optimize
            discriminator_loss.backward()
            optimizer_discriminator.step()

            """
            Training the generator
            """
            optimizer_generator.zero_grad()
            # Generating a batch of images
            generated_images = generator(z)
            # Adversarial loss
            generator_loss = adversarial_loss(discriminator(generated_images), valid)
            # Backward pass and optimize
            generator_loss.backward()
            optimizer_generator.step()

            """
            Progress monitoring
            """
            if (i + 1) % 30 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] " 
                    f"Batch {i + 1}/{len(dataloader)} "
                    f"Discriminator Loss: {discriminator_loss.item():.4f} "
                    f"Generator Loss: {generator_loss.item():.4f}"
                )

        # Saving generated images for every epoch
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                z = torch.randn(16, latent_dim, 1, 1, device=device)
                generated = generator(z).detach().cpu()

                # Get class labels
                class_labels = get_class_labels(train_dataset)

                # Display generated images with class labels
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for i in range(16):
                    ax = axes[i // 4, i % 4]
                    ax.imshow(np.transpose(generated[i], (1, 2, 0)))
                    ax.axis("off")
                    ax.set_title(class_labels[i % len(class_labels)])
                plt.savefig(f"./GAN/generated_images/generated_epoch_{epoch + 1}.png")


if __name__ == "__main__":
    main()