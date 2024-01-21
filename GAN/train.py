import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from GAN.gans_models import Discriminator, Generator


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
    num_epochs = 2000000
    num_classes = len(train_dataset.classes)

    generator = Generator(num_classes).to(device)
    discriminator = Discriminator(num_classes).to(device)

    adversarial_loss = nn.BCELoss()
    # auxiliary_loss = nn.CrossEntropyLoss()

    optimizer_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1,beta2))

    idx_to_class = {}

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader):
            # Getting images from batch
            real_images = images.to(device)
            labels = labels.to(device)
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
            fake_images = generator(z, labels)

            # Measuring discriminator's ability to classify real and fake images
            real_loss = adversarial_loss(discriminator(real_images, labels), valid)
            fake_loss = adversarial_loss(discriminator(fake_images.detach(), labels), fake)
            # Auxiliary loss for classifying the real images
            # real_auxiliary_loss = auxiliary_loss(discriminator.get_auxiliary_output(real_images), labels)
            discriminator_loss = (real_loss + fake_loss) / 2 # + real_auxiliary_loss
            # Backward pass nad optimize
            discriminator_loss.backward()
            optimizer_discriminator.step()

            """
            Training the generator
            """
            optimizer_generator.zero_grad()
            # Generating a batch of images
            generated_images = generator(z, labels)
            # Adversarial loss
            generator_loss = adversarial_loss(discriminator(generated_images, labels), valid)
            # Auxiliary loss for the generated images
            # generated_auxiliary_loss = auxiliary_loss(discriminator.get_auxiliary_output(generated_images), labels)
            # generator_loss += generated_auxiliary_loss
            # Backward pass and optimize
            generator_loss.backward()
            optimizer_generator.step()

            class_to_idx = train_dataset.class_to_idx
            idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

            """
            Progress monitoring
            """
            if (i + 1) % 60 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] " 
                    f"Batch {i + 1}/{len(dataloader)} "
                    f"Discriminator Loss: {discriminator_loss.item():.4f} "
                    f"Generator Loss: {generator_loss.item():.4f}"
                )

        if (epoch + 1) % 100 == 0:
            if not os.path.exists("./GAN/generated_images"):
                os.mkdir("./GAN/generated_images")
            if not os.path.exists("./GAN/models"):
                os.mkdir("./GAN/models")
            torch.save(generator.state_dict(), f"./GAN/models/generator_model_v1.0.0_epoch_{epoch + 1}.pth")
            with torch.no_grad():
                z = torch.randn(16, latent_dim, 1, 1, device=device)
                random_labels = torch.randint(0, num_classes, (16,), device=device)

                # Map the random labels to class names
                random_class_names = [idx_to_class[label.item()] for label in random_labels]

                generated = generator(z, random_labels).detach().cpu()

                for j in range(16):
                    generated_image = generated[j]
                    generated_label = random_class_names[j]

                    generated_image_np = generated_image.squeeze().numpy().transpose(1, 2, 0)
                    generated_image_np = np.clip((generated_image_np + 1) / 2.0, 0, 1)

                    if not os.path.exists(f"./GAN/generated_images/epoch_{epoch + 1}"):
                        os.mkdir(f"./GAN/generated_images/epoch_{epoch + 1}")

                    image_filename = f"./GAN/generated_images/epoch_{epoch + 1}/generated_label_{generated_label}.png"
                    plt.imshow(generated_image_np)
                    plt.axis("off")
                    plt.savefig(image_filename)


if __name__ == "__main__":
    main()