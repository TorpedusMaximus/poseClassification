import torch
import torch.nn as nn
from GAN.gans_models import Generator
import matplotlib.pyplot as plt
import numpy as np


def generate_images(generator, latent_dim, num_classes, device):
    random_label = torch.randint(0, num_classes, (1,), device=device)
    random_image = generator(torch.randn(1, latent_dim, 1, 1, device=device), random_label).detach().cpu()

    balance_label = torch.tensor([1], device=device)
    balance_image = generator(torch.randn(1, latent_dim, 1, 1, device=device), balance_label).detach().cpu()

    return random_image, balance_image


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_model_path = "./models/generator_model_v1.0.0_epoch_50.pth"

    latent_dim = 128
    num_classes = 36

    generator = Generator(num_classes).to(device)
    generator.load_state_dict(torch.load(saved_model_path))
    generator.eval()

    random_image, balance_image = generate_images(generator, latent_dim, num_classes, device)

    plt.subplot(1, 2, 1)
    plt.title("Random Label")
    plt.imshow(random_image.squeeze().numpy().transpose(1, 2, 0))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Balance Label")
    plt.imshow(balance_image.squeeze().numpy().transpose(1, 2, 0))
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
