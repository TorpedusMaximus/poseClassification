import os

import cv2
import torch

from GAN.gans_models import Generator
from constants import CLASS_TO_NUMBER


def generate_images(generator, latent_dim, class_num, device):
    class_label = torch.tensor([class_num], device=device)
    class_image = generator(torch.randn(1, latent_dim, 1, 1, device=device), class_label).detach().cpu()

    return class_image


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(100, 3800, 100):
        saved_model_path = f"./models/generator_model_v1.0.0_epoch_{i}.pth"

        latent_dim = 128
        num_classes = 35

        generator = Generator(num_classes).to(device)
        generator.load_state_dict(torch.load(saved_model_path))
        generator.eval()

        for class_number in CLASS_TO_NUMBER:
            for j in range(50):
                class_image = generate_images(generator, latent_dim, class_number, device)
                image_to_save = class_image.squeeze().numpy().transpose(1, 2, 0)
                image_to_save = image_to_save * 255
                if not os.path.exists(f"./images/epoch_{i}"):
                    os.mkdir(f"./images/epoch_{i}")
                class_name = CLASS_TO_NUMBER[class_number]
                if not os.path.exists(f"./images/epoch_{i}/{class_name}"):
                    os.mkdir(f"./images/epoch_{i}/{class_name}")
                path_image_to_save = f"./images/epoch_{i}/{class_name}/image_{class_name}_{j}.png"
                cv2.imwrite(path_image_to_save, image_to_save)


if __name__ == "__main__":
    main()
