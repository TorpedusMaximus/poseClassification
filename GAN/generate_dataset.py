import os

import cv2
import torch

from GAN.gans_models import Generator
from GAN.generate_image import generate_images
from constants import CLASS_TO_NUMBER


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_model_path = f"./models/generator_model_v1.0.0_epoch_3600.pth"

    generator = Generator(len(CLASS_TO_NUMBER)).to(device)
    generator.load_state_dict(torch.load(saved_model_path))
    generator.eval()

    latent_dim = 128

    for class_number in CLASS_TO_NUMBER:
        print(f"Generating {CLASS_TO_NUMBER[class_number]} class...")
        for i in range(200):
            class_image = generate_images(generator, latent_dim, class_number, device)
            image_to_save = class_image.squeeze().numpy().transpose(1, 2, 0)
            image_to_save = image_to_save * 255
            if not os.path.exists(f"../prepared_generator"):
                os.mkdir(f"../prepared_generator")
            class_name = CLASS_TO_NUMBER[class_number]
            if not os.path.exists(f"../prepared_generator/{class_name}"):
                os.mkdir(f"../prepared_generator/{class_name}")
            path_image_to_save = f"../prepared_generator/{class_name}/{i}.png"
            cv2.imwrite(path_image_to_save, image_to_save)
        print(f"Generating {CLASS_TO_NUMBER[class_number]} class done!")


if __name__ == "__main__":
    main()
