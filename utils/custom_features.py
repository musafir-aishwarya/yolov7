import cv2
import time
import torch
import numpy as np
from PIL import Image
from diffusers import LDMSuperResolutionPipeline
import argparse


def crop_bbox(x, img):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cropped_img = img[c1[1]:c2[1], c1[0]:c2[0]]
    return cropped_img


class Latent:

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Paper: High-Resolution Image Synthesis with Latent Diffusion Models
        model_id = "CompVis/ldm-super-resolution-4x-openimages"

        # load model
        self.pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
        self.pipeline = self.pipeline.to(device)

    def inference(self, target_img, step=10):
        start_time = time.time()
        # Convert img from BGR to RGB
        upscaled_image = Image.fromarray(
            cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        # Run pipeline in inference (sample random noise and denoise)
        upscaled_image = self.pipeline(
            upscaled_image, num_inference_steps=step, eta=1).images[0]
        # Convert img from RGB back to BGR
        upscaled_image = cv2.cvtColor(
            np.asarray(upscaled_image), cv2.COLOR_RGB2BGR)
        stop_time = time.time()

        print(f'SR inference time: {stop_time-start_time}')
        return upscaled_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-img', default='/inference/images/bus.jpg')
    parser.add_argument('--sr-area-size', default=22500, type=int)
    parser.add_argument('--sr-step', default=100, type=int)
    args = parser.parse_args()
    image = cv2.imread(args.input_img)
    size = image.shape[0] * image.shape[1]

    print(
        f'Image shape: {image.shape}, Image size: {size}, Max SR area size: {args.sr_area_size}')

    if (size <= args.sr_area_size):
        print(f'Image size is smaller than SR area size, start SR...')
        latent = Latent()
        upscaled_image = latent.inference(image, args.sr_step)
        cv2.imwrite(f'{args.input_img[:-4]}_sr.jpg', upscaled_image)
        print(f'SR image saved to {args.input_img[:-4]}_sr.jpg')
    else:
        print(f'Image size is larger than SR area size, abort SR...')
