import cv2
import time
import torch
import numpy as np
from PIL import Image
from diffusers import LDMSuperResolutionPipeline
import argparse


def crop_bbox(x, img):
    """
    Crop the image based on the bounding box coordinates

    Args:
        x: bounding box coordinates
        img: input image

    Returns:
        cropped_img: cropped image
    """
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cropped_img = img[c1[1]:c2[1], c1[0]:c2[0]]
    return cropped_img


def resize_larger_edge(img, max_edge):
    """
    Resize the larger edge of the image to max_edge while maintaining the aspect ratio

    Args:
        img: input image
        aspect_ratio: aspect ratio of the image
        max_edge: maximum size of the image

    Returns:
        resized_img: resized image
    """
    # Save the original image as temp
    resized_img = img
    width = int(img.shape[1])
    height = int(img.shape[0])
    aspect_ratio = width / height

    # Check if either width or height is larger than max pixels
    if width > max_edge or height > max_edge:
        # Resize the image while maintaining the aspect ratio
        if width > height:
            new_width = max_edge
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_edge
            new_width = int(new_height * aspect_ratio)

        # Resize the image
        resized_img = cv2.resize(
            img, (new_width, new_height))

    return resized_img


def resize_ratio(img, target_ratio):
    """
    Resize the image while maintaining the aspect ratio

    Args:
        img: input image
        target_ratio: target aspect ratio

    Returns:
        resized_img: resized image
    """
    width = int(img.shape[1])
    height = int(img.shape[0])
    original_ratio = width / height
    print(f'Original ratio: {original_ratio}, Target ratio: {target_ratio}')

    if original_ratio > target_ratio:
        new_width = int(target_ratio * height)
        new_height = height
    else:
        new_width = width
        new_height = int(width / target_ratio)

    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


class Latent:
    """
    Latent Diffusion Model for Super Resolution
    """

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Paper: High-Resolution Image Synthesis with Latent Diffusion Models
        model_id = "CompVis/ldm-super-resolution-4x-openimages"

        # load model
        self.pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
        self.pipeline = self.pipeline.to(device)

    def inference(self, target_img, sr_step=10):
        """
        Run inference on the input image

        Args:
            target_img: input image
            sr_step: number to control the effect of super-resolution, the larger, the better.

        Returns:
            upscaled_image: upscaled image
        """
        start_time = time.time()
        # Calculate the aspect ratio
        print(f'Original image shape: {target_img.shape}')
        # Calculate the original aspect ratio
        width = int(target_img.shape[1])
        height = int(target_img.shape[0])
        aspect_ratio = width / height
        max_edge = 150
        # Resize the larger edge of the image and maintain the aspect ratio
        resized_target = resize_larger_edge(target_img, max_edge)
        print(f'Resized image shape: {resized_target.shape}')
        # Convert img from BGR to RGB
        upscaled_image = Image.fromarray(
            cv2.cvtColor(resized_target, cv2.COLOR_BGR2RGB))
        # Run pipeline in inference (sample random noise and denoise)
        upscaled_image = self.pipeline(
            upscaled_image, num_inference_steps=sr_step, eta=1).images[0]
        # Convert img from RGB back to BGR
        upscaled_image = cv2.cvtColor(
            np.asarray(upscaled_image), cv2.COLOR_RGB2BGR)
        # Resize the image while maintaining the aspect ratio
        upscaled_image = resize_ratio(upscaled_image, aspect_ratio)
        stop_time = time.time()

        print(f'SR inference time: {stop_time-start_time}')
        return upscaled_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-img', default='/inference/images/bus.jpg')
    parser.add_argument('--sr-step', default=100, type=int)
    args = parser.parse_args()
    image = cv2.imread(args.input_img)
    print(f'Original image shape: {image.shape}')

    # Calculate the original aspect ratio
    width = int(image.shape[1])
    height = int(image.shape[0])
    aspect_ratio = width / height
    max_edge = 150
    # Resize the larger edge of the image and maintain the aspect ratio
    image = resize_larger_edge(image, max_edge)
    print(f'Resized image shape: {image.shape}')
    # Initialize the Latent Diffusion Model
    latent = Latent()
    # Run inference
    upscaled_image = latent.inference(image, args.sr_step)
    # Resize the image while maintaining the aspect ratio
    upscaled_image = resize_ratio(upscaled_image, aspect_ratio)

    cv2.imwrite(f'{args.input_img[:-4]}_sr.jpg', upscaled_image)
    print(f'SR image saved to {args.input_img[:-4]}_sr.jpg')
