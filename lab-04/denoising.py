import os
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

from os import mkdir
from os.path import isdir
from functools import wraps


def benchmark(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'Function {func.__name__} executed in: {np.round(end - start, 2)} seconds')
        return result

    return wrapper


def box(size: int) -> np.ndarray:
    return np.ones((size, size)) * (1 / size ** 2)


def gaussian(size: int, sigma: float = 1) -> np.ndarray:
    center = size // 2
    kernel = np.array([[np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
                        for x in range(size)] for y in range(size)])

    return kernel / np.sum(kernel)


def median(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[0:2]
    radius = size // 2
    padded = np.pad(image, pad_width=((radius, radius), (radius, radius), (0, 0)), mode='edge')
    denoised = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            for c in range(3):
                denoised[y, x, c] = np.median(padded[y:y + size, x:x + size, c].flatten())

    return denoised


def convolution2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    height, width = image.shape[0:2]
    ks = kernel.shape[0] // 2
    padded = np.pad(image, pad_width=((ks, ks), (ks, ks), (0, 0)), mode='edge')
    convolved = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            for c in range(3):
                convolved[y, x, c] = np.sum(padded[y:y + kernel.shape[0], x:x + kernel.shape[0], c] * kernel)

    return convolved


KERNELS = {
    'box': box,
    'gaussian': gaussian
}


@benchmark
def denoise(image: np.ndarray, size: int, f: str = "") -> np.ndarray:
    return convolution2d(image, KERNELS[f](size)) if f else median(image, size)


def calculate_mse_and_mae(original: np.ndarray, restored: np.ndarray) -> tuple[np.float64, np.float64]:
    mse = np.mean(np.square(np.subtract(original.astype(np.float64), restored.astype(np.float64))))
    mae = np.mean(np.abs(original.astype(np.float64) - restored.astype(np.float64)))

    return np.float64(mse), np.float64(mae)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Arguments to configure a script')

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='specify a path to the original image')

    parser.add_argument(
        '--noise-image',
        type=str,
        required=True,
        help='specify a path to the noise image')

    parser.add_argument(
        '--save',
        action='store_true',
        help='specify if generated images should be saved')

    group = parser.add_argument_group('filters')
    group.add_argument(
        '--size',
        type=int,
        required=True,
        help='specify a size of the filter')

    filters = group.add_mutually_exclusive_group(required=True)
    filters.add_argument(
        '--box',
        action='store_true',
        help='specify if BOX filter should be applied to denoise the image')

    filters.add_argument(
        '--median',
        action='store_true',
        help='specify if MEDIAN filter should be applied to denoise the image')

    filters.add_argument(
        '--gaussian',
        action='store_true',
        help='specify if GAUSSIAN filter should be applied to denoise the image')

    args = parser.parse_args()

    if not os.path.isfile(args.noise_image) and not os.path.isfile(args.image):
        raise FileNotFoundError('Cannot find specified files')

    if args.size % 2 == 0 and args.size <= 2:
        raise ValueError('Filter size should be odd number and not less than 3')

    noise_image = cv2.cvtColor(cv2.imread(args.noise_image), cv2.COLOR_BGR2RGB)
    original = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)

    denoised = denoise(noise_image, args.size, 'box') if args.box \
        else denoise(noise_image, args.size) if args.median \
        else denoise(noise_image, args.size, 'gaussian')

    mkdir('./assets') if args.save and not isdir('./assets') else None
    cv2.imwrite(f'./assets/denoised-box-{args.size}x{args.size}.jpg' if args.box
                else f'./assets/denoised-median-{args.size}x{args.size}.jpg' if args.median
                else f'./assets/denoised-gaussian-{args.size}x{args.size}.jpg',
                cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)) if args.save else None

    difference = np.square(np.subtract(original, denoised))
    cv2.imwrite('./assets/difference-original-denoised.jpg',
                cv2.cvtColor(difference, cv2.COLOR_RGB2BGR)) if args.save else None

    mse, mae = calculate_mse_and_mae(original, denoised)
    print(f'{"MSE":<10} {"MAE":<10} {"FILTER":<10}\n'
          f'{np.round(mse, 2):<10} {np.round(mae, 2):<10} {"box" if args.box else "median" if args.median else "gaussian":<10}')

    fig, axs = plt.subplots(1, 4, figsize=(12, 6))
    axs[0].imshow(original)
    axs[0].set_title('ORIGINAL')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(noise_image)
    axs[1].set_title('NOISE IMAGE')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    axs[2].imshow(denoised)
    axs[2].set_title(f'DENOISED '
                     f'({"BOX FILTER" if args.box else "MEDIAN FILTER" if args.median else "GAUSSIAN FILTER"}'
                     f' {args.size}x{args.size})')
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    axs[3].imshow(difference)
    axs[3].set_title(f'DIFFERENCE')
    axs[3].set_xticks([])
    axs[3].set_yticks([])

    plt.subplots_adjust(0.025, 0, 0.975, 1, hspace=0.01, wspace=0.05)
    plt.savefig('./assets/summary.jpg') if args.save else None
    plt.show()


if __name__ == '__main__':
    main()
