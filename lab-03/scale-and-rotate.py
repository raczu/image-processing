import os
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from functools import wraps
from os import mkdir
from os.path import isdir


def benchmark(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'Function {func.__name__} executed in: {np.round(end - start, 2)} seconds')
        return result

    return wrapper


@benchmark
def nearest(image: np.ndarray, size: int) -> np.ndarray:
    interpolated = np.zeros((size, size, 3))
    img_size = image.shape[0:2][0]
    scale = size / img_size

    for y in range(size):
        for x in range(size):
            x_nearest, y_nearest = np.int16(np.round(x / scale)), np.int16(np.round(y / scale))

            if x_nearest == img_size:  # sometimes it has problem with indexes
                x_nearest -= 1

            if y_nearest == img_size:
                y_nearest -= 1

            interpolated[x][y] = image[x_nearest][y_nearest]

    return interpolated


@benchmark
def bilinear(image: np.ndarray, size: int) -> np.ndarray:
    interpolated = np.zeros((size, size, 3))
    img_size = image.shape[0:2][0]
    scale = size / img_size

    for y in range(size):
        for x in range(size):
            x_old = x / scale
            y_old = y / scale

            x1, y1 = min(np.int16(np.floor(x_old)), img_size - 1), min(np.int16(np.floor(y_old)), img_size - 1)
            x2, y2 = min(np.int16(np.ceil(x_old)), img_size - 1), min(np.int16(np.ceil(y_old)), img_size - 1)

            q11, q12 = image[x1][y1], image[x2][y1]
            q21, q22 = image[x1][y2], image[x2][y2]

            p1 = q12 * (x_old - np.floor(x_old)) + q11 * (1.0 - (x_old - np.floor(x_old)))
            p2 = q22 * (x_old - np.floor(x_old)) + q21 * (1.0 - (x_old - np.floor(x_old)))

            p = (1.0 - (y_old - np.floor(y_old))) * p2 + (y_old - np.floor(y_old)) * p1

            interpolated[x][y] = np.round(p)

    return interpolated


def u(s: float, a: float) -> float:
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a + 2) * (abs(s)**3) - (a + 3) * (abs(s)**2) + 1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a * (abs(s)**3) - (5 * a) * (abs(s)**2) + (8 * a) * abs(s)- 4 * a
    return 0


def padding(image: np.ndarray, height: int, width: int, n_colours: int = 3) -> np.ndarray:
    padded = np.zeros((height + 4, width + 4, n_colours))
    padded[2:height + 2, 2:width + 2, :n_colours] = image

    padded[2:height + 2, 0:2, :n_colours] = image[:, 0:1, :n_colours]
    padded[height + 2:height + 4, 2:width + 2, :] = image[height - 1:height, :, :]
    padded[2:height + 2, width + 2:width + 4, :] = image[:, width - 1:width, :]
    padded[0:2, 2:width + 2, :n_colours] = image[0:1, :, :n_colours]

    padded[0:2, 0:2, :n_colours] = image[0, 0, :n_colours]
    padded[height + 2:height + 4, 0:2, :n_colours] = image[height - 1, 0, :n_colours]
    padded[height + 2:height + 4, width + 2:width + 4, :n_colours] = image[height - 1, width - 1, :n_colours]
    padded[0:2, width + 2:width + 4, :n_colours] = image[0, width - 1, :n_colours]

    return padded


@benchmark
def keys(image: np.ndarray, size: int) -> np.ndarray:
    # function was taken from:
    # https://github.com/rootpine/Bicubic-interpolation/blob/master/bicubic.py

    interpolated = np.zeros((size, size, 3))
    img_size = image.shape[0:2][0]
    image = padding(image, img_size, img_size, 3)

    a = -0.5
    h = 1 / (size / img_size)

    for c in range(3):
        for j in range(size):
            for i in range(size):
                x, y = i * h + 2, j * h + 2

                x1 = 1 + x - np.floor(x)
                x2 = x - np.floor(x)
                x3 = np.floor(x) + 1 - x
                x4 = np.floor(x) + 2 - x

                y1 = 1 + y - np.floor(y)
                y2 = y - np.floor(y)
                y3 = np.floor(y) + 1 - y
                y4 = np.floor(y) + 2 - y

                mat_l = np.matrix([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])
                mat_m = np.matrix([[image[np.int16(y - y1), np.int16(x - x1), c],
                                    image[np.int16(y - y2), np.int16(x - x1), c],
                                    image[np.int16(y + y3), np.int16(x - x1), c],
                                    image[np.int16(y + y4), np.int16(x - x1), c]],
                                   [image[np.int16(y - y1), np.int16(x - x2), c],
                                    image[np.int16(y - y2), np.int16(x - x2), c],
                                    image[np.int16(y + y3), np.int16(x - x2), c],
                                    image[np.int16(y + y4), np.int16(x - x2), c]],
                                   [image[np.int16(y - y1), np.int16(x + x3), c],
                                    image[np.int16(y - y2), np.int16(x + x3), c],
                                    image[np.int16(y + y3), np.int16(x + x3), c],
                                    image[np.int16(y + y4), np.int16(x + x3), c]],
                                   [image[np.int16(y - y1), np.int16(x + x4), c],
                                    image[np.int16(y - y2), np.int16(x + x4), c],
                                    image[np.int16(y + y3), np.int16(x + x4), c],
                                    image[np.int16(y + y4), np.int16(x + x4), c]]])

                mat_r = np.matrix([[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]])

                interpolated[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

    return interpolated


INTERPOLATIONS = {
    'nearest': nearest,
    'bilinear': bilinear,
    'keys': keys
}


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    size, _ = image.shape[0:2]
    x_center, y_center = [np.round(((size + 1) / 2) - 1)] * 2
    rotated = np.zeros_like(image)
    radians = np.radians(angle)

    for y in range(size):
        for x in range(size):
            xt = size - 1 - x - x_center
            yt = size - 1 - y - y_center

            x_prime = np.round(xt * np.cos(radians) - yt * np.sin(radians))
            y_prime = np.round(xt * np.sin(radians) + yt * np.cos(radians))

            x_prime = np.int16(x_center - x_prime)
            y_prime = np.int16(y_center - y_prime)

            if 0 <= x_prime < size and 0 <= y_prime < size and x_prime >= 0 and y_prime >= 0:
                rotated[y_prime, x_prime, :] = image[y, x, :]

    return rotated


def shrink(image: np.ndarray, factor: float, algorithm: str) -> np.ndarray:
    size, _ = image.shape[0:2]
    new_size = np.int16(np.ceil(size * factor))

    return INTERPOLATIONS[algorithm](image, new_size)


def restore_default_size(image: np.ndarray, size: int, algorithm: str) -> np.ndarray:
    return INTERPOLATIONS[algorithm](image, size)


def calculate_mse_and_mae(original: np.ndarray, restored: np.ndarray) -> tuple[np.float64, np.float64]:
    mse = np.mean(np.square(np.subtract(original.astype(np.float64), restored.astype(np.float64))))
    mae = np.mean(np.abs(original.astype(np.float64) - restored.astype(np.float64)))

    return mse, mae


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Arguments to configure a script')

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='specify a path to the image')

    parser.add_argument(
        '--save',
        action='store_true',
        help='specify if generated images should be saved')

    parser.add_argument(
        '--shrink',
        type=float,
        required=True,
        help='specify how much image size should be shrank (0 <= factor <= 1)')

    group = parser.add_argument_group('interpolation')
    interpolation = group.add_mutually_exclusive_group(required=True)

    interpolation.add_argument(
        '--nearest',
        action='store_true',
        help='specify if NEAREST interpolation should be applied to image')

    interpolation.add_argument(
        '--bilinear',
        action='store_true',
        help='specify if BILINEAR interpolation should be applied to image')

    interpolation.add_argument(
        '--keys',
        action='store_true',
        help='specify if KEYS interpolation should be applied to image')

    parser.add_argument(
        '--rotate',
        type=float,
        required=True,
        help='specify the angle of rotation of the image')

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError('Cannot find specified file')

    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)

    if not 0 < args.shrink <= 1:
        raise ValueError('Shrink factor should be greater than 0 or less-equal to 1!')

    shrank = shrink(image, args.shrink, 'nearest') if args.nearest \
        else shrink(image, args.shrink, 'bilinear') if args.bilinear \
        else shrink(image, args.shrink, 'keys')

    size, _ = image.shape[0:2]
    restored = restore_default_size(shrank, size, 'nearest') if args.nearest \
        else restore_default_size(shrank, size, 'bilinear') if args.bilinear \
        else restore_default_size(shrank, size, 'keys')

    mkdir('./assets') if args.save and not isdir('./assets') else None

    cv2.imwrite(f'./assets/nearest-scaled.png' if args.nearest
                else './assets/bilinear-scaled.png' if args.bilinear
                else './assets/keys-scaled.png', shrank) if args.save else None

    cv2.imwrite(f'./assets/nearest-rescaled.png'
                if args.nearest else './assets/bilinear-rescaled.png' if args.bilinear
                else './assets/keys-rescaled.png', restored) if args.save else None

    mse, mae = calculate_mse_and_mae(image, restored)
    print(f'{"MSE":<10} {"MAE":<10} {"ALGORITHM":<10}\n'
          f'{np.round(mse, 2):<10} {np.round(mae, 2):<10} {"nearest" if args.nearest else "bilinear" if args.bilinear else "keys":<10}')

    rotated = rotate(image, args.rotate)
    cv2.imwrite(f'./assets/rotated-by-{args.rotate}.png', rotated) if args.save else None

    fig, axs = plt.subplots(1, 4, figsize=(12, 6), sharex=True, sharey=True)
    axs[0].imshow(image.astype(np.uint8))
    axs[0].set_title('ORIGINAL')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(np.clip(shrank, 0, 255, out=shrank).astype(np.uint8))
    axs[1].set_title(f'SCALED {"(NEAREST)" if args.nearest else "(BILINEAR)" if args.bilinear else "(KEYS)"}')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    axs[2].imshow(np.clip(restored, 0, 255, out=restored).astype(np.uint8))
    axs[2].set_title(f'RESCALED TO ORIGINAL'
                     f' {"(NEAREST)" if args.nearest else "(BILINEAR)" if args.bilinear else "(KEYS)"}')
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    axs[3].imshow(np.clip(rotated, 0, 255, out=rotated).astype(np.uint8))
    axs[3].set_title(f'ROTATED (BY {args.rotate})')
    axs[3].set_xticks([])
    axs[3].set_yticks([])

    plt.savefig('./assets/summary.png') if args.save else None
    plt.show()


if __name__ == '__main__':
    main()
