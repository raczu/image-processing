import os
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

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


@benchmark
def nearest(image: np.ndarray, size: int) -> np.ndarray:
    interpolated = np.zeros((size, size, 3))
    img_size = image.shape[0:2][0]
    scale = size / img_size

    for x in range(size):
        for y in range(size):
            x_nearest, y_nearest = np.int64(np.round(x / scale)), np.int64(np.round(y / scale))

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

    for x in range(size):
        for y in range(size):
            x_old = x / scale
            y_old = y / scale

            x1, y1 = min(np.int64(np.floor(x_old)), img_size - 1), min(np.int64(np.floor(y_old)), img_size - 1)
            x2, y2 = min(np.int64(np.ceil(x_old)), img_size - 1), min(np.int64(np.ceil(y_old)), img_size - 1)

            q11, q12 = image[x1][y1], image[x2][y1]
            q21, q22 = image[x1][y2], image[x2][y2]

            p1 = q12 * (x_old - np.floor(x_old)) + q11 * (1.0 - (x_old - np.floor(x_old)))
            p2 = q22 * (x_old - np.floor(x_old)) + q21 * (1.0 - (x_old - np.floor(x_old)))

            p = (1.0 - (y_old - np.floor(y_old))) * p2 + (y_old - np.floor(y_old)) * p1

            interpolated[x][y] = np.round(p)

    return interpolated


@benchmark
def keys(image: np.ndarray, size: int) -> np.ndarray:
    pass


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

    for x in range(size):
        for y in range(size):
            xt = size - 1 - x - x_center
            yt = size - 1 - y - y_center

            x_prime = np.round(xt * np.cos(radians) - yt * np.sin(radians))
            y_prime = np.round(xt * np.sin(radians) + yt * np.cos(radians))

            x_prime = np.int64(x_center - x_prime)
            y_prime = np.int64(y_center - y_prime)

            if x_prime < size and y_prime < size:
                rotated[x_prime][y_prime] = image[x][y]

    return rotated


def shrink(image: np.ndarray, factor: float, algorithm: str) -> np.ndarray:
    size, _ = image.shape[0:2]
    new_size = np.int64(np.ceil(size * factor))

    return INTERPOLATIONS[algorithm](image, new_size)


def restore_default_size(image: np.ndarray, size: int, algorithm: str) -> np.ndarray:
    return INTERPOLATIONS[algorithm](image, size)


def calculate_mse_and_mae(original: np.ndarray, restored: np.ndarray) -> tuple[float, float]:
    size = original.shape[0:2][0]
    mse, mae = 0.0, 0.0
    original, restored = original.astype(np.float64), restored.astype(np.float64)

    for x in range(size):
        for y in range(size):
            difference = np.sum(original[x][y] - restored[x][y])
            mse += difference ** 2
            mae += np.absolute(difference)
    size = size ** 2

    return mse / size, mae / size


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

    shrank = shrank.astype(np.uint8)
    cv2.imwrite(f'nearest-scaled.png' if args.nearest
                else 'bilinear-scaled.png' if args.bilinear
                else 'keys-scaled.png', shrank) if args.save else None

    restored = restored.astype(np.uint8)
    cv2.imwrite(f'nearest-rescaled.png'
                if args.nearest else 'bilinear-rescaled.png' if args.bilinear
                else 'keys-rescaled.png', restored) if args.save else None

    mse, mae = calculate_mse_and_mae(image, restored)
    print(f'MSE: {mse}\nMAE: {mae}')

    rotated = rotate(image, args.rotate)
    rotated = rotated.astype(np.uint8)
    cv2.imwrite(f'rotated-by-{args.rotate}.png', rotated) if args.save else None

    fig, axs = plt.subplots(1, 4, figsize=(12, 6), sharex=True, sharey=True)
    axs[0].imshow(image)
    axs[0].set_title('ORIGINAL')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(shrank)
    axs[1].set_title(f'SCALED {"(NEAREST)" if args.nearest else "(BILINEAR)" if args.bilinear else "(KEYS)"}')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    axs[2].imshow(restored)
    axs[2].set_title(f'RESCALED TO ORIGINAL'
                     f' {"(NEAREST)" if args.nearest else "(BILINEAR)" if args.bilinear else "(KEYS)"}')
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    axs[3].imshow(rotated)
    axs[3].set_title(f'ROTATED (BY {args.rotate})')
    axs[3].set_xticks([])
    axs[3].set_yticks([])

    plt.show()


if __name__ == '__main__':
    main()
