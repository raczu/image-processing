import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from os import mkdir
from os.path import isdir


def split_colors(image: np.ndarray) -> tuple[np.ndarray, ...]:
    rgb = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    return tuple(np.array([np.copy(image) * rgb[i] for i in range(3)]))


def bayer_mask() -> np.array:
    # RGB MASK; 1st row R, 2nd row G, 3rd row B
    return np.array([[[0, 1], [0, 0]],
                     [[1, 0], [0, 1]],
                     [[0, 0], [1, 0]]])


def xtrans_mask() -> np.array:
    # RGB MASK; 1st row R, 2nd row G, 3rd row B
    return np.array([[[0, 0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 0, 1],
                      [0, 0, 0, 1, 0, 0],
                      [1, 0, 0, 0, 0, 0]],
                     [[1, 0, 0, 1, 0, 0],
                      [0, 1, 1, 0, 1, 1],
                      [0, 1, 1, 0, 1, 1],
                      [1, 0, 0, 1, 0, 0],
                      [0, 1, 1, 0, 1, 1],
                      [0, 1, 1, 0, 1, 1]],
                     [[0, 1, 0, 0, 0, 1],
                      [0, 0, 0, 1, 0, 0],
                      [1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0]]])


def to_cfa(image: np.ndarray, mask: np.array) -> np.ndarray:
    r, g, b = split_colors(image)

    height, width = image.shape[0:2]
    channels = {channel: np.ones((height, width, 3)) for channel in 'RGB'}

    for channel, mask_type in zip('RGB', mask):
        h_mask, w_mask = mask_type.shape[0:2]
        cfa_filter = np.array([[[mask_type[h % h_mask][w % w_mask] for _ in range(3)]
                                for w in range(width)] for h in range(height)])
        channels[channel] = channels[channel] * cfa_filter

    return r * channels['R'] + g * channels['G'] + b * channels['B']


def iterate_and_fill(channel: np.ndarray) -> np.ndarray:
    height, width = channel.shape[0:2]
    filled = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            if np.sum(channel[i, j]) != 0:
                filled[i, j] = channel[i, j]
                continue

            m = 0
            while True:
                m += 1

                # FIXME: some corners are missing
                if j + m <= width - 1 and channel[i, j + m] != 0:  # right
                    filled[i, j] = channel[i, j + m]
                    break

                if i + m <= height - 1 and channel[i + m, j] != 0:  # bottom
                    filled[i, j] = channel[i + m, j]
                    break

                if j - m >= 0 and channel[i, j - m] != 0:  # left
                    filled[i, j] = channel[i, j - m]
                    break

                if i - m <= 0 and channel[i - m, j] != 0:  # top
                    filled[i, j] = channel[i - m, j]
                    break

                if j + m <= width - 1 and i + m <= height - 1 and channel[i + m, j + m] != 0:  # bottom right
                    filled[i, j] = channel[i + m, j + m]
                    break

                if j - m >= 0 and i + m <= height - 1 and channel[i + m, j - m] != 0:  # top right
                    filled[i, j] = channel[i + m, j - m]
                    break

                if j + m <= width - 1 and i - m >= 0 and channel[i - m, j + m] != 0:  # bottom left
                    filled[i, j] = channel[i - m, j + m]
                    break

                if j - m >= 0 and i - m >= 0 and channel[i - m, j - m] != 0:  # top left
                    filled[i, j] = channel[i - m, j - m]
                    break

    return filled


def change_to_2d(data: np.ndarray, x_len: int, y_len: int) -> np.ndarray:
    return np.array([[np.sum(data[y][x]) for x in range(x_len)] for y in range(y_len)])


def merge_channels(destination: np.ndarray, channels: tuple[np.ndarray, ...]) -> None:
    height, width = destination.shape[0:2]

    for y in range(height):
        for x in range(width):
            destination[y, x, 0] = channels[0][y, x]
            destination[y, x, 1] = channels[1][y, x]
            destination[y, x, 2] = channels[2][y, x]


def nearest(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[0:2]
    interpolated = np.zeros((height, width, 3), dtype=np.uint8)
    r, g, b = split_colors(image)

    r2d = change_to_2d(r, width, height)
    g2d = change_to_2d(g, width, height)
    b2d = change_to_2d(b, width, height)

    r_filled = iterate_and_fill(r2d)
    g_filled = iterate_and_fill(g2d)
    b_filled = iterate_and_fill(b2d)

    merge_channels(interpolated, (r_filled, g_filled, b_filled))

    return interpolated


def bayer_kernels() -> list[np.array, ...]:
    # kernels based on:
    # https://www.cl.cam.ac.uk/teaching/0910/R08/work/essay-ls426-cfadetection.pdf

    kr = np.array([[1., 2., 1.],
                   [2., 4., 2.],
                   [1., 2., 1.]])

    kg = np.array([[0., 1., 0.],
                   [1., 4., 1.],
                   [0., 1., 0.]])

    kb = np.array([[.1, 2., 1.],
                   [2., 4., 2.],
                   [1., 2., 1.]])

    return [kr / 4, kg / 4, kb / 4]


def xtrans_kernels() -> list[np.ndarray, ...]:
    # kernels based on:
    # https://github.com/Bahrd/AppliedPythonology/blob/master/demosaickingX.py

    k = np.array([[0., 0., 0., 0., 0., 0.],
                  [0., 0.25, 0.5, 0.5, 0.25, 0.],
                  [0., 0.5, 1., 1., 0.5, 0.],
                  [0., 0.5, 1., 1., 0.5, 0.],
                  [0., 0.25, 0.5, 0.5, 0.25, 0.],
                  [0., 0., 0., 0., 0., 0.]])

    return [k / 2, k / 5, k / 2]


def demosaic_bilinear(image: np.ndarray, kernels: list[np.ndarray, ...]) -> np.ndarray:
    height, width = image.shape[0:2]
    r, g, b = split_colors(image)
    interpolated = np.zeros((height, width, 3))

    # changing size of arrays of channels from 3d to 2d
    r2d = change_to_2d(r, width, height)
    g2d = change_to_2d(g, width, height)
    b2d = change_to_2d(b, width, height)

    kr, kg, kb = kernels

    # convolution
    r_convolved = convolve2d(r2d, kr, 'same')
    g_convolved = convolve2d(g2d, kg, 'same')
    b_convolved = convolve2d(b2d, kb, 'same')

    merge_channels(interpolated, (r_convolved, g_convolved, b_convolved))

    return interpolated


def clear_axis(axs: np.ndarray) -> None:
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])


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

    group = parser.add_argument_group('filters')
    filters = group.add_mutually_exclusive_group(required=True)

    filters.add_argument(
        '--bayer',
        action='store_true',
        help='specify if BAYER filter should be applied to image')

    filters.add_argument(
        '--xtrans',
        action='store_true',
        help='specify if X-TRANS filter should be applied to image')

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

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError('Cannot find specified file')

    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)

    cfa = to_cfa(image, bayer_mask()) if args.bayer else to_cfa(image, xtrans_mask())

    if args.bilinear:
        demosaiced = demosaic_bilinear(cfa, bayer_kernels()) if args.bayer else \
            demosaic_bilinear(cfa, xtrans_kernels())
    elif args.nearest:
        demosaiced = nearest(cfa)

    mkdir('./assets') if args.save and not isdir('./assets') else None

    cfa = cfa.astype(np.uint8)
    cv2.imwrite(f'./assets/{"bayer" if args.bayer else "x-trans"}.bmp', cfa) if args.save else None

    demosaiced = demosaiced.astype(np.uint8)
    cv2.imwrite(f'./assets/demosaiced-{"bayer" if args.bayer else "x-trans"}-'
                f'{"nearest" if args.nearest else "bilinear"}.bmp',
                demosaiced) if args.save else None

    difference = image - demosaiced
    cv2.imwrite(f'./assets/difference-original-demosaiced-{"bayer" if args.bayer else "x-trans"}.bmp', difference) \
        if args.save else None

    fig, axs = plt.subplots(1, 4, figsize=(12, 6))
    clear_axis(axs)
    axs[0].imshow(image)
    axs[0].set_title('ORIGINAL')

    axs[1].imshow(cfa)
    axs[1].set_title(f'CFA {"(BAYER)" if args.bayer else "(X-TRANS)"}')

    axs[2].imshow(demosaiced)
    axs[2].set_title(f'DEMOSAICED {"(NEAREST)" if args.nearest else "(BILINEAR)"}')

    axs[3].imshow(image - demosaiced)
    axs[3].set_title('DIFFERENCE')

    plt.subplots_adjust(0.025, 0, 0.975, 1, hspace=0.01, wspace=0.05)
    plt.savefig('./assets/summary.png') if args.save else None
    plt.show()


if __name__ == '__main__':
    main()
