import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


def split_colors(image: np.ndarray) -> tuple[np.ndarray, ...]:
    RGB = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    return tuple(np.array([np.copy(image) * RGB[i] for i in range(3)]))


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
    R, G, B = split_colors(image)

    height, width = image.shape[0:2]
    channels = {channel: np.ones((height, width, 3)) for channel in 'RGB'}

    for channel, mask_type in zip('RGB', mask):
        h_mask, w_mask = mask_type.shape[0:2]
        bfilter = np.array([[[mask_type[h % h_mask][w % w_mask] for _ in range(3)]
                             for w in range(width)] for h in range(height)])
        channels[channel] = channels[channel] * bfilter

    return R * channels['R'] + G * channels['G'] + B * channels['B']


def mean(numbers: list[int, ...]) -> float:
    return sum(numbers) / len(numbers)


def iterate_and_fill(image: np.ndarray, channel: int, k: int) -> np.ndarray:
    height, width = image.shape[0:2]

    for i in range(height):
        for j in range(width):
            if np.sum(image[i][j]) != 0:
                continue

            neighbours = list()
            m = 0
            while True:
                m += 1

                # FIXME: some corners are missing
                if j + m <= width - 1:  # right
                    if np.sum(image[i][j + m]) != 0:
                        distance = np.sqrt((i - i) ** 2 + (j - (j + m)) ** 2)
                        neighbours.append((distance, image[i][j + m][channel]))

                if i + m <= height - 1:  # bottom
                    if np.sum(image[i + m][j]) != 0:
                        distance = np.sqrt((i - (i + m)) ** 2 + (j - j) ** 2)
                        neighbours.append((distance, image[i + m][j][channel]))

                if j - m >= 0:  # left
                    if np.sum(image[i][j - m]) != 0:
                        distance = np.sqrt((i - i) ** 2 + (j - (j - m)) ** 2)
                        neighbours.append((distance, image[i][j - m][channel]))

                if i - m <= 0:  # top
                    if np.sum(image[i - m][j]) != 0:
                        distance = np.sqrt((i - (i - m)) ** 2 + (j - j) ** 2)
                        neighbours.append((distance, image[i - m][j][channel]))

                if j + m <= width - 1 and i + m <= height - 1:  # bottom right
                    if np.sum(image[i + m][j + m]) != 0:
                        distance = np.sqrt((i - (i + m)) ** 2 + (j - (j + m)) ** 2)
                        neighbours.append((distance, image[i + m][j + m][channel]))

                if j - m >= 0 and i + m <= height - 1:  # top right
                    if np.sum(image[i + m][j - m]) != 0:
                        distance = np.sqrt((i - (i + m)) ** 2 + (j - (j - m)) ** 2)
                        neighbours.append((distance, image[i + m][j - m][channel]))

                if j + m <= width - 1 and i - m >= 0:  # bottom left
                    if np.sum(image[i - m][j + m]) != 0:
                        distance = np.sqrt((i - (i - m)) ** 2 + (j - (j + m)) ** 2)
                        neighbours.append((distance, image[i - m][j + m][channel]))

                if j - m >= 0 and i - m >= 0:  # top left
                    if np.sum(image[i - m][j - m]) != 0:
                        distance = np.sqrt((i - (i - m)) ** 2 + (j - (j - m)) ** 2)
                        neighbours.append((distance, image[i - m][j - m][channel]))

                if len(neighbours) > k:
                    break

            image[i][j][channel] = mean([value for d, value in sorted(neighbours, key=lambda x: x[0])[:k]])

    return image


def nearest(image: np.ndarray) -> np.ndarray:
    R, G, B = split_colors(image)

    R = iterate_and_fill(R, 0, 3)
    G = iterate_and_fill(G, 1, 3)
    B = iterate_and_fill(B, 2, 3)

    return R + G + B


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

    parser.add_argument(
        '--nearest',
        action='store_true',
        help='specify number of nearest neighbours to find',
        required=True
    )

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

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError('Cannot find specified file')

    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)

    CFA = to_cfa(image, bayer_mask()) if args.bayer else to_cfa(image, xtrans_mask())
    demosaiced = nearest(CFA)

    CFA = CFA.astype(np.uint8)
    cv2.imwrite(f'{"bayer" if args.bayer else "x-trans"}.bmp', CFA) if args.save else None

    demosaiced = demosaiced.astype(np.uint8)
    cv2.imwrite(f'demosaiced-{"bayer" if args.bayer else "x-trans"}-nearest.bmp', demosaiced) if args.save else None

    difference = image - demosaiced
    cv2.imwrite(f'difference-original-demosaiced-{"bayer" if args.bayer else "x-trans"}.bmp', difference)\
        if args.save else None

    fig, axs = plt.subplots(1, 4, figsize=(12, 6))
    clear_axis(axs)
    axs[0].imshow(image)
    axs[0].set_title('ORIGINAL')

    axs[1].imshow(CFA)
    axs[1].set_title(f'CFA {"(BAYER)" if args.bayer else "(X-TRANS)"}')

    axs[2].imshow(demosaiced)
    axs[2].set_title('DEMOSAICED (NEAREST)')

    axs[3].imshow(image - demosaiced)
    axs[3].set_title('DIFFERENCE')

    plt.subplots_adjust(0.025, 0, 0.975, 1, hspace=0.01, wspace=0.05)
    plt.show()


if __name__ == '__main__':
    main()
