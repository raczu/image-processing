import os
import argparse
import cv2
import numpy as np

from copy import deepcopy
from scipy.fftpack import dct, idct
from pywt import dwt2
from os import mkdir
from os.path import isdir


def lpf(image: np.ndarray, size: int) -> np.ndarray:
    copied = deepcopy(image)
    height, width = image.shape[0:2]
    h1, w1 = height // 2, width // 2

    copied[0:height, 0:w1 - size // 2] = 0
    copied[0:height, w1 + size // 2:width] = 0
    copied[0:h1 - size // 2, 0:width] = 0
    copied[h1 + size // 2:height, 0:width] = 0

    return copied


def hpf(image: np.ndarray, size: int) -> np.ndarray:
    copied = deepcopy(image)
    height, width = image.shape[0:2]
    h1, w1 = height // 2, width // 2

    copied[h1 - size // 2:h1 + size // 2, w1 - size // 2:w1 + size // 2] = 0

    return copied


def fourier(image: np.ndarray, size: int, f: str) -> np.ndarray:
    dft = np.fft.fft2(image)  # discrete Fourier transform
    c_shift = np.fft.fftshift(dft)  # shifting the frequency domain to the centre
    filtered = lpf(c_shift, size) if f == 'lpf' else hpf(c_shift, size)
    luc_shift = np.fft.ifftshift(filtered)  # shifting the frequency domain from centre to the left upper corner
    ift = np.fft.ifft2(luc_shift)  # inverse Fourier transform

    return np.abs(ift).astype(np.uint8)


def discrete(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dct_image = dct(dct(image.T, norm='ortho').T, norm='ortho')

    return dct_image, idct(idct(dct_image.T, norm='ortho').T, norm='ortho')


def wavelet(image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    return dwt2(image, 'haar')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Arguments to configure a script')

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='specify a path to the image')

    group = parser.add_argument_group('transforms')
    transforms = group.add_mutually_exclusive_group(required=True)
    transforms.add_argument(
        '--fourier',
        action='store_true',
        help='specify if FOURIER transform should be applied to the image')

    group.add_argument(
        '--size',
        type=int,
        help='specify a size of the low pass or high pass filter')

    filter_group = parser.add_argument_group('filters')
    filters = filter_group.add_mutually_exclusive_group()
    filters.add_argument(
        '--lpf',
        action='store_true',
        help='specify if LOW PASS filter should be applied to the image')

    filters.add_argument(
        '--hpf',
        action='store_true',
        help='specify if HIGH PASS filter should be applied to the image')

    transforms.add_argument(
        '--dct',
        action='store_true',
        help='specify if DCT filter should be applied to the image')

    transforms.add_argument(
        '--wavelet',
        action='store_true',
        help='specify if WAVELET transform should be applied to the image')

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError('Cannot find specified file')

    if args.fourier and not args.size:
        raise Exception('the following arguments are required: --size')

    if args.fourier and not (args.lpf or args.hpf):
        raise Exception('the following arguments are required: --lpf or --hpf')

    mkdir('./assets') if not isdir('./assets') else None
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    if args.fourier:
        approximated = fourier(image, args.size, 'lpf') if args.lpf else fourier(image, args.size, 'hpf')
        cv2.imwrite(f'./assets/approximated-fourier-{"lpf" if args.lpf else "hpf"}-{args.size}.jpg', approximated)

    if args.dct:
        amplitude, approximated = discrete(image)
        cv2.imwrite('./assets/amplitude-dct.jpg', cv2.applyColorMap(amplitude.astype(np.uint8), cv2.COLORMAP_JET))
        cv2.imwrite('./assets/approximated-dct.jpg', approximated)

    if args.wavelet:
        LL, (LH, HL, HH) = wavelet(image)
        cv2.imwrite('./assets/approximated-wavelet.jpg', cv2.convertScaleAbs(LL, alpha=0.5, beta=0))
        cv2.imwrite('./assets/lh-wavelet.jpg', cv2.convertScaleAbs(LH, alpha=1.5, beta=90))
        cv2.imwrite('./assets/hl-wavelet.jpg', cv2.convertScaleAbs(HL, alpha=1.5, beta=90))
        cv2.imwrite('./assets/hh-wavelet.jpg', cv2.convertScaleAbs(HH, alpha=1.5, beta=90))

    cv2.imwrite('./assets/lenna-gray.jpg', image)


if __name__ == '__main__':
    main()
