<div id="top"></div>

# Image processing

![GitHub last commit](https://img.shields.io/github/last-commit/raczu/image-processing)
![python](https://img.shields.io/badge/python-3.9-blue.svg)

Labolatory assignments related to image processing using python3.9+ (due to the use of typing methods supported from version 3.9). The tasks contain solutions to a variety of challanges, including the use of self-written interpolation algorithms, conversion of an image into a CFA (bayer, X-TRANS), image scaling or rotation and denoising.

## Table of contents

* [Solutions](#solutions)
* [Examples](#examples)
* [License](#license)
* [Contact](#contact)

## Solutions
* [lab-01 (rolling shutter effect simulation)](https://github.com/raczu/image-processing/tree/main/lab-01)
* [lab-02 (demosaicing)](https://github.com/raczu/image-processing/tree/main/lab-02)
* [lab-03 (scaling and rotating raster images)](https://github.com/raczu/image-processing/tree/main/lab-03)
* [lab-04 (denoising images)](https://github.com/raczu/image-processing/tree/main/lab-04)
* [lab-05 (orthogonal basis)](https://github.com/raczu/image-processing/tree/main/lab-05)

## Examples
All scripts are configurable through the use of flags to check which options can be used to execute script, simply use the help flag.
```bash
$ python3.9 script-name.py -h
```

<br />

| [lab-01 (rolling shutter effect simulation)](https://github.com/raczu/image-processing/tree/main/lab-01) |
| :-: |
| <img src="https://github.com/raczu/image-processing/blob/main/lab-01/assets/rolling-shutter-simulation.gif" /> |

```bash
$ python3.9 rshutter.py -b 5 -l 3 --save
```

<br />

| [lab-02 (demosaicing)](https://github.com/raczu/image-processing/tree/main/lab-02) |
| :-: |
| <img src="https://github.com/raczu/image-processing/blob/main/lab-02/image.bmp" width="24.5%" /> <img src="https://github.com/raczu/image-processing/blob/main/lab-02/assets/bayer.png" width="24.5%" /> <img src="https://github.com/raczu/image-processing/blob/main/lab-02/assets/demosaiced-bayer-bilinear.png" width="24.5%" /> <img src="https://github.com/raczu/image-processing/blob/main/lab-02/assets/difference-original-demosaiced-bayer-bilinear.png" width="24.5%" /> |

```bash
$ python3.9 demosaicing.py --image ./image.bmp --bayer --nearest --save
```

<br />

| [lab-03 (scaling and rotating raster images)](https://github.com/raczu/image-processing/tree/main/lab-03) |
| :-: |
| <img src="https://github.com/raczu/image-processing/blob/main/lab-03/image.jpg" width="33%" /> <img src="https://github.com/raczu/image-processing/blob/main/lab-03/assets/keys-rescaled.png" width="33%" /> <img src="https://github.com/raczu/image-processing/blob/main/lab-03/assets/rotated-by-22.5.png" width="33%" /> |

```bash
$ python3.9 scale-and-rotate.py --image ./image.jpg --shrink 0.65 --rotate 65 --nearest --save
```

<br />

| [lab-04 (denoising images)](https://github.com/raczu/image-processing/tree/main/lab-04) |
| :-: |
| <img src="https://github.com/raczu/image-processing/blob/main/lab-04/noise-leopard.jpg" width="33%" /> <img src="https://github.com/raczu/image-processing/blob/main/lab-04/assets/denoised-gaussian-7x7.jpg" width="33%" /> <img src="https://github.com/raczu/image-processing/blob/main/lab-04/assets/difference-original-denoised-gaussian-7x7.jpg" width="33%" /> |

```bash
$ python3.9 denoising.py --image ./image.jpg --noise-image ./noise-leopard.jph --box --size 7 --save
```

<br />

| [lab-05 (orthogonal basis)](https://github.com/raczu/image-processing/tree/main/lab-05) |
| :-: |
| <img src="https://github.com/raczu/image-processing/blob/main/lab-05/assets/lenna-gray.jpg" width="33%" /> <img src="https://github.com/raczu/image-processing/blob/main/lab-05/assets/approximated-fourier-lpf-80.jpg" width="33%" /> <img src="https://github.com/raczu/image-processing/blob/main/lab-05/assets/approximated-fourier-hpf-60.jpg" width="33%" /> |

```bash
$ python3.9 orthogonal-basis.py --image ./lenna.png --fourier --hpf --size 80
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Feel free to contact: [@raczuu1](https://twitter.com/raczuu1) - contact@raczu.me

Project link: [https://github.com/raczu/image-processing](https://github.com/raczu/image-processing)
<p align="right">(<a href="#top">back to top</a>)</p>
