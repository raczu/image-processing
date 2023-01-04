<div id="top"></div>

# Image processing

![GitHub last commit](https://img.shields.io/github/last-commit/raczu/image-processing)
![python](https://img.shields.io/badge/python-3.9-blue.svg)

Labolatory assignments related to image processing using python3.9+ (due to the use of typing methods supported from version 3.9). The tasks contain solutions to a variety of challanges, including the use of self-written interpolation algorithms, conversion of an image into a CFA (bayer, X-TRANS) and image scaling or rotation.

## Table of contents

* [Solutions](#solutions)
* [Example scripts executions](#example-scripts-executions)
* [Outputs](#outputs)
* [License](#license)
* [Contact](#contact)

## Solutions
* [lab-01 (rolling shutter effect simulation)](https://github.com/raczu/image-processing/tree/main/lab-01)
* [lab-02 (demosaicing)](https://github.com/raczu/image-processing/tree/main/lab-02)
* [lab-03 (scaling and rotating raster images)](https://github.com/raczu/image-processing/tree/main/lab-03)

## Example scripts executions
[lab-01 (rolling shutter effect simulation)](https://github.com/raczu/image-processing/tree/main/lab-01)
```bash
$ python3.9 rshutter.py -b 5 -l 3 --save
```

**FULL USAGE**:
```bash
usage: rshutter.py [-h] -b BLADES -l LINES [--save]

Arguments to configure a script

options:
  -h, --help            show this help message and exit
  -b BLADES, --blades BLADES
                        specify a number of propeller blades (3 or 5)
  -l LINES, --lines LINES
                        specify a number of lines that sensor can read at once
  --save                specify if generated animation should be saved
```

[lab-02 (demosaicing)](https://github.com/raczu/image-processing/tree/main/lab-02)
```bash
$ python3.9 demosaicing.py --image ./image.bmp --bayer --nearest --save
```

**FULL USAGE**:
```bash
usage: demosaicing.py [-h] --image IMAGE [--save] (--nearest | --linear) (--bayer | --xtrans)

Arguments to configure a script

options:
  -h, --help     show this help message and exit
  --image IMAGE  specify a path to the image
  --save         specify if generated images should be saved

filters:
  --bayer        specify if BAYER filter should be applied to image
  --xtrans       specify if X-TRANS filter should be applied to image

interpolation:
  --nearest      specify if NEAREST interpolation should be applied to image
  --bilinear     specify if BILINEAR interpolation should be applied to image

```

[lab-03 (scaling and rotating raster images)](https://github.com/raczu/image-processing/tree/main/lab-03)
```bash
$ python3.9 scale-and-rotate.py --image ./image.jpg --shrink 0.65 --rotate 65 --nearest --save
```

**FULL USAGE**:
```bash
usage: scale-and-rotate.py [-h] --image IMAGE [--save] --shrink SHRINK (--nearest | --bilinear | --keys) --rotate ROTATE

Arguments to configure a script

options:
  -h, --help       show this help message and exit
  --image IMAGE    specify a path to the image
  --save           specify if generated images should be saved
  --shrink SHRINK  specify how much image size should be shrank (0 <= factor <= 1)
  --rotate ROTATE  specify the angle of rotation of the image

interpolation:
  --nearest        specify if NEAREST interpolation should be applied to image
  --bilinear       specify if BILINEAR interpolation should be applied to image
  --keys           specify if KEYS interpolation should be applied to image
```

## Outputs
[lab-01 (rolling shutter effect simulation)](https://github.com/raczu/image-processing/tree/main/lab-01)

![rolling-shutter-effect-simulation](https://github.com/raczu/image-processing/blob/main/lab-01/assets/rolling-shutter-simulation.gif)

[lab-02 (demosaicing)](https://github.com/raczu/image-processing/tree/main/lab-02)

![demosaicing](https://github.com/raczu/image-processing/blob/main/lab-02/assets/summary.png)

[lab-03 (scaling and rotating raster images)](https://github.com/raczu/image-processing/tree/main/lab-03)

![scaling-and-rotating-raster-images](https://github.com/raczu/image-processing/blob/main/lab-03/assets/summary.png)


## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Feel free to contact: [@raczuu1](https://twitter.com/raczuu1) - contact@raczu.me

Project link: [https://github.com/raczu/image-processing](https://github.com/raczu/image-processing)
<p align="right">(<a href="#top">back to top</a>)</p>
