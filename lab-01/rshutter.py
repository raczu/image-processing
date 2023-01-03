import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from os import mkdir
from os.path import isdir
from shutil import rmtree
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib.image import AxesImage
from typing import List, Tuple
from copy import deepcopy

SENSOR = 256
M = 64


def propeller(blades: int, x: float, m: float) -> float:
    return np.sin(blades * x + ((m * np.pi) / 10))


def get_data(blades: int) -> np.array:
    rads = np.arange(0, 2 * np.pi, 0.01)
    ms = np.linspace(-M / 2, M / 2, M)

    return np.array(
        [[[propeller(blades, x, m) * np.cos(x) * ((SENSOR // 2) - 6) + (SENSOR // 2) for x in rads],
          [propeller(blades, x, m) * np.sin(x) * ((SENSOR // 2) - 6) + (SENSOR // 2) for x in rads]] for m in ms])


def update(frame: int, prop: Line2D, prop_data: np.array) -> Line2D:
    prop.set_data(prop_data[frame][0], prop_data[frame][1])

    return prop


def get_shuttered_prop_data(video: cv2.VideoCapture, lines: int) -> Tuple[List[np.array], List[np.array]]:
    shutter_data = list()
    prop_data = list()
    is_captured, frozen_frame = video.read()
    is_captured, current_frame = video.read()
    frozen_frame, current_frame = cv2.cvtColor(frozen_frame, cv2.COLOR_BGR2RGB), \
                                  cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    read_frames = 1

    while is_captured and read_frames <= SENSOR:
        frozen_frame[0:SENSOR - read_frames - 1, :] = current_frame[0:SENSOR - read_frames - 1, :]
        shutter_data.append(deepcopy(frozen_frame))
        prop_data.append(deepcopy(current_frame))

        is_captured, current_frame = video.read()
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        read_frames += lines

    video.release()

    return prop_data, shutter_data


def animate(frame: int, prop: AxesImage, shuttered_prop: AxesImage, data: Tuple[np.array, List[np.array]],
            lines: Tuple[Line2D, Line2D], nlines: int) -> Tuple[AxesImage, AxesImage, Line2D, Line2D]:
    prop_data, shuttered_prop_data = data
    l1, l2 = lines

    l1.set_ydata(SENSOR - frame * nlines)
    prop.set_data(prop_data[frame])
    l2.set_ydata(SENSOR - frame * nlines)
    shuttered_prop.set_data(shuttered_prop_data[frame])

    return prop, shuttered_prop, l1, l2


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Arguments to configure a script')

    parser.add_argument(
        '-b',
        '--blades',
        type=int,
        required=True,
        help='specify a number of propeller blades (3 or 5)')

    parser.add_argument(
        '-l',
        '--lines',
        type=int,
        required=True,
        help='specify a number of lines that sensor can read at once')

    parser.add_argument(
        '--save',
        action='store_true',
        help='specify if generated animation should be saved')

    args = parser.parse_args()

    if args.blades not in (3, 5):
        raise ValueError('Number of propeller blades should be equal to 3 or 5!')

    prop_data = get_data(args.blades)

    fig = plt.figure(figsize=(SENSOR / 100, SENSOR / 100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_xlim(0, SENSOR)
    ax.set_ylim(0, SENSOR)
    ax.set_axis_off()
    fig.add_axes(ax)

    prop, = plt.plot(prop_data[0][0], prop_data[0][1])

    animation = FuncAnimation(fig, update, frames=M, fargs=(prop, prop_data), interval=M)
    mkdir('./tmp') if not isdir('./tmp') else None
    animation.save('./tmp/propeller.gif', PillowWriter(fps=30))
    propeller_animation = cv2.VideoCapture('./tmp/propeller.gif')

    prop_data, shuttered_prop_data = get_shuttered_prop_data(propeller_animation, args.lines)
    rmtree('./tmp') if isdir('./tmp') else None

    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].set_xlim(0, SENSOR)
    axes[0].set_ylim(0, SENSOR)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title('PROPELLER ANIMATION')
    l1 = axes[0].axhline(y=SENSOR, xmin=0, xmax=SENSOR, lw=1, color='black')
    prop = axes[0].imshow(prop_data[0])

    axes[1].set_xlim(0, SENSOR)
    axes[1].set_ylim(0, SENSOR)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title('ROLLING SHUTTER EFFECT SIMULATION')
    l2 = axes[1].axhline(y=SENSOR, xmin=0, xmax=SENSOR, lw=1, color='black')
    shuttered_prop = axes[1].imshow(shuttered_prop_data[0])

    animation = FuncAnimation(fig, animate, frames=len(shuttered_prop_data),
                              fargs=(prop, shuttered_prop, (prop_data, shuttered_prop_data),
                                     (l1, l2), args.lines), interval=100)

    mkdir('./assets') if args.save and not isdir('./assets') else None
    animation.save('./assets/rolling-shutter-simulation.gif', PillowWriter(fps=30))

    plt.show()


if __name__ == '__main__':
    main()
