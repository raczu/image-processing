import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
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

    return prop_data, shutter_data


def animate(frame: int, prop: AxesImage, shuttered_prop: AxesImage,
            data: Tuple[np.array, List[np.array]]) -> Tuple[AxesImage, AxesImage]:
    prop_data, shuttered_prop_data = data

    prop.set_data(prop_data[frame])
    shuttered_prop.set_data(shuttered_prop_data[frame])

    return prop, shuttered_prop


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

    args = parser.parse_args()

    if args.blades not in (3, 5):
        raise ValueError('Number of propeller blades should be equal to 3 or 5!')

    prop_data = get_data(args.blades)

    fig = plt.figure(figsize=(SENSOR / 100, SENSOR / 100), dpi=50)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_xlim(0, SENSOR)
    ax.set_ylim(0, SENSOR)
    ax.set_axis_off()
    fig.add_axes(ax)

    prop, = plt.plot(prop_data[0][0], prop_data[0][1])

    animation = FuncAnimation(fig, update, frames=M, fargs=(prop, prop_data), interval=M)
    animation.save('./propeller.gif', PillowWriter(fps=30))
    propeller_animation = cv2.VideoCapture('./propeller.gif')

    prop_data, shuttered_prop_data = get_shuttered_prop_data(propeller_animation, args.lines)

    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_xlim(0, SENSOR)
    axs[0].set_ylim(0, SENSOR)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    prop = axs[0].imshow(prop_data[0])

    axs[1].set_xlim(0, SENSOR)
    axs[1].set_ylim(0, SENSOR)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    shuttered_prop = axs[1].imshow(shuttered_prop_data[0])

    _ = FuncAnimation(fig, animate, frames=len(shuttered_prop_data),
                      fargs=(prop, shuttered_prop, (prop_data, shuttered_prop_data)), interval=100)

    plt.show()


if __name__ == '__main__':
    main()
