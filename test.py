import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import *
from radar_frame import Frame
from radar_slam import RSLAM

# Radar Configuration
RADAR_CONFIG_FREQ = 77 # GHz
DESIGNED_FREQ = 76.8 # GHz
RANGE_RESOLUTION = 0.1953125 # m
VELOCITY_RESOLUTION = 0.41968030701528203 # m/s
RANGE_SIZE = 256
DOPPLER_SIZE = 64
AZIMUTH_SIZE = 256
ANGULAR_RESOLUTION = np.pi / 2 / AZIMUTH_SIZE # radians
VELOCITY_MIN = - VELOCITY_RESOLUTION * DOPPLER_SIZE/2
VELOCITY_MAX = VELOCITY_RESOLUTION * DOPPLER_SIZE/2

def generateImage(RAD_mag, ax):
    new_mag = getLog(getSumDim(RAD_mag, ax), scalar=10, log_10=True)
    image = norm2Image(new_mag)[..., :3]
    return image

def main(radar_dir, img_dir, mask_dir, sequences, save_dir, \
            window_sizes, max_disparity_channels):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    slam = RSLAM(window_sizes, max_disparity_channels)
    checkoutDir(save_dir)
    for frame_i in tqdm(range(sequences[0], sequences[1])):
        RAD = readRAD(radar_dir, frame_i)
        RAD_mask = readRADMask(mask_dir, frame_i)
        img = readImg(img_dir, frame_i)
        if RAD is None or RAD_mask is None or img is None:
            continue
        RAD_mag = getMagnitude(RAD, power_order=2)
        RA_img = generateImage(RAD_mag, -1)
        RD_img = generateImage(RAD_mag, 1)

        this_frame = Frame(RAD_mag, RAD_mask)
        test_pairs = slam(this_frame)

        ax1.clear()
        ax2.clear()
        ax1.imshow(RA_img)
        ax2.imshow(RD_img)
        if test_pairs is not None:
            print(test_pairs.shape)
            for i in range(len(test_pairs)):
                r1, a1, d1, r2, a2, d2 = test_pairs[i]

                ax1.scatter(a1, r1, s=0.5, c='r')
                ax1.plot(np.array([[a1, r1], [a2, r2]]), 'r')
                ax1.scatter(a2, r2, s=0.5, c='b')

                ax2.scatter(d1, r1, s=0.5, c='r')
                ax2.plot(np.array([[d1, r1], [d2, r2]]), 'r')
                ax2.scatter(d2, r2, s=0.5, c='b')

        fig.canvas.draw()
        plt.pause(0.1)


if __name__ == "__main__":
    time_stamp = "2020-08-09-15-50-38"
    radar_dir = "/DATA/" + time_stamp + "/ral_outputs_" + time_stamp + "/RAD_numpy"
    img_dir = "/DATA/" + time_stamp + "/stereo_imgs/left"
    mask_dir = "/home/ao/Documents/stereo_radar_calibration_annotation/radar_process/radar_ral_process/radar_RAD_mask"

    save_dir = "./results"
    sequences = [0, 1000]

    window_half_sizes = [2, 2, 2]
    max_disparity_channels = [10, 10, 10]
    main(radar_dir, img_dir, mask_dir, sequences, save_dir, \
        window_half_sizes, max_disparity_channels)

