import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from glob import glob

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

def checkoutDir(directory):
    """ if dir not exists, build one; if exists, remove all files """
    if not os.path.exists(directory):
        os.mkdir(directory)
    elif len(glob(os.path.join(directory, "*"))) != 0:
        for _ in glob(os.path.join(directory, "*")):
            os.remove(_)

def readRAD(radar_dir, frame_id):
    """ read RAD from dir """
    if os.path.exists(os.path.join(radar_dir, "%.6d.npy"%(frame_id))):
        return np.load(os.path.join(radar_dir, "%.6d.npy"%(frame_id)))
    else:
        return None
    
def readRADMask(mask_dir, frame_i):
    """ read RAD detection from dir """
    filename = os.path.join(mask_dir, "RAD_mask_%.d.npy"%(frame_i))
    if os.path.exists(filename):
        RAD_mask = np.load(filename)
    else:
        RAD_mask = None
    return RAD_mask

def readImg(img_dir, frame_i):
    """ read image from dir """
    filename = os.path.join(img_dir, "%.6d.jpg"%(frame_i))
    if os.path.exists(filename):
        img = cv2.imread(filename)
    else:
        img = None
    return img

def getMagnitude(target_array, power_order=2):
    """ get magnitude of complex numbers and power 2 """ 
    target_array = np.abs(target_array)
    target_array = pow(target_array, power_order)
    return target_array 

def getLog(target_array, scalar=1., log_10=True):
    """ get Log with scale """
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return target_array

def getSumDim(target_array, target_axis):
    """ sum up one dimension """
    output = np.sum(target_array, axis=target_axis)
    return output 

def norm2Image(array):
    """ change a float32 array to uint8 opencv format image """
    norm_sig = plt.Normalize()
    img = plt.cm.viridis(norm_sig(array))
    img *= 255.
    img = img.astype(np.uint8)
    return img

def cartesianToPolar(x, y):
    """ Cartesian to Polar """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def polarToCartesian(rho, phi):
    """ Polar to Cartesian """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def raId2CartPnt(r, a):
    """ transfer range, angle to x, z """
    point_range = ((RANGE_SIZE-1) - r) * RANGE_RESOLUTION
    point_angle = (a * (2*np.pi/AZIMUTH_SIZE) - np.pi) / \
                    (2*np.pi*0.5*RADAR_CONFIG_FREQ/DESIGNED_FREQ)
    point_angle = np.arcsin(point_angle)
    point_zx = polarToCartesian(point_range, point_angle)
    return point_zx[1], point_zx[0]

def addonesToLastCol(target_array):
    """ add ones to last column """
    adding_ones = np.ones([target_array.shape[0], 1])
    output_array = np.concatenate([target_array, adding_ones], axis=-1)
    return output_array

def imgPlot(img, ax, cmap, alpha, title=None):
    """ image plotting (customized when plotting RAD) """
    ax.imshow(img, cmap=cmap, alpha=alpha)
    if title == "RD":
        ax.set_xticks([0, 16, 32, 48, 63])
        ax.set_xticklabels([-13, -6.5, 0, 6.5, 13])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("velocity (m/s)")
        ax.set_ylabel("range (m)")
    elif title == "RA":
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_xticklabels([-90, -45, 0, 45, 90])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("angle (degrees)")
        ax.set_ylabel("range (m)")
    elif title == "RA mask in cartesian":
        ax.set_xticks([0, 128, 256, 384, 512])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
    else:
        ax.axis('off')
    if title is not None:
        ax.set_title(title)


