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
    if not os.path.exists(directory):
        os.mkdir(directory)
    elif len(glob(os.path.join(directory, "*"))) != 0:
        for _ in glob(os.path.join(directory, "*")):
            os.remove(_)

def readRAD(radar_dir, frame_id):
    if os.path.exists(os.path.join(radar_dir, "%.6d.npy"%(frame_id))):
        return np.load(os.path.join(radar_dir, "%.6d.npy"%(frame_id)))
    else:
        return None
    
def readRADMask(mask_dir, frame_i):
    filename = os.path.join(mask_dir, "RAD_mask_%.d.npy"%(frame_i))
    if os.path.exists(filename):
        RAD_mask = np.load(filename)
    else:
        RAD_mask = None
    return RAD_mask

def readImg(img_dir, frame_i):
    filename = os.path.join(img_dir, "%.6d.jpg"%(frame_i))
    if os.path.exists(filename):
        img = cv2.imread(filename)
    else:
        img = None
    return img

def getMagnitude(target_array, power_order=2):
    target_array = np.abs(target_array)
    target_array = pow(target_array, power_order)
    return target_array 

def getLog(target_array, scalar=1., log_10=True):
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return target_array

def getSumDim(target_array, target_axis):
    output = np.sum(target_array, axis=target_axis)
    return output 

def norm2Image(array):
    norm_sig = plt.Normalize()
    img = plt.cm.viridis(norm_sig(array))
    img *= 255.
    img = img.astype(np.uint8)
    return img

def cartesianToPolar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def polarToCartesian(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def raId2CartPnt(r, a):
    point_range = ((RANGE_SIZE-1) - r) * RANGE_RESOLUTION
    point_angle = (a * (2*np.pi/AZIMUTH_SIZE) - np.pi) / \
                    (2*np.pi*0.5*RADAR_CONFIG_FREQ/DESIGNED_FREQ)
    point_angle = np.arcsin(point_angle)
    point_zx = polarToCartesian(point_range, point_angle)
    return point_zx[1], point_zx[0]

def addonesToLastCol(target_array):
    adding_ones = np.ones([target_array.shape[0], 1])
    output_array = np.concatenate([target_array, adding_ones], axis=-1)
    return output_array


