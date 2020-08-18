import os
import cv2
import numpy as np
import math
import random
import colorsys
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from collections import Counter

from glob import glob
from tqdm import tqdm

# detectron2 all classes
CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', \
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', \
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', \
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', \
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', \
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', \
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', \
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', \
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', \
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', \
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', \
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', \
        'hair drier', 'toothbrush']

# road user classes for this research
ROAD_USERS = ['person', 'bicycle', 'car', 'motorcycle',
            'bus', 'train', 'truck', 'boat']

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

def RandomColors(N, bright=True): 
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # detectron uses random shuffle to give the differences
    random.seed(8888)
    random.shuffle(colors)
    return colors

def checkoutDir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        for _ in glob(os.path.join(directory, "*")):
            os.remove(_)

def roadUsersColors():
    colors_out = []
    all_colors = RandomColors(len(CLASS_NAMES))
    for class_i in ROAD_USERS:
        class_ind = CLASS_NAMES.index(class_i)
        colors_out.append(all_colors[class_ind])
    return colors_out

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

def DbscanDenoise(pcl, epsilon=0.3, minimum_samples=100, dominant_op=False):
    clustering = DBSCAN(eps=epsilon, min_samples=minimum_samples).fit(pcl)
    output_labels = clustering.labels_
    if not dominant_op:
        output_pcl = []
        for label_i in np.unique(output_labels):
            if label_i == -1:
                continue
            output_pcl.append(pcl[output_labels == label_i])
    else:
        if len(np.unique(output_labels)) == 1:
            output_pcl = np.zeros([0,3])
        else:
            counts = Counter(output_labels)
            output_pcl = pcl[output_labels == counts.most_common(1)[0][0]]
    return output_pcl

def toCartesianMask(RA_mask):
    output_mask = np.zeros([RA_mask.shape[0], RA_mask.shape[0]*2])
    for i in range(RA_mask.shape[0]):
        for j in range(RA_mask.shape[1]):
            if RA_mask[i, j] > 0:
                point_range = ((RANGE_SIZE-1) - i) * RANGE_RESOLUTION
                point_angle = (j * (2*np.pi/AZIMUTH_SIZE) - np.pi) / \
                                (2*np.pi*0.5*RADAR_CONFIG_FREQ/DESIGNED_FREQ)
                point_angle = np.arcsin(point_angle)
                point_zx = polarToCartesian(point_range, point_angle)
                new_i = int(output_mask.shape[0] - \
                        np.round(point_zx[0]/RANGE_RESOLUTION)-1)
                new_j = int(np.round((point_zx[1]+50)/RANGE_RESOLUTION)-1)
                output_mask[new_i,new_j] = RA_mask[i, j] 
    return output_mask

def toRAMask(RA_cart_mask):
    output_mask = np.zeros([RANGE_SIZE, AZIMUTH_SIZE]) 
    for i in range(RA_cart_mask.shape[0]):
        for j in range(RA_cart_mask.shape[1]):
            if RA_cart_mask[i, j] > 0:
                z = (RA_cart_mask.shape[0] - i + 1)*RANGE_RESOLUTION
                x = (j + 1)*RANGE_RESOLUTION - 50
                range_, angle_ = cartesianToPolar(z, x)
                angle_ = np.sin(angle_)
                new_i = int( (RANGE_SIZE-1) - (range_/RANGE_RESOLUTION) )
                new_j = int( (angle_*(2*np.pi*0.5*RADAR_CONFIG_FREQ/DESIGNED_FREQ) \
                        + np.pi) / (2*np.pi/AZIMUTH_SIZE) )
                if new_i > output_mask.shape[0]-1:
                    new_i = output_mask.shape[0]-1
                if new_j > output_mask.shape[1]-1:
                    new_j = output_mask.shape[1]-1
                output_mask[new_i, new_j] = RA_cart_mask[i, j]
    return output_mask
##################### coordinate transformation ######################
def cartesianToPolar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def polarToCartesian(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def applyMask(image, mask, color, alpha=1):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                 image[:, :, c] *
                                (1 - alpha) + alpha * color[c] * 255,
                                 image[:, :, c])
    return image

def main(radar_dir, img_dir, mask_dir, sequences, save_dir):
    ##### NOTE: mouse control class: GetMousePnts #####
    checkoutDir(save_dir)
    for frame_i in tqdm(range(sequences[0], sequences[1])):
        RAD = readRAD(radar_dir, frame_i)
        RAD_mask = readRADMask(mask_dir, frame_i)
        img = readImg(img_dir, frame_i)
        if RAD is not None and RAD_mask is not None and img is not None:
            RAD_mag = getMagnitude(RAD, power_order=2)
            RA_mag = getLog(getSumDim(RAD_mag, -1), scalar=10, log_10=True)
            RD_img = norm2Image(getLog(getSumDim(RAD_mag, 1), \
                                scalar=10, log_10=True))[..., :3]
            RA_mask = np.where(getSumDim(RAD_mask, -1) > 0., 1., 0.)
            RA_masked_cart_mag = toCartesianMask(RA_mag) * toCartesianMask(RA_mask)
            RA_cart_img = norm2Image(RA_masked_cart_mag)[..., :3]

            scale = RA_mag.shape[0] / img.shape[0]
            display_img = np.concatenate([\
                                    cv2.resize(img, (int(scale*img.shape[1]), \
                                    RA_mag.shape[0])), \
                                    RA_cart_img[..., ::-1]], 1)

            # display_img = RA_cart_img[..., ::-1]

            cv2.imwrite(os.path.join(save_dir, "%.d.png"%(frame_i)), display_img)
            # cv2.imshow("img", display_img)
            # cv2.waitKey(10)


if __name__ == "__main__":
    time_stamp = "2020-08-09-15-50-38"
    radar_dir = "/DATA/" + time_stamp + "/ral_outputs_" + time_stamp + "/RAD_numpy"
    img_dir = "/DATA/" + time_stamp + "/stereo_imgs/left"
    mask_dir = "../radar_RAD_mask"

    save_dir = "./checkout"
    sequences = [0, 1000]
    main(radar_dir, img_dir, mask_dir, sequences, save_dir)





