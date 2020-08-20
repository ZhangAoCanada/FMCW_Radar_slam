import numpy as np
from util import *

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

class Frame:
    """
    This is a class for conveniencely getting the information we want
    """
    def __init__(self, RAD_mag, RAD_mask):
        self.mask = RAD_mask
        self.mag = RAD_mag
        self.idxes = self.getIdx()
        self.pnts = self.getPcl()

    def getIdx(self):
        """ Get the indexes from Range, Azimuth, Doppler dimensions """
        output_idxes = []
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                for k in range(self.mask.shape[2]):
                    if self.mask[i, j, k] > 0:
                        output_idxes.append([i,j,k])
        if len(output_idxes) == 0:
            output_idxes = None
        else:
            output_idxes = np.array(output_idxes)
        return output_idxes

    def getPcl(self):
        """ Transfer [range, angle] to [x, z] """
        self.RA_mask = np.where(getSumDim(self.mask, -1)>0., 1., 0.)
        output_pcl = []
        for i in range(self.RA_mask.shape[0]):
            for j in range(self.RA_mask.shape[1]):
                if self.RA_mask[i, j] > 0:
                    point_range = ((RANGE_SIZE-1) - i) * RANGE_RESOLUTION
                    point_angle = (j * (2*np.pi/AZIMUTH_SIZE) - np.pi) / \
                                    (2*np.pi*0.5*RADAR_CONFIG_FREQ/DESIGNED_FREQ)
                    point_angle = np.arcsin(point_angle)
                    point_zx = polarToCartesian(point_range, point_angle)
                    output_pcl.append([point_zx[1], point_zx[0]])
        if len(output_pcl) == 0:
            output_pcl = None
        else:
            output_pcl = np.array(output_pcl)
        return output_pcl



