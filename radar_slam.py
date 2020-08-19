import numpy as np
import cv2
from util import *
from radar_frame import Frame
import scipy.linalg as la

def mirrorPad(arr, ax, pad_size):
    assert isinstance(pad_size, int)
    assert isinstance(ax, int)
    assert ax <= len(arr.shape)-1
    if ax == -1:
        ax = len(arr.shape)-1

    pad_front = arr
    for i in range(arr.shape[ax] - pad_size):
        pad_front = np.delete(pad_front, 0, ax)
    pad_end = arr
    for i in range(arr.shape[ax] - pad_size):
        pad_end = np.delete(pad_end, -1, ax)

    output = np.concatenate([pad_front, arr, pad_end], axis=ax)
    return output

def matchFrames(frame1, frame2, pad_sizes, max_search_chs):
    assert isinstance(pad_sizes, list)
    assert isinstance(max_search_chs, list)
    assert len(pad_sizes) == 3
    assert len(max_search_chs) == 3

    frame1_pad = frame1.mag.copy()
    frame2_pad = frame2.mag.copy()
    for i in range(len(frame1.mag.shape)):
        frame1_pad = mirrorPad(frame1_pad, i, pad_sizes[i])
        frame2_pad = mirrorPad(frame2_pad, i, pad_sizes[i])

    ### match from frame1 to frame2, may have overlaps
    all_match_idx_pairs = []
    for i in range(len(frame1.idxes)):
        r_id, a_id, d_id = frame1.idxes[i]
        r_pid, a_pid, d_pid = r_id+pad_sizes[0], a_id+pad_sizes[1], d_id+pad_sizes[2]
        frame1_block = frame1_pad[r_pid-pad_sizes[0]: r_pid+pad_sizes[0]+1, \
                                a_pid-pad_sizes[1]: a_pid+pad_sizes[1]+1, \
                                d_pid-pad_sizes[2]: d_pid+pad_sizes[2]+1]
        match_errs = []
        match_idexes = []
        for j in range(len(frame2.idxes)):
            r2_id, a2_id, d2_id = frame2.idxes[j]
            if np.abs(r2_id - r_id) <= max_search_chs[0] and \
                np.abs(a2_id - a_id) <= max_search_chs[1] and \
                np.abs(d2_id - d_id) <= max_search_chs[2]:
                r2_pid, a2_pid, d2_pid = r2_id+pad_sizes[0], a2_id+pad_sizes[1], \
                                        d2_id+pad_sizes[2]
                frame2_block = frame2_pad[r2_pid-pad_sizes[0]: r2_pid+pad_sizes[0]+1, \
                                        a2_pid-pad_sizes[1]: a2_pid+pad_sizes[1]+1, \
                                        d2_pid-pad_sizes[2]: d2_pid+pad_sizes[2]+1]
                match_errs.append(np.sum(np.abs(frame1_block - frame2_block)))
                match_idexes.append([r2_id, a2_id, d2_id])
        if len(match_errs) == 0:
            continue
        min_ind = np.argmin(match_errs)
        match_id = match_idexes[min_ind]
        id_pair = np.concatenate([np.array([r_id, a_id, d_id]), np.array(match_id)], 0)
        all_match_idx_pairs.append(id_pair)

    if len(all_match_idx_pairs) == 0:
        all_match_idx_pairs = None
    else:
        all_match_idx_pairs = np.array(all_match_idx_pairs)
        ### match back from frame2 to frame1, remove overlaps
        all_frame2_match_idx = all_match_idx_pairs[:, 3:6]
        frame2_match_idx, overlap_indices = np.unique(all_frame2_match_idx, \
                                            return_inverse=True, axis=0)
        unique_match_idx_pair = []
        for i in range(len(frame2_match_idx)):
            r2, a2, d2 = frame2_match_idx[i]
            inds = np.where(overlap_indices==i)[0]
            if len(inds) == 0:
                continue
            elif len(inds) == 1:
                r1, a1, d1 = all_match_idx_pairs[inds[0]][:3]
                unique_match_idx_pair.append([r1,a1,d1,r2,a2,d2])
            else:
                errs = []
                frame1_is = []
                r2_p, a2_p, d2_p = r2+pad_sizes[0], a2+pad_sizes[1], d2+pad_sizes[2]
                frame2_block = frame2_pad[r2_p-pad_sizes[0]: r2_p+pad_sizes[0]+1, \
                                        a2_p-pad_sizes[1]: a2_p+pad_sizes[1]+1, \
                                        d2_p-pad_sizes[2]: d2_p+pad_sizes[2]+1]
                for j in range(len(inds)):
                    ind = inds[j]
                    r1, a1, d1 = all_match_idx_pairs[ind][:3]
                    r1_p, a1_p, d1_p = r1+pad_sizes[0], a1+pad_sizes[1], d1+pad_sizes[2]
                    frame1_block = frame1_pad[r1_p-pad_sizes[0]: r1_p+pad_sizes[0]+1, \
                                            a1_p-pad_sizes[1]: a1_p+pad_sizes[1]+1, \
                                            d1_p-pad_sizes[2]: d1_p+pad_sizes[2]+1]
                    errs.append(np.sum(np.abs(frame1_block - frame2_block)))
                    frame1_is.append([r1, a1, d1])
                min_ind = np.argmin(errs)
                frame1_match = frame1_is[min_ind]
                r1, a1, d1 = frame1_match
                unique_match_idx_pair.append([r1,a1,d1,r2,a2,d2])
        if len(unique_match_idx_pair) == 0:
            unique_match_idx_pair = None
        else:
            unique_match_idx_pair = np.array(unique_match_idx_pair)
        all_match_idx_pairs = unique_match_idx_pair
    return all_match_idx_pairs

def calculateDirection(arr):
    return arr / np.expand_dims(np.sqrt(np.sum(np.square(arr), axis=-1)), axis=-1)

def transferIdpair2Pnts(pairs):
    assert len(pairs.shape) == 2
    assert pairs.shape[1] == 6
    frame1_rad = pairs[:, :3]
    frame2_rad = pairs[:, 3:]
    frame1_x, frame1_z = raId2CartPnt(frame1_rad[:,0], frame1_rad[:,1])
    frame1_xz = np.concatenate([np.expand_dims(frame1_x, -1), \
                            np.expand_dims(frame1_z, -1)], -1)
    frame2_x, frame2_z = raId2CartPnt(frame2_rad[:,0], frame2_rad[:,1])
    frame2_xz = np.concatenate([np.expand_dims(frame2_x, -1), \
                            np.expand_dims(frame2_z, -1)], -1)
    return frame1_xz, frame2_xz

def directionFilter(match_pairs, round_resolution):
    assert len(match_pairs.shape) == 2
    assert match_pairs.shape[1] == 6
    frame1_rad = match_pairs[:, :3]
    frame2_rad = match_pairs[:, 3:]
    ### transfer to Cartesian coordinates
    frame1_rad, frame2_rad = transferIdpair2Pnts(match_pairs)
    ### calculate the directions
    directions = calculateDirection(frame2_rad - frame1_rad)

    direct_round = np.around(directions/round_resolution, 0)
    direct_round, d_indices, d_round_counts = np.unique(direct_round, \
                                return_inverse=True, return_counts=True, axis=0)
    ind = np.argmax(d_round_counts)
    original_ind = np.where(d_indices == ind)[0]

    output_pairs = []
    for i in range(len(original_ind)):
        ind = original_ind[i]
        output_pairs.append(match_pairs[ind])
    if len(output_pairs) == 0:
        output_pairs = None
    else:
        output_pairs = np.array(output_pairs)
    return output_pairs

def optimizeR(R):
    sinTheta = (-R[0, 1] + R[1, 0]) / 2
    cosTheta = (R[0, 0] + R[1, 1]) / 2
    if sinTheta > 1:
        sinTheta = 1.
    if cosTheta > 1:
        cosTheta = 1.
    new_R = np.array([[cosTheta, -sinTheta], [sinTheta, cosTheta]])
    return new_R

class RSLAM:
    def __init__(self, pad_sizes, max_search_chs):
        self.prev_frame = None
        self.current_frame = None
        self.pad_sizes = pad_sizes
        self.max_search_chs = max_search_chs
        self.H_w = np.identity(3)
        self.H_all = [self.H_w]
        self.pcl = None
        self.id_pairs = None
        self.round_resolution = 0.8

    def H2Hw(self, ):
        R_w = self.H_all[-1][:2, :2]
        t_w = self.H_all[-1][:2, 2:3]
        R = self.H[:2, :2]
        t = self.H[:2, 2:3]
        R = optimizeR(R)
        new_t = np.dot(R_w, t) + t_w
        new_R = np.dot(R, R_w)
        self.H_w = np.concatenate([np.concatenate([new_R, new_t], -1), \
                                np.array([[0., 0., 1.]])], 0)
        print(len(self.H_all))
        print(R)
        print(t)
        print(self.H_w)
    
    def getRTMat(self, id_pairs):
        pnts1, pnts2 = transferIdpair2Pnts(id_pairs)
        pnts1, pnts2 = addonesToLastCol(pnts1), addonesToLastCol(pnts2)
        results = la.lstsq(pnts1, pnts2)
        self.H = results[0].T
        self.H = np.linalg.inv(self.H)
        self.H2Hw()
        self.H_all.append(self.H_w)

    def process(self,):
        self.id_pairs = matchFrames(self.prev_frame, self.current_frame, self.pad_sizes, \
                            self.max_search_chs)
        if self.id_pairs is not None:
            num_original = len(self.id_pairs)
            self.id_pairs = directionFilter(self.id_pairs, self.round_resolution)
            if self.id_pairs is not None:
                print("%.d out of %.d are selected for calculation"%(len(self.id_pairs), \
                        num_original))
                self.getRTMat(self.id_pairs)

    def getPcl(self, frame):
        if frame.pnts is not None and len(self.H_all) != 0:
            pnts_col1 = addonesToLastCol(frame.pnts) 
            pnts = np.dot(pnts_col1, self.H_all[-1].T)
            pnts = np.delete(pnts, -1, -1)
            if self.pcl is None:
                self.pcl = pnts
            else:
                self.pcl = np.concatenate([self.pcl, pnts], 0)

    def __call__(self, frame):
        if self.prev_frame == None:
            self.prev_frame = frame
        else:
            self.current_frame = frame
            self.process()
            self.prev_frame = frame
        self.getPcl(frame)

if __name__ == "__main__":
    pad_size = 4
    a = np.arange(12).reshape(3,4)
    print(a)
    a = mirrorPad(a, 0, pad_size)
    a = mirrorPad(a, 1, pad_size)
    print(a)
