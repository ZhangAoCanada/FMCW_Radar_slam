import numpy as np
import cv2
from util import *
from radar_frame import Frame

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
    return all_match_idx_pairs

class RSLAM:
    def __init__(self, pad_sizes, max_search_chs):
        self.prev_frame = None
        self.current_frame = None
        self.pad_sizes = pad_sizes
        self.max_search_chs = max_search_chs

    def process(self,):
        id_pairs = matchFrames(self.prev_frame, self.current_frame, self.pad_sizes, \
                            self.max_search_chs)
        print(id_pairs.shape)
        return id_pairs

    def __call__(self, frame):
        if self.prev_frame == None:
            self.prev_frame = frame
        else:
            self.current_frame = frame
            test = self.process()
            self.prev_frame = frame
            return test

if __name__ == "__main__":
    pad_size = 4
    a = np.arange(12).reshape(3,4)
    print(a)
    a = mirrorPad(a, 0, pad_size)
    a = mirrorPad(a, 1, pad_size)
    print(a)
