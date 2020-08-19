# Information you need before working on it

## Background

Frequency Modulated Continuous Wave (FMCW) Radar, as you may have already googled, is a kind of aradar that constantly transmits signals with the frequency linearly changing w.r.t time during one period. We often call this period/cycle a "chirp".

The radar we are using in this repo is AWR1843Boost, developed by Texas Instruments. The sampling rate of the on-board receivers is fixed and there are totally 256 samples captured within 1 chirp. These 256 samples can be used for range estimation. As for doppler, since 1 chirp only takes 6 usec, we will collect 64 chirps each time for generating decent velocity spectrum. Finally, the radar itself has on-board 4 transmitters and 3 receivers. 2 receivers are enabled. Collecting all the information above, the raw data will be in size [256, 8, 64]. 

You can review the raw data as the signal you received from a random wave. Therefore, for the next step, you will probably decomposate fraquency information using Fast-Fourier Transform (FFT). Same as here, a 3D FFT will be conducted on this [256, 8, 64] data (zero-padding involved during FFT on azimuth, a.k.a angle dimension). The output will be in size [256, 256, 64], representing **RANGE-AZIMUTH-DOPPLER**, **RAD** for short. 

## What are the data

As you can find out in `test.py`, there are two directories you need to provide. These are the 2 "10G" data I talked about. 

- The first directory you will be providing is called `RAD_numpy`. This dir includes the **RAD** matrices of the whole sequence. Note that the size of data will be [256, 256, 64, 2] with last 2 channels storing real-part and imagrinary-part of the **RAD** matrices. After geting the magnitude, powering it by 2, scaling it up by 10, then taking the log of this shxxt, you will see a human-readable 3D matrice with size [256, 256, 64] presenting in front of you. Since it is hard to visualize a 3D matrice, normally we will visualize it by two different views. A sample is shown as below,
![Image](./vis.png?raw=true)
Left is called **Range-Doppler** spectrum, **RD** for short. Right is **Range-Azimuth** spectrum, **RA** for short.

- The second dir is called `radar_RAD_mask`, as you can see. It has same number of frames as **RAD** data with size [256, 256, 64]. They are results of me processing all the **RAD** matrices. They are generally just masks, with 1 telling you there is an object point detected and 0 telling you there is nothing but noise.

## Algorithms that I implemented and Things that could be improved

### Points Matching from one frame to another

As you may probably understand, this is the crucial part of SLAM. 
