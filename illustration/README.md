# Information you need before working on it

## Background

Frequency Modulated Continuous Wave (FMCW) Radar, as you may have already googled, is a kind of aradar that constantly transmits signals with the frequency linearly changing w.r.t time during one period. We often call this period/cycle a "chirp".

The radar we are using in this repo is AWR1843Boost, developed by Texas Instruments. The sampling rate of the on-board receivers is fixed and there are totally 256 samples captured within 1 chirp. These 256 samples can be used for range estimation. As for doppler, since 1 chirp only takes 6 usec, we will collect 64 chirps each time for generating decent velocity spectrum. Finally, the radar itself has on-board 4 transmitters and 3 receivers. 2 receivers are enabled. Collecting all the information above, the raw data will be in size [256, 8, 64]. 

You can review the raw data as the signal you received from a random wave. Therefore, for the next step, you will probably decomposate fraquency information using Fast-Fourier Transform (FFT). Same as here, a 3D FFT will be conducted on this [256, 8, 64] data (zero-padding involved during FFT on azimuth, a.k.a angle dimension). The output will be in size [256, 256, 64], representing **RANGE-AZIMUTH-DOPPLER**, **RAD** for short. 

## What are the data

As you can find out in `test.py`
