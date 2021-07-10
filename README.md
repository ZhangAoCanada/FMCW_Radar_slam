# Let's try some side projects, and to see whether we can figure it out.

## Notice

This repo was originally created for some extracurricular activities, but I recently found more and more people are getting interested in this project. Therefore, after negotiating with the company, I here open source a very small portion of the data (several frames maybe). The data can be found at [Google Drive](https://drive.google.com/drive/folders/1DU2_lH9rVMZXmB7okKKQG6G8NOtTQJA7?usp=sharing). The data should be organized in the following way.
```yaml
RAD_numpy: Range-Azimuth-Doppler Data
radar_RAD_mask: Detection masks for the RAD Data
```
For details of how to use the data, see [illustration.md](illustration/README.md)

## Current state

It basically works, but sometimes it throws some rotation or translation errors. A screenshot is shown as below,

![Image](./results/showcase.png?raw=true)

## TODO

- 1. Find a way to optimize the R | t matrices.

- 2. Find a better way for matching points filtering.

- 3. Consider multiple frames instead of only 2 frames each time (Key Frame?).
