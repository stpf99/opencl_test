❯ python opencl_test.py 

Array size: 25,000,000 elements

Results match: True

GPU time: 0.004102 seconds

CPU time: 0.060731 seconds

Speedup: 14.80x



❯ python3.13 manual-kernel-opencl.py

PyOpenCL version: (2024, 2, 6)

Using platform: rusticl

Using device: AMD Radeon R7 Graphics (radeonsi, carrizo, ACO, DRM 3.61, 6.13.6-2-cachyos)

CPU time: 0.320004 seconds

Program built successfully

Kernel created using _cl module

Array size: 100,000,000 elements

Results match: True

GPU time: 0.091999 seconds

CPU time: 0.320004 seconds

Speedup: 3.48x
