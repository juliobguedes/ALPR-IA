# Image Processing Notes

## 1. What is a Kernel?

A kernel is a matrix-like structure that defines the number of pixels considered during an operation. It can be used with weights or as a mask, but also to delimitate the region of considered pixels. Usually the kernel is an odd-sided square area, for its use to replace the middle pixel.

## 2. What is Average Blurring?

Based on a kernel delimitation and an existing image, it creates a new image based on the existing one replacing the pixel in the middle of the kernel by the average of pixels in the kernel. The blur can be enhanced by increasing the side of the kernel: a (5,5) kernel blurs stronger than a (3,3) kernel.

## 3. What is Gaussian Blurring?

Just like the Average Blurring, the Gaussian Blur takes the average of the surrounding pixels, with one main difference: it is a weighted average based on the gaussian curve, and the weight of each pixel in the kernel is based on its distance to the central pixel. The user can define the kernel and the standard deviation for the gaussian curve. It is usually less blurred than the average blur.

## 4. What is Median Blurring?