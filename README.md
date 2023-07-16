# Google Scraped Image Dataset Denoising using my SVD Algo

## Introduction

In this project, I utilized the Google Scraped Image Dataset containing architecture images. I selected ten images from the dataset and loaded them into the program. Then, I applied random Gaussian noise to these images. Finally, I denoised the images using my impelimented Singular Value Decomposition (SVD) algorithm.

For each image, three states are displayed:
1. The original image.
2. The image with added noise.
3. The denoised image.

The process is repeated for all selected images, and the noisy images are saved in one folder, while the denoised images are stored in another.

## Program Structure

The program is divided into three main parts:

### Part 1: Image Loading and Preprocessing
- Load each image and prepare them for applying noise.
- Display the original images.

### Part 2: Applying Gaussian Noise
- Apply random Gaussian noise to each image using the implemented functions.
- Display the images with added noise.

### Part 3: Denoising using SVD
- Implement functions to denoise each image using the SVD algorithm.
- Display the denoised images.

## Implementation Details

### Loading and Preprocessing
The dataset, Google Scraped Image Dataset, is accessed, and ten images are selected for the experiment. These images are loaded into the program for further processing.

### Applying Gaussian Noise
Random Gaussian noise is added to each image using custom functions. The noise-added images are displayed to visualize the effect of the noise.

### Denoising using SVD
The Singular Value Decomposition (SVD) algorithm is applied to each noisy image to denoise them. The denoised images are displayed to observe the quality of the denoising process.

## Conclusion

The use of the SVD algorithm for denoising the Google Scraped Image Dataset provides an effective way to remove Gaussian noise from architecture images. The program showcases the three states of each image: original, noisy, and denoised. By visually comparing these states, the effectiveness of the denoising process can be evaluated.

Please note that the results of denoising may vary based on the noise level and the specific images used in the experiment. The project aims to demonstrate the application of the SVD algorithm for image denoising and its potential in improving image quality.
