import Data_loader
import random
import cv2
import numpy as np
import os
import datetime
import math



def save_image(image, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique filename using a timestamp
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f'image_{current_time}.jpg'
    output_path = os.path.join(output_dir, filename)

    # Save the image
    cv2.imwrite(output_path, image)

def generate_non_uniform_random(mean, sigma):
    # Generate 2 random number from a uniform distribution [0, 1)
    u= random.random()
    v= random.random()
    # Transform the uniform random number to a non-uniform distribution (Apply the Box-Muller transform i got this idea from https://en.wikipedia.org/wiki/Normal_distribution )
    x = math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v) # x has standard gaussian distribution 
    #y = math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)

    non_uniform_random = mean + sigma * x
    return non_uniform_random


def noise_maker(RGB_matrix):
    mean = 0 # britness
    std_dev = 10
    
    rows = len(RGB_matrix)
    columns = len(RGB_matrix[0])
    
    # Create an empty matrix to store the result
    result = [[0] * columns for _ in range(rows)]

    # Add random Gaussian noise to RGB matrixes
    for i in range(rows):
        for j in range(columns):
            random_number = generate_non_uniform_random(mean,std_dev) # generate_non_uniform_random is my gaussian function !
            result[i][j] = RGB_matrix[i][j] + random_number

            if result[i][j] > 255:
                result[i][j] = 255
            elif result[i][j] < 0:
                result[i][j] = 0
    
    return result

"""
from numpy.polynomial.polynomial import Polynomial
from scipy.linalg import null_space

def calculate_eigenvalues(matrix):
    # Compute the characteristic polynomial
    char_poly = Polynomial(matrix.flatten(), domain=[-1, 1])

    # Solve the characteristic polynomial equation
    eigenvalues = char_poly.roots()

    # Extract the real parts of the eigenvalues
    eigenvalues = [complex(eigenvalue).real for eigenvalue in eigenvalues]

    # Remove duplicates
    eigenvalues = list(set(eigenvalues))

    # Compute the matrix of eigenvectors
    n = matrix.shape[0]
    U = np.zeros((n, n), dtype=np.complex128)
    for i in range(len(eigenvalues)):
        A = matrix - eigenvalues[i] * np.eye(n)
        null_space = np.linalg.null_space(A)
        U[:, i] = null_space.flatten()

    return eigenvalues, U
"""

def calculate_U_s_Vt(matrix):
    # Compute the SVD of the matrix
    A = matrix @ matrix.T
    eigenvalues, U = np.linalg.eig(A)
    eigenvalues_sorted_index = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    eigenvalues_sorted = eigenvalues[eigenvalues_sorted_index]
    U = U[:, eigenvalues_sorted_index]

    # Compute the singular values and V^T
    singular_values = np.sqrt(eigenvalues_sorted)
    Vt = (matrix.T @ U) / singular_values

    return U, singular_values, Vt.T

def svd_denoising(image, threshold):
    # Convert the image to float32 for SVD computation
    image_float = image.astype(np.float32)

    # Perform Singular Value Decomposition
    U, s, Vt = calculate_U_s_Vt(image_float)
    # Apply denoising by thresholding the singular values
    s_denoised = np.where(s > threshold, s, 0)

    # Reconstruct the denoised image (multiplie 3 matrixes)
    image_denoised = U @ np.diag(s_denoised) @ Vt

    # Convert the denoised image back to uint8
    image_denoised = np.clip(image_denoised, 0, 255).astype(np.uint8)

    return image_denoised

def main():

    input_file_dir = 'C:/Users/alini/OneDrive/Desktop/SVD_Project_Denoising/Data_set/'

    # Get a list of filenames in the directory
    filenames = os.listdir(input_file_dir)

    # Process each image
    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            input_file = os.path.join(input_file_dir, filename)
            R_img, G_img, B_img = Data_loader.get_data(input_file)
            # This part is for making Noise on images :
            noised_R_img=noise_maker(R_img)
            noised_G_img=noise_maker(G_img)
            noised_B_img=noise_maker(B_img)
            
            # Convert the lists to NumPy arrays
            noised_R_img = np.array(noised_R_img, dtype=np.uint8)
            noised_G_img = np.array(noised_G_img, dtype=np.uint8)
            noised_B_img = np.array(noised_B_img, dtype=np.uint8)

            # Combine the channels back into a single image to show noised image
            noised_combined_img = cv2.merge([noised_R_img, noised_G_img,noised_B_img ])
            
            # This part is for DeNoise the images using SVD algorithm :
            threshold = 310 # value of threshold is based on try and error checks of inputs and outputs of SVD
            denoised_R_img = svd_denoising(noised_R_img,threshold)
            denoised_G_img = svd_denoising(noised_G_img,threshold)
            denoised_B_img = svd_denoising(noised_B_img,threshold)
            
            # Combine the channels back into a single image to show denoised image
            Denoised_combined_img = cv2.merge([denoised_R_img, denoised_G_img,denoised_B_img ])
           
            # Show the resulting image
            Data_loader.show_noised_image(noised_combined_img)
            Data_loader.show_Denoised_image(Denoised_combined_img)
            
            save_image(noised_combined_img, 'C:/Users/alini/OneDrive/Desktop/SVD_Project_Denoising/Data_set/Noised/')
            save_image(Denoised_combined_img, 'C:/Users/alini/OneDrive/Desktop/SVD_Project_Denoising/Data_set/DeNoised/')
        


if __name__ == '__main__':
    main()


