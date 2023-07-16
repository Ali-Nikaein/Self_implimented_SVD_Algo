import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def check_image_size(image_path):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print("Failed to read the image.")
        return

    # Get the current shape of the image
    current_shape = image.shape

    # Check if the shape is (128, 128, 3)
    if current_shape == (128, 128, 3) and image.dtype == np.uint8:
        print("Image shape and dtype are already (128, 128, 3) uint8.")
        return image
    
    # Resize the image if the shape is different
    resized_image = cv2.resize(image, (128, 128))

    # Cast the image to uint8 if necessary
    if resized_image.dtype != np.uint8:
        resized_image = resized_image.astype(np.uint8)

    # Ensure the image has 3 channels
    if len(resized_image.shape) == 2:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

    print("Resized image shape and dtype: ", resized_image.shape, resized_image.dtype)
    return resized_image

def get_data(input_file_directory):
    
    img = check_image_size(input_file_directory)
    #img = cv2.imread(input_file_directory)

    R_img =  img[:, :, 0]
    #img_list.append(R_img)
    G_img =  img[:, :, 1]
    #img_list.append(G_img)
    B_img =  img[:, :, 2]
    #img_list.append(B_img)
    show_Orginal_image(img)
    return R_img,G_img,B_img
    
def show_Orginal_image(img):
    cv2.namedWindow('orginal Image', cv2.WINDOW_NORMAL)  # Create a named window with adjustable size
    cv2.imshow('orginal Image', img)
    cv2.waitKey(1000)
    # Resize the window to a larger size
    cv2.resizeWindow('orginal Image', 800, 600)  # Set the width and height in pixels

    cv2.destroyAllWindows()

def show_noised_image(img):
    cv2.namedWindow('noised Image', cv2.WINDOW_NORMAL)
    cv2.imshow('noised Image', img)
    cv2.resizeWindow('noised Image', 800, 600)  # Set the width and height in pixels
    cv2.waitKey(1000)  # Wait for 5000 milliseconds (5 seconds)
    cv2.destroyAllWindows()

def show_Denoised_image(img):
    cv2.namedWindow('Denoised Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Denoised Image', img)
    cv2.resizeWindow('Denoised Image', 800, 600)  # Set the width and height in pixels
    cv2.waitKey(1000)  # Wait for 5000 milliseconds (5 seconds)
    cv2.destroyAllWindows()


