import math

import numpy as np
from PIL import Image
from skimage import color, io

def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None
    out = io.imread(image_path)
    io.imread(image_path)
    ### YOUR CODE HERE
    # Use skimage io.imread
    # pass
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out

def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None
    out = image[start_row:start_row+num_rows, start_col:start_col+num_cols, :]
    ### YOUR CODE HERE
    # pass
    ### END YOUR CODE

    return out

def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    out = 0.5 * image**2
    ### YOUR CODE HERE
    # pass
    ### END YOUR CODE

    return out

def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    ### YOUR CODE HERE
    # pass
    row_scale_factor = input_rows / output_rows
    col_scale_factor = input_cols / output_cols

    for i in range(output_rows):
        for j in range(output_cols):
            input_i = int(i * row_scale_factor)
            input_j = int(j * col_scale_factor)
            output_image[i, j, :] = input_image[input_i, input_j, :]
    ### END YOUR CODE

    # 3. Return the output image
    return output_image

def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!

    ## YOUR CODE HERE
    # pass
    x, y = point
    x_prime = x * np.cos(theta) - y * np.sin(theta)
    y_prime = x * np.sin(theta) + y * np.cos(theta)
    return np.array([x_prime, y_prime])
    ### END YOUR CODE

def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)
    center_row, center_col = input_rows // 2, input_cols // 2
    ## YOUR CODE HERE
    # pass
    for row in range(input_rows):
        for col in range(input_cols):
            # 4. Calculate the corresponding input coordinates for each output pixel
            input_row = int((row - center_row) * np.cos(theta) - (col - center_col) * np.sin(theta) + center_row)
            input_col = int((row - center_row) * np.sin(theta) + (col - center_col) * np.cos(theta) + center_col)

            # 5. Fill in the output image if the input coordinates are valid
            if (0 <= input_row < input_rows) and (0 <= input_col < input_cols):
                output_image[row][col] = input_image[input_row][input_col]
    ### END YOUR CODE

    # 3. Return the output image
    return output_image

