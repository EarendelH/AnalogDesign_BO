import numpy as np
from PIL import Image


def change_colors(img):
    # Convert image to numpy array
    data = np.array(img)

    # Define the color white and black in RGBA
    white = [255, 255, 255, 255]
    black = [0, 0, 0, 255]

    # Identify non-transparent and non-white pixels (alpha channel is not 0 and not white)
    non_white_or_transparent = (data[:, :, 3] != 0) & ~np.all(data[:, :, :3] == 255, axis=-1)

    # Change these pixels to black
    data[non_white_or_transparent] = black

    # Identify transparent pixels (alpha channel is 0)
    transparent = (data[:, :, 3] == 0)

    # Change transparent pixels to white
    data[transparent] = white

    # Convert array back to an image
    new_img = Image.fromarray(data)
    return new_img


# Apply the function to the image
image = Image.open('/Users/hanwu/Downloads/Joblib/LDO_SSF.png')
modified_image = change_colors(image)

# Save the modified image to a new file
output_path = '/Users/hanwu/Downloads/Joblib/LDO_SSF_modify.png'
modified_image.save(output_path)
