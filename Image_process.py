from PIL import Image
import numpy as np
import glob
import os

# Variables
size = 40,40
path = '/Users/Hemanth/Documents/My UB/Deep learning /Project 1/Test'
path_save = '/Users/Hemanth/Documents/My UB/Deep learning /Project 1/Modified Images'


# Reads all the files ending with .png and resizes them
image_array = []
for file in glob.glob(os.path.join(path, '*.png')):
    A = Image.open(file)
    B = A.resize(size, Image.ANTIALIAS)
    C = np.round(np.array(B) / 255)
    image_array.append(C.flatten())
    # Save the modified files
    file_path = os.path.join(path_save, 'N' + os.path.basename(file))
    B.save(file_path)

# Convert and save the image array
Final_array = np.array(image_array)
np.save(path_save, np.array(image_array))

B = np.load('/Users/Hemanth/Documents/My UB/Deep learning /Project 1/Modified images.npy')
print(B.shape)
