import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from helper_function import*

test_path = os.listdir("test images/")

test_output = "test_output_images"
if not os.path.exists(test_output):
    os.makedirs(test_output)

for test_path in test_path:
    test_img = mpimg.imread(os.path.join('test images', test_path))
    get_momentum()
    final_output = process_image(test_img)
    plt.imshow(final_output)
    plt.show()
