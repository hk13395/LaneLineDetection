import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from helper_function import*
from IPython.display import HTML
from moviepy.editor import VideoFileClip

# implementing on test images before videos
test_path = os.listdir("test images/")

test_output = "test_output_images"
if not os.path.exists(test_output):
    os.makedirs(test_output)

for test_path in test_path:
    test_img = mpimg.imread(os.path.join('test images', test_path))
    get_momentum()
    final_output = process_image(test_img)
    # plt.imshow(final_output)
    # plt.show()


# implementing on video file
white_output = 'test_video_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

HTML("""<video width="960" height="540" control>
<source src="{0}">
</video>""".format(white_output))

yellow_output = 'test_video_output/solidYellowLeft.mp4'
clip2 = VideoFileClip("test videos/solidYellowLeft.mp4")
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

HTML("""<video width="960" height="540" control>
<source src="{0}">
</video>""".format(yellow_output))
