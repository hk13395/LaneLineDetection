# importing the libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def process_image(image):

    # convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # dimensions of the image
    rows, cols = gray_image.shape

    # Add gaussian blur to the image
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)

    # canny edge detection
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # creating the region of interest and vertices
    left_bottom = (0, rows)
    right_bottom = (cols, rows)
    left_top = (450, 325)
    right_top = (550, 325)

    vertices = np.array(
        [[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)

    # defining the mask region
    mask = np.zeros_like(edges)

    if len(edges.shape) > 2:
        channel_count = edges.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(edges, mask)

    # hough transform implementation
    rho = 2
    theta = np.pi/180
    threshold = 12
    min_line_len = 25
    max_line_gap = 20
    friction = 0.9

    lines = cv2.HoughLinesP(masked_image, rho, theta,
                            threshold, np.array([]), min_line_len, max_line_gap)
    line_img = np.zeros(
        (masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    final = cv2.addWeighted(image, 0.8, line_img, 1, 0)

    return final
