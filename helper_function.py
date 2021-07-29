# importing the libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def reject_outlier(data, filter_data, m=1, epsilon=1e-4):
    # removing inf and nan
    ffilter = ~(np.logical_or(np.isinf(data), np.isnan(data)))
    data = data[ffilter]
    filter_data = filter_data[ffilter]

    # remove more standard deviation
    ffilter = (abs(data - np.mean(data)) < m*np.std(data) + epsilon)
    return filter_data[ffilter]


def get_slope(slopes, prev_slope, friction=0.9):
    slope = 0.0
    if len(slopes) > 0:
        slope = np.mean(np.asarray(slopes))
        if not math.isnan(slope) and prev_slope != 0.0:
            slope = prev_slope*friction + slope*(1.0-friction)
        elif prev_slope != 0.0:
            slope = prev_slope
        return slope
    return prev_slope


def get_intercept(intercepts, prev_intercept, friction=0.9):
    intercept = 0.0
    if len(intercepts) > 0.0:
        intercept = np.mean(np.asarray(intercepts))
        if not math.isnan(intercept) and prev_intercept != 0:
            intercept = prev_intercept*friction + intercept*(1.0-friction)
        elif prev_intercept != 0:
            intercept = prev_intercept
        return intercept
    return prev_intercept


def get_x(y, m, b):
    return int(round((y-b)/m))


def get_momentum():
    global prev_slope_left, prev_slope_right, prev_intercept_left, prev_intercept_right
    prev_slope_left = 0.0
    prev_slope_right = 0.0
    prev_intercept_left = 0.0
    prev_intercept_right = 0.0


def get_lanes(lines, bottom=540, top=325, friction=0.9):

    global weights, prev_slope_left, prev_slope_right, prev_intercept_left, prev_intercept_right

    result = []
    left_lane = []
    right_lane = []
    left_slopes = []
    right_slopes = []
    left_intercepts = []
    right_intercepts = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope < 0.0:
                left_lane.append(line)
                left_slopes.append(slope)
                left_intercepts.append(y1-slope*x1)
            else:
                right_lane.append(line)
                right_slopes.append(slope)
                right_intercepts.append(y1-slope*x1)
    left_slopes = np.asarray(left_slopes)
    right_slopes = np.asarray(right_slopes)
    left_intercepts = np.asarray(left_intercepts)
    right_intercepts = np.asarray(right_intercepts)

    left_lane = reject_outlier(left_slopes, np.asarray(left_lane))
    right_lane = reject_outlier(right_slopes, np.asarray(right_lane))

    left_slopes = reject_outlier(left_slopes, left_slopes)
    right_slopes = reject_outlier(right_slopes, right_slopes)

    left_intercepts = reject_outlier(left_intercepts, left_intercepts)
    right_intercepts = reject_outlier(right_intercepts, right_intercepts)

    left_slope = get_slope(left_slopes, prev_slope_left, friction)
    right_slope = get_slope(right_slopes, prev_slope_right, friction)
    left_intercept = get_intercept(
        left_intercepts, prev_intercept_left, friction)
    right_intercept = get_intercept(
        right_intercepts, prev_intercept_right, friction)

    # calculate the endpoints
    ry1 = bottom
    rx1 = get_x(ry1, right_slope, right_intercept)
    ry2 = top
    rx2 = get_x(ry2, right_slope, right_intercept)

    ly1 = bottom
    lx1 = get_x(ly1, left_slope, left_intercept)
    ly2 = top
    lx2 = get_x(ly2, left_slope, left_intercept)

    result.append([(lx1, ly1, lx2, ly2)])
    result.append([(rx1, ry1, rx2, ry2)])

    prev_slope_left = left_slope
    prev_slope_right = right_slope
    prev_intercept_left = left_intercept
    prev_intercept_right = right_intercept

    return result


debug = False


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
    threshold = 50
    min_line_len = 50
    max_line_gap = 150
    friction = 0.9

    lines = cv2.HoughLinesP(masked_image, rho, theta,
                            threshold, np.array([]), min_line_len, max_line_gap)
    line_img = np.zeros(
        (masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    final = cv2.addWeighted(image, 0.8, line_img, 1, 0)

    # debugging option for intermediate results
    if debug:
        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        x = [left_bottom[0], left_top[0], right_top[0], right_bottom[0]]
        y = [left_bottom[1], left_top[1], right_top[1], right_bottom[1]]
        plt.plot(x, y, 'b--', lw=2)
        plt.show()

        plt.figure(figsize=(15, 15))
        plt.imshow(gray_image, cmap='gray')
        plt.show()

        plt.figure(figsize=(15, 15))
        plt.imshow(blur_gray, cmap='gray')
        plt.show()

        plt.figure(figsize=(15, 15))
        plt.imshow(edges, cmap='gray')
        plt.show()

        plt.figure(figsize=(15, 15))
        plt.imshow(masked_image, cmap='gray')
        plt.show()

        plt.figure(figsize=(15, 15))
        plt.imshow(line_img)
        plt.show()

        plt.figure(figsize=(15, 15))
        plt.imshow(final)
        plt.show()

        cv2.line(final, (left_bottom[0], left_bottom[1]),
                 (left_top[0], left_top[1]), [0, 0, 255], 4)
        cv2.line(final, (right_bottom[0], right_bottom[1]),
                 (right_top[0], right_top[1]), [0, 0, 255], 4)

    return final
