import numpy as np
import cv2
import os

def grayscale(img):
    """Applies the Grayscale transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    """

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=8):
    """
    line segments are separated by their slope ((y2-y1)/(x2-x1)) to decide which segments
    are part of the left line vs. the right line.  Then, you can average the position of
    each of the lines and extrapolate to the top and bottom of the lane.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    initial_img * α + img * β + γ
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def main():
    input_folder = 'Specify input folder'
    output_folder = 'Specify output folder'

    for filename in os.listdir(input_folder):
        image = cv2.imread(os.path.join(input_folder, filename))
        cv2.imshow('Image',image)
        cv2.waitKey()

        # Convert the image to grayscale
        grayScale = grayscale(image)
        cv2.imshow('GrayScale',grayScale)
        cv2.waitKey()

        # Apply Gaussian blur
        gaussianBlur = gaussian_blur(grayScale, 3)
        cv2.imshow('Gaussian_Blur', gaussianBlur)
        cv2.waitKey()

        # Apply Canny Edge
        low_threshold = 50
        high_threshold = 150
        cannyEdge = canny(gaussianBlur, low_threshold, high_threshold)
        cv2.imshow('Canny_Edge',cannyEdge)
        cv2.waitKey()


        # Finding region of interest
        imshape = image.shape
        vertices = np.array([[(131, 538), (443, 324), (540, 324), (imshape[1], imshape[0])]], dtype=np.int32)
        roiImage = region_of_interest(cannyEdge, vertices)
        cv2.imshow('Region_of_Interest',roiImage)
        cv2.waitKey()

        # Apply Hough Transform
        rho = 2
        theta = np.pi / 180
        threshold = 1
        min_line_len = 1
        max_line_gap = 5

        houghLines = hough_lines(roiImage, rho, theta, threshold, min_line_len, max_line_gap)
        cv2.imshow('Hough_Lines',houghLines)
        cv2.waitKey()

        # Extend the lines
        weightImage = weighted_img(image, houghLines, 0.8, β=1., γ=0.)
        cv2.imshow('Weighted_Image',weightImage)
        cv2.waitKey()
        stack = np.hstack((image,weightImage))
        cv2.imshow('Side_by_side',stack)
        cv2.waitKey()
        cv2.imwrite(output_folder + '/' + filename,stack)


if __name__ == '__main__':
    main()