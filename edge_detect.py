'''
Edge detection.

Author: Bernie Pope (bjpope@unimelb.edu.au).

This program implements the Sobel edge detection algorithm. The input
is a computer graphics image and the output is an edge image.
It can read and write standard image formats such as PNG and JPEG. 
The output edge image has the same dimensions as the input image.
White pixels with intensity 255 indicate edges in the input image.
Black pixels with intensity 0 indicate non-edges in the input image.
Each image coordinate in the input image is assigned an edge score
according to the output of the Sobel filter. A threshold parameter
is used to determine the edge response. Edge scores above the threshold
are turned into white pixels; everything else is turned into 
black pixels.

The minimum edge score is 0, the maximum edge score is roughly 1140.

Images are represented as lists of lists of integers. Each integer
P in an image represents a single (greyscale) pixel intensity in the
range 0 <= P <= 255, where 0 appears absolutely black, 255 appears
bright white, and everything in between is a shade of grey. Image
coordinates are zero-based and are written as (row, column) pairs. 
In an image with R rows and C columns the top-left pixel of the image is
at (0, 0) and the bottom right pixel of the image is at (R - 1, C - 1).
Images are stored in row major order, which means that the innermost lists
represent rows of pixels. We assume that images are rectangular, implying
that all rows have the same length. 

Colour images are supported as input. They are automatically converted
to greyscale before computing the edges.

Empty images (with zero pixels) are not supported.

The program does no error checking.

Example Usage:

    edge_detect('input_file.png', 'edge_image_output.png', 200)

This will read the file called input_file.png, detect edges with
edge score above 200, and write the resulting edge image to
edge_image_output.png.

Testing:

    run_tests()

This will run various unit tests on various parts of the program
and check the output for correctness.

Revision history:

3  Jun 2014: Implemented row and column gradient calculations.
4  Jun 2014: Implemented gradient magnitude and threshold as one function.
9  Jun 2014: Split gradient magnitude and threshold into two functions, and
             added CHECKER test case.
18 Jun 2014: Initial version of image convolution.
25 Jun 2014: Modified image convolution to avoid mutating input image. 
5  Jul 2014: Added test cases.
19 Jul 2014: Comments added.
'''

from SimpleImage import (get_height, get_width, read_image, write_image)
import math

# Intensity of the most white pixel.
WHITE_PIXEL = 255
# Intensity of the most black pixel:
BLACK_PIXEL = 0


def gradient_row(image, row, col):
    '''
    Approximate the ROW GRADIENT in the image at (row, col).

    The gradient is computed using the row kernel of the Sobel filter:

        1    2    1
        0    0    0
       -1   -2   -1

    The center of the above kernel is overlaid onto the image at
    (row, col). Note that when the kernel overlaps the edge of
    the image each missing pixel is replaced by its nearest in-bounds
    neighbor.

    The result is calculated as follows, for image I, row r, and 
    column c:

    G_row(I, r, c) = I(r - 1, c - 1) + 
                     2 * I(r - 1, c) + 
                     I(r - 1, c + 1) - 
                     I(r + 1, c - 1) - 
                     2 * I(r + 1, c) - 
                     I(r + 1, c + 1) 

    Parameters:

        image: a list of lists of pixel intensities.
        row: an integer within the bounds of the image height.
        col: an integer within the bounds of the image width.

    Result:

        An integer that is computed by G_row above.

    Examples, using CHECKER image defined in this module:

    >>> gradient_row(CHECKER, 0, 0)
    -37
    >>> gradient_row(CHECKER, 5, 2)
    -885
    >>> gradient_row(CHECKER, 7, 5)
    23
    >>> gradient_row(CHECKER, 9, 0)
    -47  
    '''
    # Note that get_pixel finds the nearest in-bounds
    # pixel in situations where (row, col) falls outside
    # the image bounds. Otherwise it finds the pixel
    # within the image at the desired coordinates.
    return get_pixel(image, row - 1, col - 1) + \
           2 * get_pixel(image, row - 1, col) + \
           get_pixel(image, row - 1, col + 1) - \
           get_pixel(image, row + 1, col - 1) - \
           2 * get_pixel(image, row + 1, col) - \
           get_pixel(image, row + 1, col + 1)


def gradient_col(image, row, col):
    '''
    Approximate the COLUMN GRADIENT in the image at (row, col).

    The gradient is computed using the column kernel of the Sobel filter:

       -1    0    1
       -2    0    2
       -1    0    1

    The center of the above kernel is overlaid onto the image at
    (row, col). Note that when the kernel overlaps the edge of
    the image each missing pixel is replaced by its nearest in-bounds
    neighbor.

    The result is calculated as follows, for image I, row r, and 
    column c:

    G_col(I, r, c) = I(r - 1, c + 1) +
                     2 * I(r, c + 1) +
                     I(r + 1, c + 1) -
                     I(r - 1, c - 1) -
                     2 * I(r, c - 1) -
                     I(r + 1, c - 1) 

    Parameters:

        image: a list of lists of pixel intensities.
        row: an integer within the bounds of the image height.
        col: an integer within the bounds of the image width.

    Result:

        An integer that is computed by G_col above.

    Examples, using CHECKER image defined in this module:

    >>> gradient_col(CHECKER, 0, 0)
    -27
    >>> gradient_col(CHECKER, 5, 2)
    31
    >>> gradient_col(CHECKER, 7, 5)
    -825
    >>> gradient_col(CHECKER, 9, 0)
    45
    '''
    # Note that get_pixel finds the nearest in-bounds
    # pixel in situations where (row, col) falls outside
    # the image bounds. Otherwise it finds the pixel
    # within the image at the desired coordinates.
    return get_pixel(image, row - 1, col + 1) + \
           2 * get_pixel(image, row, col + 1) + \
           get_pixel(image, row + 1, col + 1) - \
           get_pixel(image, row - 1, col - 1) - \
           2 * get_pixel(image, row, col - 1) - \
           get_pixel(image, row + 1, col - 1)


def gradient_magnitude(image, row, col):
    '''
    Approximate the magnitude of the steepest GRADIENT in the image
    at (row, col).

    The result is computed by combining the gradient row and gradient
    column as follows, for image I, row r, and column c:

    G_mag(I, r, c) = sqrt(G_col(I, r, c) ** 2 + G_row(I, r, c) ** 2)

    That is, we treat the gradient row and gradient column as vectors
    in their respective directions. The steepest gradient is then
    taken to be the magnitude of the vector resulting from adding
    the gradient row and gradient column together.

    The value of the magnitude is overestimated by a factor of 8,
    but we ignore this fact by treating its value as an edge
    score. The higher the score the more certain we are of the
    presence of an edge in the image.

    Parameters:

        image: a list of lists of pixel intensities.
        row: an integer within the bounds of the image height.
        col: an integer within the bounds of the image width.

    Result:

        An float that is computed by G_mag above.

    Examples, using CHECKER image defined in this module:

    >>> gradient_magnitude(CHECKER, 0, 0)
    45.803929962395145
    >>> gradient_magnitude(CHECKER, 5, 2)
    885.5427714119742
    >>> gradient_magnitude(CHECKER, 7, 5)
    825.3205437888967
    >>> gradient_magnitude(CHECKER, 9, 0)
    65.06919393998976
    '''
    grad_row = gradient_row(image, row, col)
    grad_col = gradient_col(image, row, col)
    return math.sqrt(grad_row ** 2 + grad_col ** 2)


def gradient_threshold(image, row, col, threshold):
    '''
    Compute the edge pixel intensity for the image at (row, col).

    The result is computed by comparing the gradient magnitude
    at the coordinate against the threshold parameter. If the
    gradient magnitude is GREATER THAN the threshold then the result
    edge pixel is WHITE, otherwise it is BLACK.

    Parameters:

        image: a list of lists of pixel intensities.
        row: an integer within the bounds of the image height.
        col: an integer within the bounds of the image width.
        threshold: an integer or a float indicating the edge score
        above which an edge is detected.

    Note: using the 3*3 Sobel filter, the gradient magnitude will not
    exceed 1141, so there is no point providing a threshold value
    higher than that. The minimum gradient magnitude will not be
    less than 0. However, if you want to treat 0 as an edge then
    the threshold will have to be -1. It makes no sense to provide
    a threshold less than -1.

    Result:

        An integer edge pixel intensity which is either absolutely
        BLACK or absolutely WHITE.

    Examples, using CHECKER image defined in this module:

    >>> gradient_threshold(CHECKER, 0, 0, 0)
    255
    >>> gradient_threshold(CHECKER, 0, 0, 50)
    0
    >>> gradient_threshold(CHECKER, 5, 2, 200)
    255
    >>> gradient_threshold(CHECKER, 7, 5, 900)
    0
    >>> gradient_threshold(CHECKER, 9, 0, 100)
    0
    '''
    grad_mag = gradient_magnitude(image, row, col)
    if grad_mag > threshold:
        return WHITE_PIXEL 
    else:
        return BLACK_PIXEL


def convolute(image, threshold):
    '''
    Compute an edge image by convolving the Sobel filter over
    the entire input image, for the given threshold. That is,
    we apply the gradient_threshold function at each input image
    coordinate, computing a new BLACK and WHITE edge image as
    a result. The output image has the same coordinates as the
    input image.

    Parameters:

        image: a list of lists of pixel intensities.
        threshold: an integer or a float indicating the edge score
        above which an edge is detected.

    Note: using the 3*3 Sobel filter, the gradient magnitude will not
    exceed 1141, so there is no point providing a threshold value
    higher than that. The minimum gradient magnitude will not be
    less than 0. However, if you want to treat 0 as an edge then
    the threshold will have to be -1. It makes no sense to provide
    a threshold less than -1.

    The input image is not mutated.

    Result:

        An edge image consisting of absolutely BLACK or 
        absolutely WHITE pixels. WHITE pixels indicate the presence
        of an edge (above the threshold) in the same coordinate
        in the input image.

    Example, using CHECKER image defined in this module:

    >>> convolute(CHECKER, 200)
    [[0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
     [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ]]
    '''
    # Compute the maximum dimensions of the input image.
    num_rows = get_height(image)
    num_cols = get_width(image)
    # Iterate over the input image and build a new edge
    # image as the result. Each pixel in the input image
    # is turned into either a BLACK or WHITE pixel in the output.
    new_image = []
    for row in range(num_rows):
        new_row = []
        for col in range(num_cols):
            # Approximate the image gradient at this coordinate,
            # and compare it to the threshold.
            edge_pixel = gradient_threshold(image, row, col, threshold)
            new_row.append(edge_pixel)
        new_image.append(new_row)
    return new_image


def clamp(val, lower_bound, upper_bound):
    '''Restrict a value to be within the range:

        lower_bound <= val <= upper_bound

    Return val if it is within the range, otherwise
    return the nearest bound to val.

    We assume (but do not check) that
    lower_bound <= upper_bound. The function may
    return the wrong result if this is not satisfied.

    This function was provided in the project
    specification.

    Examples:

    >>> clamp(10, 0, 100)
    10
    >>> clamp(-10, 0, 100)
    0 
    >>> clamp(110, 0, 100)
    100
    ''' 
    return max(min(val, upper_bound), lower_bound)


def get_pixel(image, row, col):
    '''Return the intensity of a pixel in image at
    coordinate (row, col) if that coordinate is
    within the bounds of the image. If the coordinate
    is outside the bounds of the image then return
    the intensity of its nearest in-bounds neighbor.

    image is a list of lists of pixel intensities
    (integers). row and col are integers.

    We assume (but do not check) that the image
    is not empty. An empty input image will result
    in an IndexError.

    This function was provided in the project
    specification.

    Examples:

    >>> example_image = [[1,2,3], [4,5,6]]
    >>> get_pixel(example_image, 0, 0)
    1
    >>> get_pixel(example_image, 0, 1)
    2
    >>> get_pixel(example_image, -1, 0)
    1
    >>> get_pixel(example_image, 1, 3)
    6
    '''
    # Find the bounds of the image.
    max_row = get_height(image) - 1
    max_col = get_width(image) - 1
    # Make sure the coordinate is within the image
    # bounds.
    new_row = clamp(row, 0, max_row)
    new_col = clamp(col, 0, max_col)
    return image[new_row][new_col]

def edge_detect(in_filename, out_filename, threshold):
    '''Apply the edge detection algorithm to an image file
    and save the result to an output file. Gradient scores
    above the threshold parameter are considered edges.

    Example, assuming we have a file called 'floyd.png'
    in the same directory as the program. The output
    will be saved in a file called 'floyd_edge.png':

    in_filename and out_filename are strings. threshold
    is a number (integer or float).

    The result is None.

    >>> edge_detect('floyd.png', 'floyd_edge.png', 200)
    '''
    in_image = read_image(in_filename)
    out_image = convolute(in_image, threshold)
    write_image(out_image, out_filename)


# Testing code and data below.

# 2 x 2 checkerboard image with random noise applied.
CHECKER = \
    [[11,  1,   2,   15,  35,  247, 205, 240, 214, 219],
     [17,  20,  24,  35,  0,   235, 235, 238, 249, 223],
     [17,  31,  27,  31,  46,  209, 239, 236, 247, 230],
     [12,  1,   37,  24,  38,  241, 219, 220, 231, 211],
     [37,  13,  19,  10,  44,  255, 220, 235, 227, 252],
     [243, 205, 227, 224, 239, 12,  21,  47,  42,  9  ],
     [220, 241, 234, 237, 223, 50,  16,  0,   1,   28 ],
     [248, 241, 207, 247, 218, 13,  14,  48,  39,  42 ],
     [204, 213, 230, 248, 213, 50,  3,   28,  25,  8  ],
     [215, 227, 249, 226, 254, 40,  11,  35,  48,  45 ]]

# The result of detecting edges in CHECKER with a threshold of 200.
CHECKER_EDGE_200 = \
    [[0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
     [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ],
     [0,   0,   0,   0,   255, 255, 0,   0,   0,   0  ]]

# An image with one row and column.
ONE_PIXEL_IMAGE = [[100]]

# An image with one row and seven columns.
ONE_ROW_IMAGE = [[100, 0, 200, 50, 50, 100, 22]]

# An image with seven rows and one column.
ONE_COL_IMAGE = [[100], [0], [200], [50], [50], [100], [22]]

# An image with all pixels having the same intensity.
FLAT_IMAGE = [[99, 99, 99, 99, 99],
              [99, 99, 99, 99, 99],
              [99, 99, 99, 99, 99],
              [99, 99, 99, 99, 99],
              [99, 99, 99, 99, 99]]

# An image with a sharp vertical edge.
VERT_EDGE = [[0, 0, 255, 255, 255],
             [0, 0, 255, 255, 255],
             [0, 0, 255, 255, 255],
             [0, 0, 255, 255, 255]]

# An image with a sharp horizontal edge.
HORIZ_EDGE = [[255, 255, 255],
              [255, 255, 255],
              [0, 0, 0],
              [0, 0, 0]]

# An image with random pixel intensities.
RANDOM_IMAGE = [[108, 171, 27,  252, 196],
                [19,  183, 134, 219, 169],
                [246, 246, 108, 105, 77],
                [177, 232, 22,  32,  95],
                [109, 211, 219, 218, 127],
                [134, 90,  177, 143, 150],
                [11,  4,   82,  112, 151],
                [243, 34,  148, 27,  84],
                [252, 137, 71,  125, 194]]

# Test cases for gradient_row.
# Each tuple contains (function call as string, expected output).
GRADIENT_ROW_TESTS = [
    ('gradient_row(ONE_PIXEL_IMAGE, 0, 0)', 0),
    ('gradient_row(ONE_ROW_IMAGE, 0, 3)', 0),
    ('gradient_row(ONE_COL_IMAGE, 2, 3)', -200),
    ('gradient_row(FLAT_IMAGE, 3, 1)', 0),
    ('gradient_row(VERT_EDGE, 1, 2)', 0),
    ('gradient_row(HORIZ_EDGE, 2, 1)', 1020),
    ('gradient_row(RANDOM_IMAGE, 6, 3)', 327)
    ]

# Test cases for gradient_col.
# Each tuple contains (function call as string, expected output).
GRADIENT_COL_TESTS = [
    ('gradient_col(ONE_PIXEL_IMAGE, 0, 0)', 0),
    ('gradient_col(ONE_ROW_IMAGE, 0, 6)', -312),
    ('gradient_col(ONE_COL_IMAGE, 6, 0)', 0),
    ('gradient_col(FLAT_IMAGE, 4, 0)', 0),
    ('gradient_col(VERT_EDGE, 3, 1)', 1020),
    ('gradient_col(HORIZ_EDGE, 1, 2)', 0),
    ('gradient_col(RANDOM_IMAGE, 0, 4)', -218)
    ]

# Test cases for gradient_magnitude.
# Each tuple contains (function call as string, expected output).
GRADIENT_MAG_TESTS = [
    ('gradient_magnitude(ONE_PIXEL_IMAGE, 0, 0)', 0.0),
    ('gradient_magnitude(ONE_ROW_IMAGE, 0, 0)', 400.0),
    ('gradient_magnitude(ONE_COL_IMAGE, 0, 0)', 400.0),
    ('gradient_magnitude(FLAT_IMAGE, 2, 4)', 0.0),
    ('gradient_magnitude(VERT_EDGE, 2, 1)', 1020.0),
    ('gradient_magnitude(HORIZ_EDGE, 1, 0)', 1020.0),
    ('gradient_magnitude(RANDOM_IMAGE, 1, 3)', 391.775445887)
    ]

# Test cases for gradient_threshold.
# Each tuple contains (function call as string, expected output).
GRADIENT_THRESH_TESTS = [
    ('gradient_threshold(FLAT_IMAGE, 2, 2, 0)', 0),
    ('gradient_threshold(FLAT_IMAGE, 2, 2, -1)', 255),
    ('gradient_threshold(VERT_EDGE, 3, 1, 1000)', 255),
    ('gradient_threshold(VERT_EDGE, 3, 1, 2000)', 0),
    ('gradient_threshold(HORIZ_EDGE, 2, 1, 1000)', 255),
    ('gradient_threshold(HORIZ_EDGE, 2, 1, 2000)', 0),
    ('gradient_threshold(RANDOM_IMAGE, 1, 3, 390)', 255),
    ]

# Test cases for convolute.
# Each tuple contains (function call as string, expected output).
CONVOLUTE_TESTS = [
    ('convolute(ONE_PIXEL_IMAGE, 0)', [[0]]),
    ('convolute(ONE_ROW_IMAGE, 300)',
        [[255, 255, 0, 255, 0, 0, 255]]),
    ('convolute(ONE_COL_IMAGE, 200)',
        [[255], [255], [0], [255], [0], [0], [255]]),
    ('convolute(FLAT_IMAGE, 0)',
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]),
    ('convolute(VERT_EDGE, 1000)',
        [[0, 255, 255, 0, 0],
         [0, 255, 255, 0, 0],
         [0, 255, 255, 0, 0],
         [0, 255, 255, 0, 0]]),
    ('convolute(HORIZ_EDGE, 1000)',
        [[0, 0, 0],
         [255, 255, 255],
         [255, 255, 255],
         [0, 0, 0]]),
    ('convolute(RANDOM_IMAGE, 500)',
        [[0,   0,   0,   255, 0],
         [255, 0,   0,   0,   255],
         [255, 0,   255, 255, 0],
         [0,   0,   255, 0,   0],
         [0,   0,   0,   0,   0],
         [255, 255, 255, 0,   0],
         [0,   0,   0,   0,   0],
         [255, 255, 0,   0,   0],
         [255, 255, 0,   0,   255]]),
    ('convolute(CHECKER, 200)', CHECKER_EDGE_200)
    ]


def tester(function_name, test_cases, equality):
    '''Test a function on a number of test cases, comparing
    the actual output with the expected output.
    Report messages for PASSED and FAILED test cases and
    report the number of each kind.

    Parameters:

        function_name: a string naming the function being tested.
        test_cases: a list of 2-tuples of the form:
            (test_code as a string, expected result)
        equality: a function which determines if two objects are
            equal. Used to check if the actual result is equal to
            the expected result.

    Result: None.
    '''
    print("*** TESTING %s ***\n" % function_name)
    num_passes = 0
    num_fails = 0
    for test_code, expect in test_cases:
        # Execute the test_code.
        # The test code is a string representing a Python
        # expression. We execute the Python expression by calling
        # the eval built in function.
        result = eval(test_code)
        # Check if the output of the test case is equal to the expected
        # output using the supplied equality function.
        if equality(result, expect):
            print("PASSED: " + test_code)
            num_passes += 1
        else:
            print("FAILED: " + test_code)
            # Print the expected result.
            print("\t\tExpected: %s" % expect)
            # Print the actual result.
            print("\t\tResult: %s" % result)
            num_fails += 1
    # Display a summary of the number of passes and fails.
    print("\nNum passes = %s" % num_passes)
    print("Num fails = %s\n" % num_fails)


def exact_equal(x, y):
    '''Test for exact equality between two objects.'''
    return x == y


def near_equal(x, y):
    '''Test if one float is sufficiently close to another to
    be considered equal.

    Note this function compares the absolute difference to
    some small value EPSILON. In general this technique is not
    recommended for floating point comparison, but it is
    sufficient for the purposes of this project.
    '''
    # Use a very lenient epsilon. We don't need to be too
    # stringent for this project. Generally this will be used
    # for comparing edge magnitudes, and we are not too concerned
    # if they are slightly different than the expected value.
    EPSILON = 0.005
    return x == y or abs(x - y) < EPSILON


def run_tests():
    '''Run all the test cases.'''
    tester('gradient_row', GRADIENT_ROW_TESTS, exact_equal)
    tester('gradient_col', GRADIENT_COL_TESTS, exact_equal)
    tester('gradient_magnitude', GRADIENT_MAG_TESTS, near_equal)
    tester('gradient_threshold', GRADIENT_THRESH_TESTS, exact_equal)
    tester('convolute', CONVOLUTE_TESTS, exact_equal)
