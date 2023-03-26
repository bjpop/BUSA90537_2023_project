'''
This module implements histogram equalization of grey-scale images.
Images are represented as rectangular lists of lists of pixel intensities.
Pixel intensities are integers in the range [0,255] inclusive.

Histogram equalization is a technique for increasing contrast in digital
images which tends to make details more apparent to the human eye. It works
by spreading out the distribution of pixel intensities, producing a (near)
linear cumulative distribution.

The module provides four main functions which test images for validity; 
compute histograms of pixel intensities; compute cumulative distributions
of pixel intensities; and perform histogram equalization on images.
'''

###############################################################################
# Authors:
#
# Bernie Pope (bjpope@unimelb.edu.au)
#
# Date created:
#
# 29 July 2012
#
# Wikipedia (dated 29 July 2012) contains a discussion of the algorithm
# employed by this program:
#
#    http://en.wikipedia.org/wiki/Histogram_equalization
#
# Date modified and reason:
#
# 20 September 2012: Added more detailed docstring comments to each function.
# 21 September 2012: Added more detailed header documentation.
# 25 September 2012: Refactored equalize to make algorithm clearer.
#
###############################################################################

from SimpleImage import get_width, get_height, read_image, write_image
import sys

# The maximum possible pixel intensity in an image.
intensity_upper_bound = 255
# The minimum possible pixel intensity in an image.
intensity_lower_bound = 0

def validate_image(image):
    '''validate_image(image) -> bool

    Test if the input image is valid according to the following rules:
        - An image is a list of lists of integers only (no other types are
          valid). 
        - All rows in the image are of the same length (the image is
          rectangular).
        - All integers i in the image lie in the range 0 <= i <= 255. 

    Examples:

    >>> validate_image([]) # the empty image is valid
    True
    >>> validate_image([1]) # not a list of lists of integers
    False
    >>> validate_image(None) # not a list at all
    False
    >>> validate_image([[0,23,97,45], [125,3,12,1], [8,8,8,8]])
    True
    >>> validate_image([[1,2,3], [4,5]]) # rows different length
    False
    >>> validate_image([[1,2,3], [4,5,256]]) # 256 > maximum intensity
    False
    '''
    # An image must be a list.
    if type(image) == list:
        len_first_row = None
        for row in image:
            # Each row in the image must be a list.
            if type(row) == list:
                # All rows in the image must be of the same length.
                if len_first_row == None:
                    # This is the first row, record its length.
                    len_first_row = len(row)
                # This is not the first row, compare its length
                # with the length of the first row.
                elif len(row) != len_first_row:
                    return False
                for pixel in row:
                    # Each pixel in a row must be an integer.
                    if type(pixel) == int:
                        # Each integer must be within the allowed intensity
                        # range.
                        if pixel < intensity_lower_bound or \
                           pixel > intensity_upper_bound:
                            return False
                    else:
                        return False
            else:
                return False
        # The image passed all the tests.
        return True
    else:
        return False

def histogram(image):
    '''histogram(image) -> histogram of pixel intensities

    Compute a histogram of pixel intensities for the input image.

    An image is a list of lists of pixel intensities and is assumed to be
    valid according to the validate_image function (histogram does not
    check the input image for validity).

    The output is a histogram of pixel intensities, which is represented
    as a list of 256 non-negative integers, such that the value at position N
    counts the number of times a pixel of intensity N appears in the image.
    If there are no pixels in the image of a particular intensity, then
    its count will be zero. 

    Examples:

    >>> histogram([]) == [0] * 256
    True
    >>> histogram([[2, 1, 4], [5, 4, 4]])
    [0, 1, 1, 0, 3, 1, 0, ... # ending in 249 consecutive zeros.
    '''
    # Initialise all the frequency counters to zero, one per pixel intensity.
    result = (intensity_upper_bound + 1) * [0]
    # Iterate over all the pixels in the image and count the
    # frequency of their intensities.
    for row in image:
        for pixel in row:
            result[pixel] += 1
    return result

def cumulative(histogram):
    '''cumulative(histogram) -> cumulative distribution

    Compute the cumulative distribution for the input histogram.

    The input histogram is (assumed to be) a list of non-negative integers,
    such that the value at index i represents the frequency of the value i.

    The output list has the same length as the input list. The value of the
    output list at index N is equal to the sum of all the values in the input
    list from index 0 to index N inclusive.

    Examples:

    >>> cumulative([])
    []
    >>> cumulative([4])
    [4]
    >>> cumulative([4, 7, 0, 12])
    [4, 11, 11, 23]
    '''
    # Compute a running sum of the frequencies in the histogram, and store
    # each value in the result.
    result = []
    sum = 0
    for frequency in histogram:
        sum += frequency 
        result.append(sum)
    return result

def equalize(image):
    '''equalize(image) -> image

    Compute a histogram-equalized version of the input image.

    An image is a list of lists of pixel intensities and is assumed to be
    valid according to the validate_image function (equalize does not
    check the input image for validity).

    Let d be the cumulative distribution of the input image, p be the total
    number of pixels in the image, and d_min be the minimum non-zero value
    in d. For each pixel in the input image with intensity i, the
    corresponding pixel in the output image is computed by:

        round ((d[i] - d_min) / (p - d_min) * 255)

    If the input image is empty, or all the pixels have the same intensity,
    equalization would be ineffective so we return the input image
    unchanged.

    Examples:
    
    >>> equalize([])               # empty image
    [] 
    >>> equalize([[10, 10, 10]])   # all pixels the same intensity
    [[10, 10, 10]] 
    >>> equalize([[100,101,102]])  # low contrast becomes high contrast
    [[0, 128, 255]] 
    '''
    if len(image) == 0:
        # The image is empty, return it unchanged.
        return image
    cumul_dist = cumulative(histogram(image))
    # Find the frequency of the minimum pixel intensity in the image.
    # This is equivalent to the minimum non-zero value in the
    # cumulative distribution. We take advantage of the fact that,
    # by definition, the cumulative distribution is monotonically
    # increasing.
    freq_min_pixel = 0
    for frequency in cumul_dist:
        if frequency != 0:
            freq_min_pixel = frequency
            break
    # If the image is not empty the frequency of the minimum 
    # pixel intensity must be greater than zero.
    assert(freq_min_pixel > 0)
    num_pixels = get_width(image) * get_height(image)
    # If all the pixels are the same intensity then the frequency of
    # the minimum pixel intensity will equal the total number of
    # pixels in the image. There is no point trying to equalize
    # such an image to we return it unchanged.
    if freq_min_pixel == num_pixels:
        return image
    # The difference between the number of pixels in the image and
    # the frequency of the minimum pixel intensity gives us a
    # measure of the range of pixel intensities used by original
    # image relative to the image size.
    cumul_range = float(num_pixels - freq_min_pixel)
    # Iterate over the original image and build a new image
    # by mapping the original pixel intensities to their new
    # equalized values.
    result = []
    for row in image:
        new_row = []
        for pixel in row:
            # Compute the proportion of total pixel intensities
            # which are used up-to-and-including this pixel
            # intensity. This yields a number in the range [0,1].
            norm_cumul_dist = float(cumul_dist[pixel] - freq_min_pixel)
            # Cumul_range is not zero because we have checked
            # that freq_min_pixel != num_pixels. This is floating
            # point division.
            proportion = norm_cumul_dist / cumul_range
            # Re-scale the pixel to the full intensity range.
            new_pixel = proportion * intensity_upper_bound
            new_row.append(int(round(new_pixel)))
        result.append(new_row)
    return result

def main():
    if len(sys.argv) == 3:
        input_filename = sys.argv[1]
        output_filename = sys.argv[2]
        input_image = read_image(input_filename)
        eq_image = equalize(input_image)
        write_image(eq_image, output_filename)
    else:
        print("Usage: python3 equalize.py input_filename output_filename")
        exit(1)
        
if __name__ == '__main__':
    main()

# Test cases

# Call a function on a test input and check if the actual output
# matches the expected output. Print whether the test passed,
# failed or raised an exception. Returns True if the test passes
# and False otherwise.
def run_test(function, test_input, expected_output):
    function_name = function.__name__
    # Print the test case.
    print("{0}({1})".format(function_name, str(test_input)))
    # Run the test, save the result, and check for any exceptions.
    try:
        actual_output = function(test_input)
    except:
        # The test case caused an exception to be raised.
        # Generate a nice textual representation of the exception,
        # and print it out.
        exception_string = str(sys.exc_info()[1])
        print("\traised exception: " + exception_string)
        return False
    else:
        # Check if the actual output of the function is equal to
        # the expected output.
        if actual_output != expected_output:
            print("\tfailed")
            print("\tactual output: " + str(actual_output))
            print("\texpected output: " + str(expected_output))
            return False
        else:
            print("\tpassed")
            return True

# Run a function on a list of tests and return the number of successes
# and failures.
def run_tests(function, tests):
    success, fail = 0, 0
    for test_input, expected_output, in tests:
        if run_test(function, test_input, expected_output):
            success += 1
        else:
            fail += 1
    return success, fail

# Test cases for the validate_image function.
def test_validate_image():
    tests = [
        ([], True),                     # Empty image.
        (None, False),                  # Image not list.
        ([[1]], True),                  # Single row, single pixel.
        ([[1], [None], [2]], False),    # Not integer pixel.
        ([[1], None, [2]], False),      # Not list row.
        ([[0], [1,2,3], [4,5]], False), # Rows different length.
        ([[1,256,3], [4,-5,6]], False), # Pixel not in range [0,255]
    ] 
    return run_tests(validate_image, tests)

# Test cases for the histogram function.
def test_histogram():
    tests = [
        ([], [0] * 256),                # Empty image.
        ([[1]], [0,1] + [0] * 254),     # Single pixel.
        ([[3,3,3],[3,3,3]], [0,0,0,6] + [0] * 252), # Repeat same pixel.
        (test_image, histogram_test_image)
    ] 
    return run_tests(histogram, tests)

# Test cases for the cumulative function.
def test_cumulative():
    tests = [
        ([0] * 256, [0] * 256),         # All zeros.
        ([1] * 256, range(1,257)),      # All ones.
        (histogram_test_image, cumulative_test_image)
    ] 
    return run_tests(cumulative, tests)

# Test cases for the equalize function.
def test_equalize():
    tests = [
        ([], []), # Empty image.
        ([[10,10,10]], [[10,10,10]]), # All the same intensity. 
        ([[10, 100]], [[0, 255]]), # Two different intensities
        (test_image, test_image_equalized)
    ] 
    return run_tests(equalize, tests)

# Run all the test cases for each of the four key functions.
# Count and print the number of passes and failures.
def run_all_tests():
    tests = [
        test_validate_image(),
        test_histogram(),
        test_cumulative(),
        test_equalize(),
    ]
    total_success, total_fail = 0, 0
    for success, fail in tests:
        total_success += success
        total_fail += fail
    print("{0} tests passed, {1} tests failed".format(total_success,
        total_fail))

# A test image taken from http://en.wikipedia.org/wiki/Histogram_equalization,
# dated 29 July 2012. 
test_image = [
   [ 52,  55,  61, 66,  70,  61,  64, 73 ],
   [ 63,  59,  55, 90,  109, 85,  69, 72 ],
   [ 62,  59,  68, 113, 144, 104, 66, 73 ],
   [ 63,  58,  71, 122, 154, 106, 70, 69 ],
   [ 67,  61,  68, 104, 126, 88,  68, 70 ],
   [ 79,  65,  60, 70,  77,  68,  58, 75 ],
   [ 85,  71,  64, 59,  55,  61,  65, 83 ],
   [ 87,  79,  69, 68,  65,  76,  78, 94 ] ]

# Histogram of pixel intensities for the test image.
histogram_test_image = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 2, 3, 1, 4, 1, 2, 2, 3, \
    2, 1, 5, 3, 4, 2, 1, 2, 0, 1, 1, 1, 1, 2, 0, 0, 0, 1, 0, 2, 0, 1, \
    1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 1, \
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

# Cumulative distribution of pixel intensities for the test image.
cumulative_test_image = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 4, 4, 4, 6, 9, 10, 14, 15, 17, 19, \
    22, 24, 25, 30, 33, 37, 39, 40, 42, 42, 43, 44, 45, 46, 48, 48, 48, \
    48, 49, 49, 51, 51, 52, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55, 55, \
    55, 55, 55, 55, 55, 57, 57, 58, 58, 58, 59, 59, 59, 59, 60, 60, 60, \
    60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 62, \
    62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, \
    63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, \
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, \
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, \
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, \
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, \
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, \
    64, 64, 64, 64 ]

# The equalized version of test_image, which is corroborated by the example
# on http://en.wikipedia.org/wiki/Histogram_equalization, dated 29 July 2012.
test_image_equalized = [
   [ 0,   12,  53,  93,  146, 53,  73,  166 ],
   [ 65,  32,  12,  215, 235, 202, 130, 158 ],
   [ 57,  32,  117, 239, 251, 227, 93,  166 ],
   [ 65,  20,  154, 243, 255, 231, 146, 130 ],
   [ 97,  53,  117, 227, 247, 210, 117, 146 ],
   [ 190, 85,  36,  146, 178, 117, 20,  170 ],
   [ 202, 154, 73,  32,  12,  53,  85,  194 ], 
   [ 206, 190, 130, 117, 85,  174, 182, 219 ] ]
