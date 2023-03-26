'''
###############################################################################

Steganography
-------------

This module implements a steganographic scheme whereby ASCII text messages
can be embedded (and hidden) within digital computer images. The purpose
of the scheme is to allow users to hide secret information in otherwise
innocent-looking pictures.

Images are represented internally as rectangular lists of lists of pixel
intensities. Pixels are 3-element tuples of integer intensity values
in the range [0,255] inclusive.

Two steganographic schemes are implemented. In both schemes the ASCII message
is turned into a sequence of bits by converting each character to its
ASCII code. The ASCCI code is then converted to 8 binary digits (a byte).
Finally, the bytes for all characters are appended together in order.

In the first scheme, the least significant bit (LSB) of each intensity
value is assigned to the next bit from the binary message. In the second
scheme, the bottom N bits of each intensity value is assigned to the next
N bits from the message, where N is in the range [1,8] inclusive.
The second scheme is a generalisation of the first scheme. The fewer
bottom bits we use in each intensity value the lesser the visual
change to the original image. However, using more bottom bits per
intensity value allows us to encode larger messages in the same image.

If there are fewer intensity values in the image than bits in the message
both schemes encode as many bits as possible; the encoded
message will be truncated. Conversely, if there are fewer bits in the
message than intensity values in the image, then the binary message
is paddded with sufficiently many 0 digits.

When decoding messages from encoded images we retrieve all the encoded
bits from the intensity values first. Then we decode each byte chunk
in order, yielding one ASCII character per chunk. We ignore
any bits left over if there are fewer than 8 at the end. We stop
decoding the bits if we encounter a all-zero byte 00000000 - this
indicates that there are no more encoded characters remaining. The
zero byte works because of the way we padded the binary message with
0 bits where necessary.

This code is an attempt to satisfy the requirements of Project 2 for the
University of Melbourne subject COMP10001 "Foundations of Computing",
Semester 2 2013.

The inputs to each function are assumed to be valid, we make no attempt
to check this in the program.

The program is designed to be used with Python 2.6 and 2.7.

You can run all the tests for the project using the run_all_tests()
function:
>>> run_all_tests()
This will report how many tests have passed and failed. Note that some of
the tests rely on external files.

Authors:

Bernie Pope (bjpope@unimelb.edu.au)

Date created:

30 Sep 2013

Date modified and reason:

1  Oct 2013: Encode and decode implemented.
6  Oct 2013: Round trip testing added.
9  Oct 2013: Extended encode and decode implemented.
10 Oct 2013: Extended test harness to support functions with multiple args.
16 Oct 2013: More detailed header documentation.
20 Oct 2013: Documentation for functions, and more test cases added.
21 Oct 2013: Larger test cases for extended encode/decode, based on floyd.png.
22 Oct 2013: Examples added to doc strings for functions.
23 Oct 2013: Fixed warnings from pep8 and pylint.

###############################################################################
'''

from bits import char_to_bits, bits_to_char, get_bit, set_bit
from SimpleImage import read_image


def message_to_bits(message):
    '''Convert a string of ASCII characters to a string of bits.
    Each character is converted to its ASCII code, and the code is then
    turned into an 8 bit string. The bit strings for all characters
    in the message are appended together to form the final result.

    INPUTS:

        message: a string of ASCII characters.

    RESULT:

        A string of bits: '0's and '1's.

    EXAMPLES:

        >>> message_to_bits('')
        ''
        >>> message_to_bits('abc 123')
        '01100001011000100110001100100000001100010011001000110011'
    '''
    # Convert each character to an 8 bit byte, and append them all together
    # in order.
    result = ''
    for char in message:
        result += char_to_bits(char)
    return result


# Eight zero bits, indicates the end of the encoded message
ZERO_BYTE = '00000000'
# Number of bits in a byte, we assume ASCII characters are represented
# in this many bits.
BYTE_SIZE = 8


def bits_to_message(bits):
    '''Convert a string of bits into a string of ASCII characters.

    The input is decoded in 8 bit chunks. Each chunk is converted
    into an integer, and that integer is then converted to
    an ASCII character. The integer is taken to be the ASCII code
    of the character.

    We stop decoding if:
        - there are fewer than 8 bits left in the input.
        - we encounter a sequence of 8 zero bits.

    INPUTS:

        bits: A string of bits: '0's and '1's

    RESULT:

        A string of ASCII characters.

    EXAMPLES:

        >>> bits_to_message('')
        ''
        >>> bits_to_message('0100000')
        ''
        # Observere truncation of message due to insufficient bits
        >>> bits_to_message('011010000110010101')
        'he'
        >>> bits_to_message('0110100001100101011011000110110001101111')
        'hello'
        # Observe the truncation of the message due to zero byte
        >>> bits_to_message('011010000110010101101100000000000110110001101111')
        'hel'
    '''
    result = ''
    while bits:
        # Grab the next 8 bit chunk
        next_chunk = bits[:BYTE_SIZE]
        # Check if the next chunk is 8 bits long and not
        # the zero byte.
        if len(next_chunk) == BYTE_SIZE and next_chunk != ZERO_BYTE:
            # Convert this chunk into an ASCII character
            result += bits_to_char(next_chunk)
            # Move on to the rest of the bits after the chunk
            bits = bits[BYTE_SIZE:]
        else:
            # Stop decoding the bits.
            # We either ran out of bits, or we hit the zero byte.
            break
    return result


def get_message_bit(bits, index):
    '''Get the bit at a given position within a string of bits.
    If the position is out of bounds, return '0'.

    INPUTS:

        bits:  A possibly empty string of bits: '0's and '1's.

        index: A integer index (may be outside the bounds of
               the bit string).

    RESULT:

        A single bit as a string. If the index is within the bounds
        of bits then the result is bits[index], otherwise the
        result is '0'.

    EXAMPLES:

        >>> get_message_bit('', 0)
        '0'
        >>> get_message_bit('1011', 0)
        '1'
        >>> get_message_bit('1011', 1)
        '0'
        >>> get_message_bit('1011', 2)
        '1'
        >>> get_message_bit('1011', 3)
        '1'
        # Note the index is out of range:
        >>> get_message_bit('1011', 4)
        '0'
        # Note the index is out of range:
        >>> get_message_bit('1011', -1)
        '0'
    '''
    if index < 0 or index >= len(bits):
        return '0'
    else:
        return bits[index]

# Index of the least significant bit in a sequence of bits.
LSB_POS = 0


def encode(image, message):
    '''Embed a message within an image.

    This is done by turning the message into a sequence of bits.
    Each bit from the message is then set as the least significant
    bit (LSB) in a corresponding pixel intensity value.

    The message can be decoded from the image using the decode
    function below.

    If there are NUM_INTENSITY values in the image then
    the following equality holds for all messages (using integer
    division):

        NUM_ENCODED_CHARS = NUM_INTENSITY_VALUES / 8
        decode(encode(image, message)) == message[:NUM_ENCODED_CHARS]

    That is to say that the result of decoding an encoded image should
    be equal to a prefix of the original message. The exact length
    of the prefix depends on the number of intensity values in the
    image, which determines how many characters can be stored.

    INPUTS:

        image: A rectangular lists of lists of pixel intensities.
               Pixels are 3-element tuples of integer intensity values
               in the range [0,255] inclusive.

        message: A possibly empty string of ASCII characters.

    RESULT:

    An image in the same format as the input image. The output image
    contains as many bits as possible from the input message,
    encoded in the LSBs of the pixel intensities.

    EXAMPLE:

    >>> test_image = [[(15, 103, 255), (0, 3, 19)],
                     [(22, 200, 1), (8, 8, 8)],
                     [(0, 0, 0), (5, 123, 19)]]
    >>> encode(test_image, "hello")
    [[(14, 103, 255), (0, 3, 18)], [(22, 200, 0), (9, 9, 8)],
     [(0, 1, 0), (5, 122, 19)]]
    '''
    # Convert the entire message to a string of bits
    message_bits = message_to_bits(message)
    # The position of the next bit in the message to encode
    msg_bit_pos = 0
    # Construct a new image, one row at a time
    new_image = []
    for row in image:
        new_row = []
        for pixel in row:
            # Build a new pixel as a list so we can append
            # intensity values onto it. We turn it back into a tuple
            # just before adding it to the new row.
            new_pixel = []
            for intensity in pixel:
                # Get the next bit from the message (will be '0' if
                # we run past the end).
                next_bit = get_message_bit(message_bits, msg_bit_pos)
                # Set the LSB of the next intensity value to be equal
                # to the next bit.
                new_intensity = set_bit(intensity, next_bit, LSB_POS)
                new_pixel.append(new_intensity)
                # Move onto the next bit.
                msg_bit_pos += 1
            # Turn the pixel into a tuple and then add it to the next row
            new_row.append(tuple(new_pixel))
        new_image.append(new_row)
    return new_image


def decode(image):
    '''Retrieve an encoded message from an image.

    The input image is assumed to have a message encoded in its LSBs according
    to the scheme implemented by the encode function. See the comments on
    encode for more detailed information.

    INPUTS:

        image: A rectangular lists of lists of pixel intensities.
               Pixels are 3-element tuples of integer intensity values
               in the range [0,255] inclusive.

    RESULT:

        A possibly empty string of ASCII characters.

    EXAMPLE:
    >>> test_image = [[(15, 103, 255), (0, 3, 19)],
                      [(22, 200, 1), (8, 8, 8)],
                      [(0, 0, 0), (5, 123, 19)]]
    # Note that the image is not large enough to encode the whole message
    >>> decode(encode(test_image, 'hello'))
    'he'
    '''
    # Collect all the LSBs of each intensity value as one long string
    # of bits, then decode back to a string of ASCII characters.
    message_bits = ''
    for row in image:
        for pixel in row:
            for intensity in pixel:
                message_bits += get_bit(intensity, LSB_POS)
    return bits_to_message(message_bits)

###############################################################################
# Bonus task
#
# The bonus task is a generalisation of the encoding scheme above, where
# we allow the bottom N bits of each intensity value to encode the next N bits
# from the message. When N == 1 the schemes are identical.
#
###############################################################################


def get_message_bits(bits, index, num_bits):
    '''Get a sequence of bits of length num_bits at a given position within
    a string of bits. The result is padded with '0's if the requested
    sequence runs over the end of the input string of bits.

    INPUTS:

        bits:  A possibly empty string of bits: '0's and '1's.

        index: A integer index (may be outside the bounds of
               the bit string.

        num_bits: A positive integer indicating the number of bits requested.

    RESULT:

        A bit string of length num_bits.

    EXAMPLES:

        >>> get_message_bits('', 0, 3)
        '000'
        >>> get_message_bits('0110011', 0, 3)
        '011'
        >>> get_message_bits('0110011', 1, 3)
        '110'
        >>> get_message_bits('0110011', 2, 3)
        '100'
        >>> get_message_bits('0110011', 3, 3)
        '001'
        >>> get_message_bits('0110011', 4, 3)
        '011'
        >>> get_message_bits('0110011', 5, 3)
        '110'
        >>> get_message_bits('0110011', 6, 3)
        '100'
        >>> get_message_bits('0110011', 7, 3)
        '000'
    '''
    # We call upon the get_message_bit function
    # num_bits in a row. The get_message_bit function handles the
    # situation where the index falls outside the bounds of
    # the bit string, thus padding with '0's as necessary.
    result = ''
    for bit_index in range(index, index + num_bits):
        result += get_message_bit(bits, bit_index)
    return result


def encode_ext(image, message, num_bits):
    '''Embed a message within an image.

    This is done by turning the message into a sequence of bits.
    Each sequence of num_bits from the message is then set as
    bottom num_bits of each intensity value in the image.

    The message can be decoded from the image using the decode_ext
    function below.

    If there are NUM_INTENSITY values in the image then
    the following equality holds for all messages (using integer
    division):

        NUM_ENCODED_CHARS = NUM_INTENSITY_VALUES * num_bits / 8
        decode_ext(encode_ext(image, message)) == message[:NUM_ENCODED_CHARS]

    That is to say that the result of decoding an encoded image should
    be equal to a prefix of the original message. The exact length
    of the prefix depends on the number of intensity values in the
    image and the number of bits encoded in each intensity value
    (num_bits) which determines how many characters can be stored.

    INPUTS:

        image: A rectangular lists of lists of pixel intensities.
               Pixels are 3-element tuples of integer intensity values
               in the range [0,255] inclusive.

        message: A possibly empty string of ASCII characters.

        num_bits: The number of bottom (least significant) bits
                  to use for encoding the message in each intensity
                  value.

    RESULT:

    An image in the same format as the input image. The output image
    contains as many bits as possible from the input message,
    encoded in the bottom num_bits of the pixel intensities.

    EXAMPLE:

    >>> test_image = [[(15, 103, 255), (0, 3, 19)],
                     [(22, 200, 1), (8, 8, 8)],
                     [(0, 0, 0), (5, 123, 19)]]
    >>> encode_ext(test_image, "hello", 2)
    [[(14, 101, 253), (0, 2, 17)], [(22, 202, 2), (9, 11, 8)],
     [(2, 1, 3), (4, 122, 17)]]
    >>> encode_ext(test_image, "hello", 8)
    [[(22, 166, 54), (54, 246, 0)], [(0, 0, 0), (0, 0, 0)],
     [(0, 0, 0), (0, 0, 0)]
    '''
    # Turn the whole message into a sequence of bits
    message_bits = message_to_bits(message)
    # The position of the next bit sequence in the message to encode
    msg_bit_pos = 0
    # Construct a new image, one row at a time
    new_image = []
    for row in image:
        new_row = []
        for pixel in row:
            # Build a new pixel as a list so we can append
            # intensity values onto it. We turn it back into a tuple
            # just before adding it to the new row.
            new_pixel = []
            for intensity in pixel:
                # Get the next bit sequence from the message (will be padded
                # with '0's if we run past the end).
                next_bits = get_message_bits(message_bits, msg_bit_pos,
                                             num_bits)
                # Set the bottom num_bits of the next intensity value to be
                # equal to the next bit.
                for pos, next_bit in enumerate(next_bits):
                    intensity = set_bit(intensity, next_bit, pos)
                new_pixel.append(intensity)
                # Move onto the next sequence of bits
                msg_bit_pos += num_bits
            # Turn the pixel into a tuple and then add it to the next row
            new_row.append(tuple(new_pixel))
        new_image.append(new_row)
    return new_image


def decode_ext(image, num_bits):
    '''Retrieve an encoded message from an image.

    The input image is assumed to have a message encoded in the bottom
    num_bits according to the scheme implemented by the encode_ext function.
    See the comments on encode_ext for more detailed information.

    INPUTS:

        image: A rectangular lists of lists of pixel intensities.
               Pixels are 3-element tuples of integer intensity values
               in the range [0,255] inclusive.

        num_bits: The number of bottom (least significant) bits
                  to use for encoding the message in each intensity
                  value. It is important that the value of num_bits
                  is used for both encode_ext and decode_ext.

    RESULT:

        A possibly empty string of ASCII characters.

    EXAMPLE:

    >>> decode_ext(encode_ext(test_image, 'hello', 0), 0)
    ''
    >>> decode_ext(encode_ext(test_image, 'hello', 1), 1)
    'he'
    >>> decode_ext(encode_ext(test_image, 'hello', 2), 2)
    'hell'
    >>> decode_ext(encode_ext(test_image, 'hello', 3), 3)
    'hello'
    >>> decode_ext(encode_ext(test_image, 'hello', 4), 4)
    'hello'
    >>> decode_ext(encode_ext(test_image, 'hello', 8), 8)
    'hello'
    '''
    # Collect all bottom num_bits of each intensity value as one long string
    # of bits, then decode back to a string of ASCII characters.
    message_bits = ''
    for row in image:
        for pixel in row:
            for intensity in pixel:
                # Retrieve each of the bottom num_bits of the intensity value.
                # We must be careful to retrieve them in the same order that
                # they were set in encode_ext.
                for pos in range(num_bits):
                    message_bits += get_bit(intensity, pos)
    return bits_to_message(message_bits)


###############################################################################
# Automated testing
###############################################################################

def run_all_tests():
    '''Run all the test cases for each of the key functions.
    Count and print the number of passes and failures.
    '''
    tests = [
        test_message_to_bits(),
        test_bits_to_message(),
        test_round_trip(),
        test_encode_small(),
        test_decode_small(),
        test_encode_decode_file(),
        test_encode_decode_ext_file()
    ]
    total_success, total_fail = 0, 0
    for success, fail in tests:
        total_success += success
        total_fail += fail
    print("{0} tests passed, {1} tests failed".format(total_success,
          total_fail))


def run_test(function, test_inputs, expected_output):
    '''Call a function on a test input and check if the actual output
    matches the expected output. Print whether the test passed or
    failed.
    '''
    function_name = function.__name__
    # Print the test case.
    print("{0}({1})".format(function_name, ', '.join(map(repr, test_inputs))))
    # Run the test, save the result
    # Note the use of * args to allow the testing of functions with
    # any number of arguments.
    actual_output = function(*test_inputs)
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


def run_tests(function, tests):
    '''Run a function on a list of tests and return the number of successes
    and failures.
    '''
    success, fail = 0, 0
    for test_inputs, expected_output, in tests:
        if run_test(function, test_inputs, expected_output):
            success += 1
        else:
            fail += 1
    return success, fail


def test_message_to_bits():
    "Test cases for the message_to_bits function"
    tests = [
        ([''], ''),
        (['A'], '01000001'),
        (['hello'], '0110100001100101011011000110110001101111'),
        (['ABC\n123'],
         '01000001010000100100001100001010001100010011001000110011'),
        (["je t'aime"],
         '011010100110010100100000011101000'
         '010011101100001011010010110110101100101')
    ]
    return run_tests(message_to_bits, tests)


def test_bits_to_message():
    "Test cases for the bits_to_message function"
    tests = [
        ([''], ''),
        (['0100000'], ''),
        (['01000001'], 'A'),
        (['011010000110010101'], 'he'),
        (['0110100001100101011011000110110001101111'], 'hello'),
        (['011010000110010101101100000000000110110001101111'], 'hel'),
        (['01000001010000100100001100001010001100010011001000110011'],
         'ABC\n123')
    ]
    return run_tests(bits_to_message, tests)


def round_trip(message):
    "Round trip on bits_to_message and message_to_bits"
    return bits_to_message(message_to_bits(message)) == message


def test_round_trip():
    "Test cases for round trip on bits_to_message and message_to_bits"
    tests = [
        ([''], True),
        (['X'], True),
        (['XX'], True),
        (['XY'], True),
        (['\t\n '], True),
        (['#!@~%^&*(){}[]:;"\'<>.,/?+=-_`'], True)
    ]
    return run_tests(round_trip, tests)

# A small image for testing purposes
SMALL_TEST_IMAGE = [
    [(15, 103, 255), (0, 3, 19)],
    [(22, 200, 1), (8, 8, 8)],
    [(0, 0, 0), (5, 123, 19)]
]

# The small test image with a messsage "he" encoded
# within it.
SMALL_TEST_IMAGE_ENCODED = [
    [(14, 103, 255), (0, 3, 18)],
    [(22, 200, 0), (9, 9, 8)],
    [(0, 1, 0), (5, 122, 19)]
]


def test_encode_small():
    "Test cases for the encode function on the small test image."
    tests = [
        ([SMALL_TEST_IMAGE, 'hello'], SMALL_TEST_IMAGE_ENCODED)
    ]
    return run_tests(encode, tests)


def test_decode_small():
    "Test cases for the decode function on the small test image."
    tests = [
        ([SMALL_TEST_IMAGE_ENCODED], 'he')
    ]
    return run_tests(decode, tests)


# We should add test cases where the message is larger than
# the image,
def test_encode_decode_file():
    "Test cases for encode and decode from an image file."
    def encode_decode_file(filename, message):
        image = read_image(filename)
        encoded_image = encode(image, message)
        decoded_message = decode(encoded_image)
        return decoded_message
    tests = [
        (['floyd.png', 'Floyd is cute!'], 'Floyd is cute!'),
        (['floyd.png', ''], '')
    ]
    return run_tests(encode_decode_file, tests)


# We should add test cases where the message is larger than
# the image,
def test_encode_decode_ext_file():
    "Test cases for encode_ext and decode_ext from an image file."
    def encode_decode_ext_file(filename, message, num_bits):
        image = read_image(filename)
        encoded_image = encode_ext(image, message, num_bits)
        decoded_message = decode_ext(encoded_image, num_bits)
        return decoded_message
    tests = [
        (['floyd.png', 'Floyd is cute!', 2], 'Floyd is cute!'),
        (['floyd.png', 'Floyd is cute!', 8], 'Floyd is cute!'),
    ]
    return run_tests(encode_decode_ext_file, tests)
