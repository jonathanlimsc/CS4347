import sys
import getopt
import scipy.io.wavfile
import numpy as np
import math

# Global constants
SOURCE_DIR = 'music_speech/'
CONVERSION_CONST = 32768.0

def extract_features(input_file, output_file):
    '''
    Reads an input_file line-by-line with each line having the format of <audio_file_path, label>
    The audio file is read and its audio features extracted (Root-mean-squared,
    Peak-to-average, Zero Crossing, Median Absolute Deviation, Mean Absolute Deviation)
    Writes the audio file path and the audio feature calculated values into a CSV output file.
    Output string format per audio file:
    <audio_file_path>,feature1,feature2,...,featureN
    '''
    with open(input_file, 'r') as in_file:
        with open(output_file, 'w') as out_file:
           for line in in_file:
                file_name, label = line.split('\t')
                rate, arr = scipy.io.wavfile.read(SOURCE_DIR + file_name)
                converted = [x/CONVERSION_CONST for x in arr]

                # Feature extraction methods
                rms = root_mean_squared(converted)
                pta = peak_to_ave(converted, rms)
                zc = zero_cross(converted)
                mad = median_abs_dev(converted)
                mean_ad = mean_abs_dev(converted)

                output_arr = [rms, pta, zc, mad, mean_ad]
                output_str = ','.join([format(x, '.6f') for x in output_arr])
                output_str = file_name + "," + output_str + "\n"
                print "Output String: " + output_str
                print " "
                out_file.write(output_str)
        out_file.close()
    in_file.close()

def root_mean_squared(data):
    '''
    Returns the root-mean-squared value of a list of data values
    '''
    sum = 0
    for num in data:
        sum += num*num
    rms = math.sqrt(sum / float(len(data)))
    return rms

def peak_to_ave(data, rms=None):
    '''
    Returns the peak-to-average ratio of a list of data values, given their root-mean-squared value
    '''
    if rms == None:
        rms = root_mean_squared(data)
    pta = max([math.fabs(x) for x in data]) / float(rms)
    return pta

def zero_cross(data):
    '''
    Returns the zero-crossing value of a list of data values
    '''
    sum = 0
    for idx in range(1, len(data)):
        x = data[idx]
        y = data[idx-1]
        prod = x * y
        if prod < 0:
            sum += 1
    zero_cross = sum / float((len(data)-1))
    return zero_cross

def median_abs_dev(data):
    '''
    Returns the median absolute deviation value of a list of data values
    '''
    median = np.median(data)
    dev_arr = [math.fabs(x-median) for x in data]
    mad = np.median(dev_arr)
    return mad

def mean_abs_dev(data):
    '''
    Returns the mean absolute deviation value of a list of data values
    '''
    n = len(data)
    mean = sum(data) / float(n)
    total = 0
    for x in data:
        total += math.fabs(x-mean)
    mad = total / float(n)
    return mad

def usage():
    print "Usage: " + sys.argv[0] + " -i <input_file_path> -o <output_file_path>"

input_file = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:o:')
except getopt.GetoptError, err:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-i':
        input_file = a
    elif o == '-o':
        output_file = a
    else:
        assert False, "unhandled option"
if input_file == None or output_file == None:
    usage()
    sys.exit(2)

extract_features(input_file, output_file)
