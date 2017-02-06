import sys
import getopt
import scipy.io.wavfile
import numpy as np
import math

SOURCE_DIR = 'music_speech/'
CONVERSION_CONST = 32768.0

def extract_features(input_file, output_file):
    '''
    Reads an input_file line-by-line with each line having the format of <audio_file_path, label>
    The audio file is read and its audio features extracted (Root-mean-squared,
    Peak-to-average, Zero Crossing, Median Absolute Deviation, Mean Absolute Deviation)
    Writes the audio file path and the audio feature calculated values into a CSV output file.
    '''
    with open(input_file, 'r') as in_file:
        with open(output_file, 'w') as out_file:
           for line in in_file:
                file_name, label = line.split('\t')
                rate, arr = scipy.io.wavfile.read(SOURCE_DIR + file_name)
                converted = [x/CONVERSION_CONST for x in arr]
                rms = root_mean_squared(converted)
                pta = peak_to_ave(rms, converted)
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
    sum = 0
    for num in data:
        sum += num*num
    rms = math.sqrt(sum / float(len(data)))
    print "root mean squared: " + str(rms)
    return rms

def peak_to_ave(rms, data):
    pta = max(data) / rms
    print "peak to ave: " + str(pta)
    return pta

def zero_cross(data):
    sum = 0
    for idx in range(1, len(data)):
        x = data[idx]
        y = data[idx-1]
        prod = x * y
        if prod < 0:
            sum += 1
    zero_cross = sum / float((len(data)-1))
    print "Zero cross: " + str(zero_cross)
    return zero_cross

def median_abs_dev(data):
    median = np.median(data)
    dev_arr = [math.fabs(x-median) for x in data]
    mad = np.median(dev_arr)
    print "Median abs dev: " + str(mad)
    return mad

def mean_abs_dev(data):
    n = len(data)
    mean = sum(data) / float(n)
    total = 0
    for x in data:
        total += math.fabs(x-mean)
    mad = total / float(n)
    print "Mean abs dev: " + str(mad)
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
