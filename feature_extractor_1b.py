import sys
import getopt
import scipy.io.wavfile
import numpy as np
import math

# Global constants
SOURCE_DIR = 'music_speech/'
CONVERSION_CONST = 32768.0
STEP_SIZE = 512
WINDOW_SIZE = 1024
NUM_ROWS = 1290
NUM_FEATURES = 5
HEADER = '''@RELATION music_speech
@ATTRIBUTE RMS_MEAN NUMERIC
@ATTRIBUTE PAR_MEAN NUMERIC
@ATTRIBUTE ZCR_MEAN NUMERIC
@ATTRIBUTE MAD_MEAN NUMERIC
@ATTRIBUTE MEAN_AD_MEAN NUMERIC
@ATTRIBUTE RMS_STD NUMERIC
@ATTRIBUTE PAR_STD NUMERIC
@ATTRIBUTE ZCR_STD NUMERIC
@ATTRIBUTE MAD_STD NUMERIC
@ATTRIBUTE MEAN_AD_STD NUMERIC
@ATTRIBUTE class {music,speech}

@DATA
'''

def extract_features(input_file, output_file):
    '''
    Reads an input_file line-by-line with each line having the format of <audio_file_path, label>
    The audio file is read into a list of floats. From that, a feature matrix is
    generated based on number of desired buffers, which is dependent on STEP_SIZE and WINDOW_SIZE of the buffer.
    Feature extraction will be applied to every buffer. Finally, the mean and std of each feature is calculated.

    Hence, each audio file will be represented as a list of mean and std values.
    An output string will be written to the output_file in this format:
    feature1_mean,feature2_mean,...,featureN_mean,feature1_std,feature2_std,...,featureN_std,label
    '''
    with open(input_file, 'r') as in_file:
        with open(output_file, 'w') as out_file:
            # Write file header
            out_file.write(HEADER)
            for line in in_file:
                file_name, label = line.split('\t')
                rate, arr = scipy.io.wavfile.read(SOURCE_DIR + file_name)
                converted = [x/CONVERSION_CONST for x in arr]

                # Calculate number of buffers
                remainder = len(converted) % STEP_SIZE
                if remainder != 0:
                    num_buffers = len(converted) / STEP_SIZE - 1
                else:
                    num_buffers = len(converted) / STEP_SIZE

                # Build feature matrix of shape(num_buffers, NUM_FEATURES)
                feature_mtx = build_feature_matrix(num_buffers, converted)

                # Generate array with mean and std stats from the feature matrix
                file_stats = generate_stats(feature_mtx)

                # Format output string
                output_str = ','.join([format(x, '.6f') for x in file_stats]) + "," + label
                print output_str
                print " "
                out_file.write(output_str)
        out_file.close()
    in_file.close()

def build_feature_matrix(num_buffers, data):
    '''
    Builds a feature matrix from a file's array representation, data.
    num_buffers denote the number of sub-arrays of data for which
    feature extraction will be done. Hence, the resulting matrix will be
    of shape(num_buffers, NUM_FEATURES)
    '''
    feature_mtx = np.zeros((num_buffers, NUM_FEATURES))
    row_idx = 0
    for idx in range(0, len(data), STEP_SIZE):
        buffer_data = data[idx:idx+WINDOW_SIZE]
        if len(buffer_data) < WINDOW_SIZE:
            # Case where last buffer is not window size
            continue

        # Feature extraction methods
        rms = root_mean_squared(buffer_data)
        pta = peak_to_ave(buffer_data, rms)
        zc = zero_cross(buffer_data)
        mad = median_abs_dev(buffer_data)
        mean_ad = mean_abs_dev(buffer_data)

        feature_mtx[row_idx] = [rms, pta, zc, mad, mean_ad]
        row_idx += 1
    return feature_mtx

def generate_stats(feature_mtx):
    '''
    Given a file's feature matrix, generate a stats array of shape(1, NUM_FEATURES*2)
    with the format [feature1_mean, feature2_mean, ..., featureN_mean, feature1_std, feature2_std, ..., featureN_std]
    '''
    mean_arr = []
    std_arr = []
    for col in range(0, NUM_FEATURES):
        feature_data = feature_mtx[:, col]
        mean_arr.append(mean(feature_data))
        std_arr.append(std(feature_data))

    file_stats = mean_arr + std_arr
    return file_stats

def mean(data):
    '''
    Returns the mean of a list of data values
    '''
    return sum(data) / float(len(data))

def std(data):
    '''
    Returns the standard deviation (uncorrected) of a list of data values
    '''
    n = len(data)
    sum_xsq = sum([x*x for x in data])
    sum_x = sum(data)
    sd = math.sqrt((sum_xsq - (sum_x ** 2 / float(n)))/float(n))
    return sd

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
