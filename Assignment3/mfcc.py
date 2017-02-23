import sys
import getopt
import scipy.io.wavfile
import scipy.signal
import scipy.fftpack
import numpy as np
import math

# Global constants
SOURCE_DIR = '../../music_speech/'
CONVERSION_CONST = 32768.0
STEP_SIZE_RATIO = 0.5
WINDOW_SIZE = 1024
NUM_FEATURES = 5
L_VALUE = 0.85
NUM_FILTER_BANKS = 26
HEADER = '''@RELATION music_speech
@ATTRIBUTE SC_MEAN NUMERIC
@ATTRIBUTE SRO_MEAN NUMERIC
@ATTRIBUTE SFM_MEAN NUMERIC
@ATTRIBUTE PARFFT_MEAN NUMERIC
@ATTRIBUTE FLUX_MEAN NUMERIC
@ATTRIBUTE SC_STD NUMERIC
@ATTRIBUTE SRO_STD NUMERIC
@ATTRIBUTE SFM_STD NUMERIC
@ATTRIBUTE PARFFT_STD NUMERIC
@ATTRIBUTE FLUX_STD NUMERIC
@ATTRIBUTE class {music,speech}

@DATA
'''

def extract_features(input_file, output_file):
    '''
    Reads an input_file line-by-line with each line having the format of <audio_file_path, label>
    The audio file is read into a list of floats. From that, a feature matrix is
    generated based on number of desired buffers, which is dependent on STEP_SIZE_RATIO and WINDOW_SIZE of the buffer.
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

                # Build feature matrix of shape(NUM_FEATURES)
                feature_mtx = build_feature_matrix(converted, rate)

                # Generate array with mean and std stats from the feature matrix
                file_stats = generate_stats(feature_mtx)

                # Format output string
                output_str = ','.join([format(x, '.6f') for x in file_stats]) + "," + label
                print output_str
                print " "
                out_file.write(output_str)
            out_file.write("\n")
        out_file.close()
    in_file.close()

def build_feature_matrix(data, rate):
    '''
    Builds a feature matrix from a file's array representation, data.
    num_buffers denote the number of sub-arrays of data for which
    feature extraction will be done. Hence, the resulting matrix will be
    of shape(num_buffers, NUM_FEATURES)
    '''
    # Calculate number of buffers
    step_size = int(WINDOW_SIZE * STEP_SIZE_RATIO)
    data_length = len(data)
    remainder = data_length % step_size
    if remainder != 0:
        num_buffers = data_length / step_size - 1
    else:
        num_buffers = data_length / step_size

    feature_mtx = np.zeros((num_buffers, NUM_FEATURES))
    row_idx = 0
    prev_data = None

    for idx in range(0, len(data), step_size):
        buffer_data = data[idx:idx+WINDOW_SIZE]
        if len(buffer_data) < WINDOW_SIZE:
            # Case where last buffer is not window size
            continue
        filtered_data = preemphasis_filter(buffer_data)
        hammed_data = apply_hamming_window(filtered_data)
        transformed_data = transform_data(hammed_data)
        # Power spectrum of frame
        power_spectrum = periodogram_spectral_estimate(transformed_data)
        filter_banks, bins = mel_freq_filter(transformed_data, rate)
        filter_bank_energies = get_filter_bank_energies(power_spectrum, filter_banks, bins)

        # Take the log on the energies
        log_filter_bank_energies = np.log10(filter_bank_energies)

        # DCT
        dct = scipy.fftpack.dct(log_filter_bank_energies, norm='ortho')
        print "DCT"
        print dct
        print np.shape(dct)
        # Feature extraction methods
        sc = spectral_centroid(transformed_data)
        sro = spectral_rolloff(transformed_data)
        sfm = spectral_flatness_measure(transformed_data)
        parfft = peak_to_ave(transformed_data)
        sf = spectral_flux(prev_data, transformed_data)

        # Save current buffer for the next iteration (for spectral flux calculation)
        prev_data = transformed_data

        feature_mtx[row_idx] = [sc, sro, sfm, parfft, sf]
        row_idx += 1
    return feature_mtx

def preemphasis_filter(data):
    n = len(data)
    result = []
    for idx in range(n):
        if idx == 0:
            result.append(data[idx])
        else:
            y = data[idx] - 0.95 * data[idx-1]
            result.append(y)
    return result

def apply_hamming_window(data):
    '''
    Applies a hamming window of WINDOW_SIZE onto the data array.
    '''
    window = scipy.signal.hamming(WINDOW_SIZE)

    for idx in range(WINDOW_SIZE):
        data[idx] = window[idx] * data[idx]

    return data

def mel_freq_filter(data, rate):
    max_mel = freq_to_mel(rate/2.0)
    min_mel = freq_to_mel(0.0)
    step = (max_mel - min_mel) * 1.0 / (NUM_FILTER_BANKS + 1)
    # mel_values = f_range(min_mel, max_mel, step)
    mel_values = np.linspace(min_mel, max_mel, NUM_FILTER_BANKS + 2)
    # Convert the mel values to freqs
    freqs = [mel_to_freq(m) for m in mel_values]
    # Convert the freqs into integer FFT bins
    num_fft_bins = WINDOW_SIZE / 2 + 1
    bins = [math.floor((num_fft_bins+1) * (f/rate)) for f in freqs]

    filter_bank = np.zeros([NUM_FILTER_BANKS, num_fft_bins])

    for filter_bank_idx in range(0, NUM_FILTER_BANKS):
        # Left-side of the triangle
        for y in range(int(bins[filter_bank_idx]), int(bins[filter_bank_idx+1])):
            filter_bank[filter_bank_idx, y] = (y - bins[filter_bank_idx]) / (bins[filter_bank_idx+1] - bins[filter_bank_idx])
        # Right-side of the triangle
        for y in range(int(bins[filter_bank_idx+1]), int(bins[filter_bank_idx+2])):
            filter_bank[filter_bank_idx, y] = (bins[filter_bank_idx+2] - y) / (bins[filter_bank_idx+2] - bins[filter_bank_idx+1])

    return filter_bank, bins


def f_range(start, stop, step):
    '''
    Rolled my own range function that allows for step sizes with float values.
    But found out that numpy's linspace does the same thing.
    '''
    sum_val = 0
    values = []
    while sum_val < stop:
        values.append(sum_val)
        sum_val += step
    values.append(stop)
    print values
    return values

def freq_to_mel(freq):
    return 1127 * math.log(1 + freq/700.0)

def mel_to_freq(mel):
    return 700 * (math.exp(mel/1127.0) - 1)

def transform_data(data):
    '''
    Takes a single buffer's data and applies Discrete Fourier Transform.
    Only N/2 positive values are kept (index 0 to N/2+1, data[0] = 0)
    Returns the array of values after conversion to their absolute values
    '''
    # DFT, taking only positive elements (0:N/2+1)
    transformed_data = scipy.fftpack.fft(data)[:WINDOW_SIZE/2 + 1]
    # Convert all values to absolute
    return [np.abs(x) for x in transformed_data]

def periodogram_spectral_estimate(data):
    return [1.0 / WINDOW_SIZE * x**2 for x in data]

def get_filter_bank_energies(power_spectrum, filter_banks, bins):
    '''
    :param power_spectrum is a single frame's power_spectrum
    :param filter_banks is a single frame's filter_banks. Each row
        is a filter, and the columns are the y-axis (amplitude) values
    :param bins is a single frame's integer FFT bin values
    Returns an array of filter bank energies, one value for each filter
    '''
    filter_bank_energies = []
    for f_idx in range(0, NUM_FILTER_BANKS):
        # Energy sum for this filter bank
         energy_sum = 0
        # For every integer from this bin value to the value of 2 bins after
         for x_idx in range(int(bins[f_idx]), int(bins[f_idx+2])):
             energy_sum += power_spectrum[x_idx] * filter_banks[f_idx][x_idx]
         filter_bank_energies.append(energy_sum)

    return filter_bank_energies

def generate_stats(feature_mtx):
    '''
    Given a file's feature matrix, generate a stats array of shape(1, NUM_FEATURES*2)
    with the format [feature1_mean, feature2_mean, ..., featureN_mean, feature1_std, feature2_std, ..., featureN_std]
    '''
    mean_arr = []
    std_arr = []
    for col in range(NUM_FEATURES):
        feature_data = feature_mtx[:, col]
        mean_arr.append(mean(feature_data))
        std_arr.append(std(feature_data))

    file_stats = mean_arr + std_arr
    return file_stats

def spectral_centroid(data):
    n = len(data)
    sum_top = 0
    sum_bottom = 0
    for idx in range(n):
        sum_top += idx * data[idx]
        sum_bottom += data[idx]

    return sum_top / float(sum_bottom)

def spectral_rolloff(data):
    n = len(data)
    l_energy = 0
    for idx in range(n):
        l_energy += L_VALUE * data[idx]
    sum = 0
    sro = 0
    for idx in range(n):
        sum += np.abs(data[idx])
        if sum >= l_energy:
            sro = idx
            break
    return sro

def spectral_flatness_measure(data):
    n = len(data)
    top = np.exp(1.0/n * sum(np.log(data)))
    bottom = 1.0/n * sum(data)
    return top / float(bottom)

def spectral_flux(data_0, data_1):
    '''
    Returns the spectral flux of data_1 given both data_1 (the current buffer) and data_0 (the preceding buffer)
    '''
    n = len(data_1)
    if data_0 is None:
        data_0 = np.zeros(n)

    flux = 0
    for idx in range(n):
        diff = data_1[idx] - data_0[idx]
        if diff > 0:
            flux += diff
    return flux

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
