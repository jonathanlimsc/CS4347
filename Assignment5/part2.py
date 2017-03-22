import math
import numpy as np

SAMPLING_RATE = 44100.0
DURATION = 1.0

def generateLookUpTable(num_samples):
    time = np.linspace(0, 1, num_samples, endpoint=False)
    y = np.sin(2 * np.pi * 1.0 * time)
    return y

def generateLookUpIndex(freq, num_samples):
    index_table = [0]
    hop = num_samples * freq / SAMPLING_RATE
    index = 1
    while len(index_table) < int(SAMPLING_RATE):
        index_table.append((index_table[index-1] + hop) % num_samples)
        index += 1

    # print "Index table", index_table
    # print "Size of index table", np.shape(index_table)
    return index_table

def synthesizeSineWave(lookup_table, lookup_index, interpolation=False):
    sine_wave = []
    index = 0
    while len(sine_wave) < int(SAMPLING_RATE):
        if interpolation:
            lookup_val = lookup_index[index]
            frac = lookup_val - np.floor(lookup_val)
            floor = int(np.floor(lookup_val))
            # Prevent overflow
            ceil = int(np.ceil(lookup_val)) % len(lookup_table)
            result = lookup_table[floor] + (lookup_table[ceil] - lookup_table[floor]) * frac
        else:
            lookup_val = round(lookup_index[index]) % len(lookup_table)
            result = lookup_table[int(lookup_val)]
        sine_wave.append(result)
        index += 1

    return sine_wave

def generateSineWave(freq):
    time = np.arange(1.0 * SAMPLING_RATE)
    return np.sin(2 * np.pi * freq / SAMPLING_RATE * time)

def calculateError(perfect_wave, synth_wave):
    max_error = np.max(np.abs(synth_wave - perfect_wave))
    max_audio_file_error = 32767 * max_error
    # print "Max error", max_error
    # print "Max audio file error", max_audio_file_error
    return max_audio_file_error

def generateResultsFromFreq(freq_arr, sampling_num_arr):
    results = {}
    for sampling_num in sampling_num_arr:
        # No interpolation
        print "Sampling number", sampling_num
        table = generateLookUpTable(sampling_num)

        for freq in freq_arr:
            index = generateLookUpIndex(freq, sampling_num)
            synth = synthesizeSineWave(table, index)
            perfect = generateSineWave(freq)
            error = calculateError(perfect, synth)
            print "Freq", freq
            print "Not interpolated Error: ", str(error)
            if freq in results.keys():
                if 'No' in results[freq].keys():
                    results[freq]['No'][sampling_num] = error
                else:
                    results[freq]['No'] = {sampling_num: error}
            else:
                results[freq] = {'No': {sampling_num: error }}


            synth = synthesizeSineWave(table, index, True)
            perfect = generateSineWave(freq)
            error = calculateError(perfect, synth)
            print "Interpolated Error: ", str(error)
            if freq in results.keys():
                if 'Linear' in results[freq].keys():
                    results[freq]['Linear'][sampling_num] = error
                else:
                    results[freq]['Linear'] = {sampling_num: error}
            else:
                results[freq] = {'Linear': {sampling_num: error }}
            print ""
        print ""
        print results
    return results

def printToFile(frequency_arr, sampling_num_arr, result_hash):
    with open('output.txt', 'w') as output_file:
        header_str = "Frequency\t\t\tInterpolation\t\t\t"
        for sampling_num in sampling_num_arr:
            header_str += str(sampling_num) + "-sample\t\t\t"
        output_file.write(header_str + "\n")

        output_str = "100Hz \t\t\t\t No \t\t\t\t\t\t\t %.6f \t\t\t\t %.6f \n" % (result_hash[100]['No'][16384], result_hash[100]['No'][2048])
        output_file.write(output_str)
        output_str = "\t\t\t\t\t\t\t Linear \t\t\t\t\t %.6f \t\t\t\t %.6f \n" % (result_hash[100]['Linear'][16384], result_hash[100]['Linear'][2048])
        output_file.write(output_str)

        output_str = "1234.56Hz \t\t No \t\t\t\t\t\t\t %.6f \t\t\t\t %.6f \n" % (result_hash[1234.56]['No'][16384], result_hash[1234.56]['No'][2048])
        output_file.write(output_str)
        output_str = "\t\t\t\t\t\t\t Linear \t\t\t\t\t %.6f \t\t\t\t %.6f \n" % (result_hash[1234.56]['Linear'][16384], result_hash[1234.56]['Linear'][2048])
        output_file.write(output_str)

result = generateResultsFromFreq([100, 1234.56], [16384, 2048])
printToFile([100,1234.56], [16384, 2048], result)
