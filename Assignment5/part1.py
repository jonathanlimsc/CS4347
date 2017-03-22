import scipy.io.wavfile as wav
import math
import numpy as np
# import scipy.signal.spectrogram
import matplotlib.pyplot as plt
from scipy import signal

SAMPLING_RATE = 44100.0
DURATION = 1.0
NUM_SAMPLING_BITS = 16
STEP_SIZE_RATIO = 0.5
WINDOW_SIZE = 512


def generateWav(data, output_file):
    # Scale the amplitude according to the NUM_SAMPLING_BITS, taking into account signed int
    maxAmplitude = math.pow(2, NUM_SAMPLING_BITS) / 2
    dataType = 'int' + str(NUM_SAMPLING_BITS)
    data = np.array(data * maxAmplitude, dtype=dataType)
    # print data
    with open(output_file, 'w') as outputFile:
        wav.write(outputFile, int(SAMPLING_RATE), data)
    outputFile.close()

def generateSinWave(freq):
    sample_range = np.linspace(0, DURATION, SAMPLING_RATE * DURATION)
    y = np.sin(2 * np.pi * freq * sample_range)
    return y

def generatePerfectSawtoothWave(freq, duration=1.0, amplitude=1.0):
    time = np.arange(int(SAMPLING_RATE * duration))
    y = amplitude * signal.sawtooth(2 * np.pi * freq / SAMPLING_RATE * time )
    # plt.plot(time, y)
    # plt.show()
    return y

def generateAdditiveSawtoothWave(freq, duration=1.0, amplitude=1.0):
    nyquist_limit = SAMPLING_RATE * 0.5
    harmonic = 1.0
    wave_freq = freq
    wave_freqs = []
    while wave_freq <= nyquist_limit:
        wave_freqs.append(wave_freq)
        harmonic += 1
        wave_freq = harmonic * freq
    max_num_waves = len(wave_freqs)
    print "Wave freqs: ", wave_freqs
    print "Max num waves: ", max_num_waves

    time = np.arange(int(duration * SAMPLING_RATE))
    sum_of_sines = None
    for k in np.arange(1, max_num_waves+1):
        if sum_of_sines is None:
            sum_of_sines = 1.0 / k * np.sin( k * 2 * np.pi * freq / SAMPLING_RATE * time )
        else:
            sum_of_sines += 1.0 / k * np.sin( k * 2 * np.pi * freq / SAMPLING_RATE * time )
    # print "Sum of sines", sum_of_sines
    result = -2 * amplitude / np.pi * sum_of_sines
    return result
    # plt.plot(time, result)
    # plt.show()

def getPowerSpectrum(data):
    # perform fft
    signal = np.fft.fft(data)
    fftData = signal[:len(data)/2+1]

    # magnitude spectrum and normalise
    magfft = np.abs(fftData)/len(data)

    # Log spectrum
    epsilon = np.power(10.0, -10)
    spectrum = 20 * np.log10(magfft + epsilon)

    return spectrum


time = 1.0/1000 * 6
x = np.linspace(0, 22050, 8192/2+1)
print "x shape: ", np.shape(x)
y1 = generateAdditiveSawtoothWave(1000, amplitude=0.5)
print "Additive shape: ", np.shape(y1)
y2 = generatePerfectSawtoothWave(1000, amplitude=0.5)
print "Perfect shape: ", np.shape(y2)


# To generate wav files
# generateWav(y1, 'generated.wav')
# generateWav(y2, 'perfect.wav')

# To plot db-magnitude graph
# data1 = y1[0:8192]
# power1 = getPowerSpectrum(data1)
# print "Power 1 ", np.shape(power1)
# data2 = y2[0:8192]
# power2 = getPowerSpectrum(data2)
# print "Power 2 ", np.shape(power2)
# x = x[0:8192/2+1]
# print "x ", np.shape(x)
# plt.plot(x, power1, label='Generated Sawtooth Wave', linewidth=1)
# plt.plot(x, power2, label='Perfect Sawtooth Wave', linewidth=1)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('dB-Magnitude')
# plt.legend(loc='upper right')
# plt.show()
