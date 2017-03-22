import scipy.io.wavfile as wav
import math
import numpy as np
# import scipy.signal.spectrogram
import matplotlib.pyplot as plt

SAMPLING_RATE = 8000.0
DURATION = 0.25
NUM_SAMPLING_BITS = 16
STEP_SIZE_RATIO = 0.5
WINDOW_SIZE = 512
OUTPUT_FILE = 'output_adsr.wav'

def generateNote(freq, step=4, ADSR=False):
    wave = None
    denom = iterativeSum(step)
    for idx in range(0, step):
        if wave is None:
            wave = (step-idx)/float(denom) * generateSinWave(freq * (idx+1))
        else:
            wave += (step-idx)/float(denom) * generateSinWave(freq * (idx+1))
    if ADSR:
        wave = wave * generateADSREnvelope(len(wave))
    return wave

def iterativeSum(num):
    nextNum = num
    result = 0
    while nextNum > 0:
        result += nextNum
        nextNum -= 1
    return result

def generateWav(data):
    # Scale the amplitude according to the NUM_SAMPLING_BITS, taking into account signed int
    maxAmplitude = math.pow(2, NUM_SAMPLING_BITS) / 2
    dataType = 'int' + str(NUM_SAMPLING_BITS)
    data = np.array(data * maxAmplitude, dtype=dataType)
    # print data
    with open(OUTPUT_FILE, 'w') as outputFile:
        wav.write(outputFile, int(SAMPLING_RATE), data)
    outputFile.close()

def generateSinWave(freq):
    sample_range = np.linspace(0, DURATION, SAMPLING_RATE * DURATION)
    y = np.sin(2 * np.pi * freq * sample_range)
    return y

def midiToFundamentalFreq(midi):
    if midi == 0:
        return 0
    else:
        exponent = (midi - 69) / 12.0
    return 440 * math.pow(2, exponent)

def generateSpectrogram():
    rate, data = wav.read(OUTPUT_FILE)
    conversionConst = math.pow(2, NUM_SAMPLING_BITS)/2.0
    data  = data / conversionConst;
    stepSize = int(WINDOW_SIZE * STEP_SIZE_RATIO)
    dataLength = len(data)
    remainder = dataLength % stepSize
    if remainder != 0:
        numBuffers = dataLength / stepSize - 1
    else:
        numBuffers = dataLength / stepSize
    fftSize = WINDOW_SIZE/2 +1
    buffers = np.zeros((numBuffers, fftSize))

    for idx in range(numBuffers):
        start = idx * stepSize
        end = start+WINDOW_SIZE
        bufferData = data[start:end]
        if len(bufferData) < WINDOW_SIZE:
            continue

        windowed_data = np.blackman(WINDOW_SIZE) * bufferData
        spectrum = getPowerSpectrum(windowed_data)
        buffers[idx] = spectrum

    buffers = buffers.transpose()
    im = plt.imshow(buffers, origin='lower')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [Buffer Number]')
    plt.show()

def generateADSREnvelope(bufferLength):
    x_values = np.arange(bufferLength)
    envelop = np.zeros(bufferLength)
    x_weights = [0.0, 0.1, 0.3, 0.7, 1.0]
    y_weights = [0.0, 1.0, 0.6, 0.6, 0.0]
    scaled_weights = [int(x * bufferLength) for x in x_weights]
    for idx in range(len(scaled_weights)-1):
        x_left = scaled_weights[idx]
        x_right = scaled_weights[idx+1]
        y_left = y_weights[idx]
        y_right = y_weights[idx+1]
        gradient = getGradient(y_left, y_right, x_left, x_right)
        for x in range(x_left, x_right):
            envelop[x] = y_left + (x - x_left) * gradient

    # plt.plot(x_values, envelop)
    # plt.ylabel('Amplitude')
    # plt.xlabel('Sample number')
    # plt.show()
    return envelop

def getGradient(y1, y2, x1, x2):
    return (y2-y1) / float((x2-x1))

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

notes = {'c': 60, 'd':62, 'e':64, 'f': 65, 'g': 67, 'a': 69, 'b': 71, 'c2': 72, 'd2': 74, 'e2': 76}
# Fur elise
# midis =  [notes['e2'], notes['e2']-1, notes['e2'], notes['e2']-1, notes['e2'], notes['b'], notes['d2'], notes['c2'], notes['a'], 0, 0, notes['c'], notes['e'], notes['a'], notes['b'], 0, 0, notes['e'], notes['a'], notes['b'], notes['c2']]
midis = [60, 62, 64, 65, 67, 69, 71, 72, 72, 0, 67, 0, 64, 0, 60]
data = None
for midi in midis:
    freq = midiToFundamentalFreq(midi)
    note = generateNote(freq, ADSR=True)
    if data is None:
        data = note
    else:
        data = np.concatenate((data, note), axis=0)

generateWav(data)
generateSpectrogram()
