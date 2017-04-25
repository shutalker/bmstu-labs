import wfdb
from scipy import signal
import matplotlib.pyplot as plt


def get_signal_from_channel(signal_array, signal_idx=0):

    '''
        signal_array is an array of n-length arrays, where n
        is amount of channels in record
        each of n-length array contains samples values from each
        channel
    '''

    sig = list()
    for sample_list in signal_array:
        sig.append(sample_list[signal_idx])

    return sig


def butter_bandpass(cutoff, sample_freq, filter_type, order=1):
    nyq_freq = 0.5 * sample_freq
    cut_freq = cutoff / nyq_freq
    b, a = signal.butter(order, cut_freq, btype=filter_type)

    return b, a


def five_point_derivative(sig, sample_freq):
    derivative = list()
    sample_period = 1 / sample_freq
    sig_length = len(sig)

    derivative_value = (1 / 8) * sample_period * (2 * sig[1] + sig[2])
    derivative.append(derivative_value)    
    derivative_value = (1 / 8) * sample_period * (-2 * sig[0] + 2 * sig[2] + sig[3])
    derivative.append(derivative_value)

    for sample_idx in range(2, (sig_length - 2)):
        first_point_value = sig[sample_idx - 2]
        second_point_value = sig[sample_idx - 1]
        third_point_value = sig[sample_idx + 1]
        fourth_point_value = sig[sample_idx + 2]
        derivative_value = -first_point_value - 2 * second_point_value + 2 * third_point_value + fourth_point_value
        derivative_value *= (1 / 8) * sample_period
        derivative.append(derivative_value)

    derivative_value = (1 / 8) * sample_period * (-sig[sig_length-4] - 2 * sig[sig_length-3] + 2 * sig[sig_length-1])
    derivative.append(derivative_value)    
    derivative_value = (1 / 8) * sample_period * (-sig[sig_length-3] - 2 * sig[sig_length-2])
    derivative.append(derivative_value)

    return derivative


def amplitude_squaring(sig):
    for sample_idx in range(len(sig)):
        sig[sample_idx] = sig[sample_idx] ** 2


if __name__ == '__main__':
    samples_amount = 1500
    sample_freq = 360
    lowcut = 5
    highcut = 15
    window_width = 41

    rec = wfdb.rdsamp('231', sampto=samples_amount, channels=[0], physical=False, pbdir='mitdb')
    filtering_signal = get_signal_from_channel(rec.d_signals, 0)
    plt.subplot(311)
    plt.plot(filtering_signal, 'r')
    plt.ylabel('adus')

    # Butterworth 1-order lowpass filter
    b, a = butter_bandpass(lowcut, sample_freq, 'lowpass')
    # initial condition for lfilter
    zi = signal.lfilter_zi(b, a)
    filtering_signal, _ = signal.lfilter(b, a, filtering_signal, zi=zi*filtering_signal[0])

    # Butterworth 1-order highpass filter
    b, a = butter_bandpass(highcut, sample_freq, 'highpass')
    zi = signal.lfilter_zi(b, a)
    filtering_signal, _ = signal.lfilter(b, a, filtering_signal, zi=zi*filtering_signal[0])

    derivative_vec = five_point_derivative(filtering_signal, sample_freq)
    plt.subplot(312)
    plt.plot(derivative_vec, 'g')
    plt.ylabel('derivative')

    amplitude_squaring(filtering_signal)
    filtering_signal = signal.medfilt(filtering_signal, window_width)

    # plot resulting signal
    plt.subplot(313)
    plt.plot(filtering_signal, 'b')
    plt.ylabel('adus')
    plt.xlabel('filtered signal')
    plt.savefig('sig.png', format='png')
