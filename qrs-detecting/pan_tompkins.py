import wfdb
from scipy import signal
import numpy as np
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

def peak_detection(amplitude, spk, npk, threshold, threshold_type):

    peak_detector = 0
    recalculated_thresholds = {}

    if amplitude >= threshold:
        peak_detector += 1
        if threshold_type == 'low_threshold':
            spk = 0.25 * amplitude + 0.75 * spk
        else:
            spk = 0.125 * amplitude + 0.875 * spk
    else:
        npk = 0.125 * amplitude + 0.875 * npk

    h_threshold = npk + 0.25 * (spk - npk)
    l_threshold = 0.5 * h_threshold

    recalculated_thresholds['h_threshold'] = h_threshold
    recalculated_thresholds['l_threshold'] = l_threshold
    recalculated_thresholds['spk'] = spk
    recalculated_thresholds['npk'] = npk
    recalculated_thresholds['peak_detector'] = peak_detector

    return recalculated_thresholds

def detect_T_wawe(derivative, sample, last_peek_idx):

    # sample offset for slope detecting
    slope_offset = 5

    T_wawe_idx = [-1, -1]

    if last_peek_idx:
        try:
            current_R_slope = derivative[sample - slope_offset]
            last_R_slope = derivative[last_peek_idx - slope_offset]
        except IndexError:
            last_R_slope = derivative[0]

        if last_R_slope >= (2 * current_R_slope):
            T_wawe_idx[1] = current_R_slope
        elif current_R_slope >= (2 * last_R_slope):
            T_wawe_idx[0] = last_R_slope

    return T_wawe_idx




def adaptive_tresholds_algorithm(clean, integrated, derivative, 
                                 search_width, peakidxs):

    # initial estimates for filtered (clean) signal (in mV)
    # Signal PeaK for Filtering waveform 
    spkf = max(clean)

    # Noise PeaK for Filtering waveform
    npkf = spkf / 2

    # adaptive thresholds (higher, lower)
    h_threshold_f = npkf + 0.25 * (spkf - npkf)
    l_threshold_f = 0.5 * h_threshold_f

    # for Integration waveform
    spki = max(integrated)
    npki = spki / 3

    h_threshold_i = npki + 0.25 * (spki - npki)
    l_threshold_i = 0.5 * h_threshold_i

    signal_duration = len(clean)
    frame_amount = int(np.ceil(signal_duration / search_width))
    start_frame_idx = 0
    end_frame_idx = search_width
    frame_delta = search_width * frame_amount - signal_duration

    # returning value
    pulse_signal = [0] * signal_duration
    last_peek_idx = None

    for frame_idx in range(frame_amount):
        peak_counter = 0

        if frame_idx == frame_amount - 1:
            end_frame_idx = search_width - frame_delta

        # clean_signal_frame = clean[start_frame_idx : end_frame_idx]
        # integrated_signal_frame = integrated[start_frame_idx : end_frame_idx]

        # forward peak searching
        for sample in range(start_frame_idx, end_frame_idx):
            peak_detector = 0
            f_amplitude = clean[sample]
            i_amplitude = integrated[sample]

            adapted_thresholds = peak_detection(f_amplitude, spkf, npkf,
                                                h_threshold_f, 'high_threshold')
            h_threshold_f = adapted_thresholds['h_threshold']
            l_threshold_f = adapted_thresholds['l_threshold']
            spkf = adapted_thresholds['spk']
            npkf = adapted_thresholds['npk']
            peak_detector += adapted_thresholds['peak_detector']

            adapted_thresholds = peak_detection(i_amplitude, spki, npki,
                                                h_threshold_i, 'high_threshold')
            h_threshold_i = adapted_thresholds['h_threshold']
            l_threshold_i = adapted_thresholds['l_threshold']
            spki = adapted_thresholds['spk']
            npki = adapted_thresholds['npk']
            peak_detector += adapted_thresholds['peak_detector']

            if peak_detector > 1:
                if sample in peakidxs:
                    T_wawes = detect_T_wawe(derivative, sample,
                                                last_peek_idx)
                    if T_wawes[0] != -1:
                        pulse_signal[T_wawes[0]] = 0

                    if T_wawes[1] == -1:
                        pulse_signal[sample] = 1
                        last_peek_idx = sample
                        peak_counter += 1

        # peek backsearch
        if peak_counter == 0:
            for sample in range(end_frame_idx, start_frame_idx):
                peak_detector = 0
                f_amplitude = clean[sample]
                i_amplitude = integrated[sample]

                adapted_thresholds = peak_detection(f_amplitude, spkf, npkf,
                                                    l_threshold_f, 'low_threshold')
                h_threshold_f = adapted_thresholds['h_threshold']
                l_threshold_f = adapted_thresholds['l_threshold']
                spkf = adapted_thresholds['spk']
                npkf = adapted_thresholds['npk']
                peak_detector += adapted_thresholds['peak_detector']

                adapted_thresholds = peak_detection(i_amplitude, spki, npki,
                                                    l_threshold_i, 'low_threshold')
                h_threshold_i = adapted_thresholds['h_threshold']
                l_threshold_i = adapted_thresholds['l_threshold']
                spki = adapted_thresholds['spk']
                npki = adapted_thresholds['npk']
                peak_detector += adapted_thresholds['peak_detector']

                if peak_detector > 1:
                    if sample in peakidxs:
                        T_wawes = detect_T_wawe(derivative, sample,
                                                last_peek_idx)
                        if T_wawes[0] != -1:
                            pulse_signal[T_wawes[0]] = 0

                        if T_wawes[1] == -1:
                            pulse_signal[sample] = 1
                            last_peek_idx = sample
                            peak_counter += 1

        start_frame_idx = end_frame_idx
        end_frame_idx += search_width

    return pulse_signal

if __name__ == '__main__':
    samples_amount = 1500
    lowcut = 5
    highcut = 15
    window_width = 41

    # configuring plot and subplots
    fig = plt.figure(figsize=(10.24, 7.68))
    plt.subplots_adjust(hspace=0.6)

    rec = wfdb.rdsamp('231', sampfrom=samples_amount*2,
                      sampto=samples_amount*3, channels=[0],
                      physical=True, pbdir='mitdb')
    sample_freq = rec.fs
    filtering_signal = get_signal_from_channel(rec.p_signals, 0)
    plt.subplot(411)
    plt.plot(filtering_signal, 'r')
    plt.ylabel('mV')
    plt.xlabel('clean signal')

    # Butterworth 1-order lowpass filter
    b, a = butter_bandpass(lowcut, sample_freq, 'lowpass')

    # initial condition for lfilter
    zi = signal.lfilter_zi(b, a)
    filtering_signal, _ = signal.lfilter(b, a, filtering_signal,
                                         zi=zi*filtering_signal[0])

    # Butterworth 1-order highpass filter
    b, a = butter_bandpass(highcut, sample_freq, 'highpass')
    zi = signal.lfilter_zi(b, a)
    clean_signal, _ = signal.lfilter(b, a, filtering_signal,
                                     zi=zi*filtering_signal[0])

    derivative_signal = five_point_derivative(clean_signal, sample_freq)
    plt.subplot(412)
    plt.plot(derivative_signal, 'g')
    plt.xlabel('derivative')

    amplitude_squaring(clean_signal)
    integrated_signal = signal.medfilt(clean_signal, window_width)

    # finding peaks for desicion rule algorithm in filtered signal
    peakidxs = signal.find_peaks_cwt(np.array(clean_signal), np.arange(1, 200))

    # signal indicates locations of R-peaks
    pulse_signal = adaptive_tresholds_algorithm(clean_signal,
                                                integrated_signal,
                                                derivative_signal,
                                                window_width,
                                                peakidxs)
    #print(pulse_signal)
    plt.subplot(413)
    plt.plot(integrated_signal, 'b')
    plt.ylabel('adus')
    plt.xlabel('filtered_signal')

    # plot resulting signal
    plt.subplot(414)
    plt.plot(pulse_signal)
    plt.ylabel('pulse')
    plt.xlabel('pulse signal with R-peak locations')
    plt.tight_layout()
    plt.savefig('sig.png', dpi=200, format='png')