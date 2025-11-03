import ast

import numpy as np

import wave

from scipy import signal

import torch.utils.data as data

from scipy.fft import rfft, fft, fftfreq
from scipy import fftpack

from pathlib import Path

from tqdm import tqdm

from scipy.io import wavfile

import os
import torch
import torchaudio
import scipy.signal as ssg
from sklearn.preprocessing import scale

AUDIO_EXTENSIONS = ['.mp3', '.wav']


def target_ref_pair(target_folder, ref_folder):
    ref_files = [Path(f).stem for f in os.listdir(ref_folder) if f.endswith('.wav')]
    target_files = [Path(f).stem for f in os.listdir(target_folder) if f.endswith('.wav')]

    target_ref_pair_dict = {}
    for target_file in target_files:
        try:
            ref_file = [f for f in ref_files if target_file in f].pop()
        except IndexError as e:
            print(f'Error: {e} -- target file {target_file} is not in ref files.')
            continue

        target_ref_pair_dict.update({target_file: ref_file})

    # _target_ref_pair = [(str(k).zfill(3) + '.wav', target_ref_pair_dict[k] + '.wav')
    #                     for k in sorted(target_ref_pair_dict, key=lambda tt: (int(tt), tt))]

    _target_ref_pair = [(k + '.wav', target_ref_pair_dict[k] + '.wav') for k in sorted(target_ref_pair_dict)]

    return _target_ref_pair


def target_ref_one_day_pair(target_folder, ref_one_day_folder):
    if ref_one_day_folder is None:
        return None

    target_files = [Path(f).stem for f in os.listdir(target_folder) if f.endswith('.wav')]
    ref_one_day_files = [Path(f).stem for f in os.listdir(ref_one_day_folder) if f.endswith('.wav')]

    target_ref_one_day_pair_dict = {}
    for target_file in target_files:
        for f in ref_one_day_files:
            try:
                x, y, _ = f.replace('_', '-').split('-')
                x, y = int(x), int(y)
                if x <= int(target_file) <= y:
                    target_ref_one_day_pair_dict.update({target_file: f})
            except ValueError:
                x, _ = f.replace('_', '-').split('-')
                x = int(x)
                if x == int(target_file):
                    target_ref_one_day_pair_dict.update({target_file: f})

    _target_ref_one_day_pair = [(str(k).zfill(3) + '.wav', target_ref_one_day_pair_dict[k] + '.wav')
                                for k in sorted(target_ref_one_day_pair_dict, key=lambda tt: (int(tt), tt))]

    # target_ref_one_day_pair_train, target_ref_one_day_pair_test, = train_test_split(target_ref_one_day_pair,
    #                                                                                 test_size=test_size,
    #                                                                                 random_state=42)
    #
    # target_ref_one_day_pair_tr_tst = target_ref_one_day_pair_train, target_ref_one_day_pair_test
    return _target_ref_one_day_pair


def read_wav_pairs(target_folder, ref_folder, ref_one_day_folder=None):
    _target_ref_pair = target_ref_pair(target_folder, ref_folder)

    _target_ref_one_day_pair = target_ref_one_day_pair(target_folder, ref_one_day_folder)

    return _target_ref_pair, _target_ref_one_day_pair


def load_wav_file(sound_file_path, sample_rate=None):
    _sample_rate, sound_np = wavfile.read(sound_file_path)
    if sample_rate is not None and sample_rate != _sample_rate:
        raise Exception(
            "Unexpected sample rate {} (expected {})".format(_sample_rate, sample_rate)
        )

    if sound_np.dtype != np.float32:
        assert sound_np.dtype == np.int16
        sound_np = sound_np / 32767  # ends up roughly between -1 and 1

    return sound_np, _sample_rate


def load_wav(fpath, verbose=False):
    """
    Loads a .wav file and returns the data and sample rate.

    :param fpath: the path to load the file from
    :param verbose: to print the loaded .wav and the fs
    :returns: a tuple of (wav file data as a list of amplitudes, sample rate)
    """
    with wave.open(fpath) as wav_f:
        wav_buf = wav_f.readframes(wav_f.getnframes())
        data = np.frombuffer(wav_buf, dtype=np.int16)
        synthetic_fs = wav_f.getframerate()

        if verbose:
            clip_len_s = len(data) / synthetic_fs
            print(f"Loaded .wav file, n_samples={len(data)} len_s={clip_len_s}")

        return data.astype(np.float32), synthetic_fs


class AudioFolder(torch.utils.data.Dataset):
    def __init__(self, ref_frames, trg_frames):
        self.trg_frames = trg_frames
        self.ref_frames = ref_frames

        self.reference_frame_size, self.reference_frame_dim = ref_frames.shape
        self.target_signal_size, self.target_signal_dim = trg_frames.shape

    def __getitem__(self, index):
        return self.trg_frames[index], self.ref_frames[index]

    def __len__(self):
        return self.target_signal_size

    # def get_dims(self):
    #     return {'reference_spectrum_size': self.reference_spectrum_size, 'target_signal_size': self.target_signal_size}


def mean_std(loader):
    sum, squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        sum += torch.mean(data, dim=[0, 2, 3])
        squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = sum / num_batches
    std = (squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
import librosa

from scipy.signal import kaiserord, firwin, filtfilt, lfilter


def bandpass_firwin(n_taps, cut_off, fs, transition_width_hz=5.,
                    ripple_db=25, window='hamming'):
    if window == 'hamming' or window == 'blackman':
        taps = firwin(n_taps, cutoff=cut_off, fs=fs, pass_zero=False,
                      window=window, scale=False)
    elif window == 'kaiser':
        taps = bandpass_kaiser(cut_off, fs, transition_width_hz, ripple_db)
    else:
        raise ValueError('window should be hamming or kaiser')

    return taps


def bandpass_kaiser(cut_off, fs, transition_width_hz=5.0, ripple_db=25.0):
    # Nyquist rate.
    nyq_rate = fs / 2

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = transition_width_hz / nyq_rate

    n_taps, beta = kaiserord(ripple_db, width=width)

    if n_taps % 2 == 0:
        n_taps = n_taps + 1

    # Estimate the filter coefficients.
    cut_off_nyq = [cof / nyq_rate for cof in cut_off]

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    # See example why we need: pass_zero=False or 'bandpass'
    taps = firwin(n_taps, cut_off_nyq,
                  window=('kaiser', beta),
                  pass_zero=False)

    return taps


import matplotlib.pyplot as plt


def get_h0_frames(args,
                  window_size_seconds,
                  dataset_dir,
                  nfft_scale=1,
                  h0_n_samples=None,
                  fft_frames=False):
    l2_transformer = Normalizer()
    scaler = MinMaxScaler()
    # https://stackoverflow.com/questions/76143317/python-how-to-filtering-out-multiple-bands-of-frequencies-either-in-frequency
    h0_files = [Path(f).stem for f in os.listdir(os.path.join(dataset_dir, 'H0')) if f.endswith('.wav')]
    h0_frames, h0_frames_id = [], []

    _h0_nperseg = args.target_fs * window_size_seconds

    h0_phi = fftpack.dct(np.eye(_h0_nperseg), n=_h0_nperseg, type=3, norm='ortho')

    # rng = check_random_state(np.random.RandomState(0))
    # h0_phi = unitary_projection(rng.randn(_h0_nperseg, _h0_nperseg))

    for h0_file_name in tqdm(h0_files[:h0_n_samples], desc='Loading H0 files'):
        h0_data, h0_fs = torchaudio.load(os.path.join(dataset_dir, 'H0', f"{h0_file_name}.wav"))

        h0_data = np.squeeze(h0_data.numpy())

        h0_data = h0_data.astype(np.float32)

        # h0_data /= np.max(np.abs(h0_data))
        # h0_data -= np.mean(h0_data)
        # h0_data = (h0_data - np.mean(h0_data)) / np.std(h0_data)

        h0_data = ssg.resample_poly(h0_data, args.target_fs, 8000)
        if args.bandpass:
            h0_data = bandpass_filter(h0_data, enf_harmonic_n=2,
                                      signal_fs=args.target_fs,
                                      freq_band_size=args.freq_band_size,
                                      nominal_freq=50)
        elif args.multi_bandpass:

            window = 'kaiser'
            nominal_frequencies = [50., 100., 150.]
            cut_off = []
            for nominal_freq in nominal_frequencies:
                cut_off.extend([nominal_freq - args.freq_band_size, nominal_freq + args.freq_band_size])

            h0_taps_kaiser = bandpass_firwin(n_taps=128,
                                             cut_off=cut_off,
                                             fs=args.target_fs,
                                             transition_width_hz=args.transition_width_hz,
                                             ripple_db=args.ripple_db,
                                             window=window)

            b, a = h0_taps_kaiser, [1]
            h0_data = filtfilt(b=b, a=1, axis=0, x=h0_data,
                               padtype='odd',
                               padlen=3 * (max(len(b), len(a)) - 1))

            # plt_power_spectrum(h0_data, h0_data_filtered, down_sample_fr)
        # else:
        #     print('No filtering is applied in H0.')

        _h0_frames, _h0_n_frames, _h0_nperseg, _ = get_wav_frames(h0_data, args.target_fs, window='hann',
                                                                  window_size_seconds=window_size_seconds)
        h0_frames_id.extend(_h0_n_frames * [f"{h0_file_name}.wav"])
        # t = np.linspace(0, np.pi, num=_h0_nperseg).reshape(_h0_nperseg, 1)
        # w_h0 = np.sin(t)
        if fft_frames:
            _rfft = True
            if _rfft:
                nfft = next_power_of_2(nfft_scale * _h0_nperseg)
                # frames are on columns
                # _h0_frames = scale(_h0_frames, axis=0)

                zxx_h0 = np.abs(rfft(_h0_frames, n=nfft, axis=0) / np.sqrt(_h0_nperseg) / 2)
                # zxx_h0 = np.abs(rfft(_h0_frames, n=nfft, axis=0)) #/ nfft

                # zxx_h0 = np.log1p(zxx_h0**2)

                # h0_spectrum = rfft(_h0_frames, n=nfft, axis=0) / nfft
                # zxx_h0 = np.abs(h0_spectrum * np.conj(h0_spectrum))
                # zxx_h0 = 20 * np.log1p(zxx_h0)  # scale to db
                # zxx_h0 = np.clip(zxx_h0, -40, 200)
            else:
                # _h0_frames = w_h0 * _h0_frames  # windowing
                # _h0_frames = l2_transformer.fit_transform(_h0_frames.T).T
                # _h0_frames = scale(_h0_frames, axis=0)
                zxx_h0 = np.abs(np.dot(h0_phi.T, _h0_frames)) ** 2

            # zxx_h0 = scaler.fit_transform(zxx_h0)
            # zxx_h0 /= np.max(zxx_h0, axis=0)

            # zxx_h0 = l2_transformer.fit_transform(zxx_h0.T).T

            h0_frames.append(zxx_h0)
        else:
            # _h0_frames[_h0_frames < 1e-5] = 0
            # _h0_data = scaler.fit_transform(_h0_frames)

            # NOISE_FLOOR = 1e-6
            # _h0_frames = -np.vectorize(convert_to_decibel)(_h0_frames + NOISE_FLOOR)

            # _h0_frames = (1 / 2) * (np.arcsin(_h0_frames) + np.pi / 2)

            # _h0_data = l2_transformer.fit_transform(_h0_frames.T).T
            # _h0_frames = scale(_h0_frames, axis=0)
            # _h0_frames /= np.max(np.abs(_h0_frames), axis=0)

            h0_frames.append(_h0_frames)

    h0_frames = np.hstack(h0_frames).T

    return h0_frames, h0_frames_id


def get_h1_frames(args,
                  window_size_seconds=16,
                  tr_n_samples=None,
                  val_n_samples=None,
                  tst_n_samples=None,
                  dataset_dir=None,
                  nfft_scale=1,
                  fft_frames=False,
                  target_folder='H1',
                  ref_folder='H1_ref'):
    tr_path = os.path.join(dataset_dir, f'folds{args.n_folds}', f"tr_data_pair_fold_{args.fold}.txt")
    print(f'Load tr: {tr_path}')
    with open(tr_path, 'r') as f:
        tr_data_pair = ast.literal_eval(f.read())
    print(f'Load tr: {len(tr_data_pair)} wav file pairs')

    val_path = os.path.join(dataset_dir, f'folds{args.n_folds}', f"val_data_pair_fold_{args.fold}.txt")
    with open(val_path, 'r') as f:
        val_data_pair = ast.literal_eval(f.read())
    print(f'Load val: {len(val_data_pair)} wav file pairs')

    tst_path = os.path.join(dataset_dir, f'folds{args.n_folds}', f"tst_data_pair_fold_{args.fold}.txt")
    with open(tst_path, 'r') as f:
        tst_data_pair = ast.literal_eval(f.read())
    print(f'Load tst: {len(tst_data_pair)} wav file pairs')

    # Get train frames
    tr_frames, tr_frames_ids = wav_frames(args, tr_data_pair,
                                          dataset_dir=dataset_dir,
                                          target_folder=target_folder,
                                          ref_folder=ref_folder,
                                          window_size_seconds=window_size_seconds,
                                          trg_fs=args.target_fs,
                                          n_samples=tr_n_samples,
                                          freq_band_size=args.freq_band_size,
                                          nfft_scale=nfft_scale,
                                          fft_frames=fft_frames,
                                          desc='Prepare training dataset...')

    # Get validation frames
    val_frames, val_frames_ids = wav_frames(args, val_data_pair,
                                            dataset_dir=dataset_dir,
                                            target_folder=target_folder,
                                            ref_folder=ref_folder,
                                            window_size_seconds=window_size_seconds,
                                            trg_fs=args.target_fs,
                                            n_samples=val_n_samples,
                                            freq_band_size=args.freq_band_size,
                                            nfft_scale=nfft_scale,
                                            fft_frames=fft_frames,
                                            noise=True,
                                            desc='Prepare validation dataset...')

    # Get test frames
    tst_frames, tst_frames_ids = wav_frames(args, tst_data_pair,
                                            dataset_dir=dataset_dir,
                                            target_folder=target_folder,
                                            ref_folder=ref_folder,
                                            window_size_seconds=window_size_seconds,
                                            trg_fs=args.target_fs,
                                            n_samples=tst_n_samples,
                                            freq_band_size=args.freq_band_size,
                                            nfft_scale=nfft_scale,
                                            fft_frames=fft_frames,
                                            desc='Prepare test dataset...')

    return tr_frames, tr_frames_ids, val_frames, val_frames_ids, tst_frames, tst_frames_ids

    # tr_dataset = AudioFolder(tr_ref_frames, tr_trg_frames)
    # del tr_trg_frames
    # del tr_ref_frames
    #
    # val_dataset = AudioFolder(val_ref_frames, val_trg_frames)
    # del val_ref_frames
    # del val_trg_frames
    #
    # tst_dataset = AudioFolder(tst_ref_frames, tst_trg_frames)
    # del tst_ref_frames
    # del tst_trg_frames
    #
    # train_loader = torch.utils.data.DataLoader(tr_dataset,
    #                                            batch_size=args.tr_batch_size,
    #                                            num_workers=args.num_workers,
    #                                            pin_memory=True,
    #                                            shuffle=True)
    #
    # validation_loader = torch.utils.data.DataLoader(val_dataset,
    #                                                 batch_size=args.val_batch_size,
    #                                                 num_workers=args.num_workers,
    #                                                 pin_memory=True)
    #
    # test_loader = torch.utils.data.DataLoader(tst_dataset,
    #                                           batch_size=args.tst_batch_size,
    #                                           num_workers=args.num_workers,
    #                                           pin_memory=True)
    #
    # print(f'Train size: {len(train_loader.dataset)} '
    #       f'-- Validation size: {len(validation_loader.dataset)} '
    #       f'-- Test size: {len(test_loader.dataset)}')
    #
    # print(f'Original Input signal dim {train_loader.dataset.target_signal_dim} '
    #       f'-- Original Output signal dim {train_loader.dataset.reference_frame_dim}')
    #
    # loaders = train_loader, validation_loader, test_loader
    # # signals_dim = tr_dataset.target_signal_dim, tr_dataset.reference_frame_dim, tr_dataset.reference_spectrum_dim
    # signals_dim = tr_dataset.target_signal_dim, tr_dataset.reference_frame_dim
    #
    # return loaders, signals_dim


def quadratic_interpolation(data, max_idx, bin_size, freq_band_size=0.1, enf_harmonic_n=1, threshold=False):
    """
        https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    """
    left = data[max_idx - 1]
    center = data[max_idx]
    right = data[max_idx + 1]

    if threshold:
        peak = (max_idx) * bin_size
    else:
        p = 0.5 * (left - right) / (left - 2 * center + right)
        peak = (max_idx + p) * bin_size  # interpolated peak

    if peak < enf_harmonic_n * 50 - freq_band_size:
        peak = enf_harmonic_n * 50 - freq_band_size
    elif peak > enf_harmonic_n * 50 + freq_band_size:
        peak = enf_harmonic_n * 50 + freq_band_size

    return peak


def get_max_freq(frames, data_fs, nperseg, freq_band_size, enf_harmonic_n, nfft_scale):
    if nfft_scale:
        nfft = next_power_of_2(int(nfft_scale * nperseg))
        bin_size = data_fs / nfft
        threshold = True
    else:
        nfft = data_fs * 2000
        bin_size = data_fs / nfft
        threshold = True

    Zxx = np.apply_along_axis(rfft, arr=frames, axis=0, n=nfft)
    max_freqs = []
    for spectrum in np.abs(Zxx.transpose()):  # Transpose to iterate on time frames
        max_amp = np.amax(spectrum)
        max_freq_idx = np.where(spectrum == max_amp)[0][0]
        # max_freq = spectrum[max_freq_idx-1] * bin_size

        max_freq = quadratic_interpolation(data=spectrum,
                                           max_idx=max_freq_idx,
                                           bin_size=bin_size,
                                           freq_band_size=freq_band_size,
                                           enf_harmonic_n=enf_harmonic_n,
                                           threshold=threshold)
        max_freqs.append(max_freq)
    # return np.array(max_freqs)
    return np.expand_dims(np.array(max_freqs), axis=1)


from scipy.signal._spectral_py import _triage_segments


# https://github.com/dpwe/audfprint/blob/cb03ba99feafd41b8874307f0f4e808a6ce34362/stft.py
def frame(data, nperseg, noverlap, equal_sized=False):
    step = nperseg - noverlap
    num_samples = data.shape[0]
    num_frames = 1 + ((num_samples - nperseg) // step)
    shape = (num_frames, nperseg) + data.shape[1:]
    if equal_sized:
        result = np.zeros(shape)
        window_pos = range(0, len(data) - nperseg + 1, step)
        for i, w in enumerate(window_pos):
            result[i] = data[w:w + nperseg]
    else:

        strides = (data.strides[0] * step,) + data.strides
        result = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    return result.T, num_frames


def get_wav_frames(data, data_fs, window_size_seconds, window='hann'):
    nperseg = data_fs * window_size_seconds  # Length of each segment.
    noverlap = data_fs * (window_size_seconds - 1)

    window_vector, _nperseg = _triage_segments(window, nperseg, input_length=len(data))
    assert _nperseg == nperseg

    frames, n_frames = frame(data, nperseg, noverlap, equal_sized=True)

    frames = frames * window_vector[:, np.newaxis]

    # if fft_output:
    #
    #     bin_size = data_fs / nperseg
    #
    #     Zxx = np.apply_along_axis(rfft, 0, frames)
    #
    #     max_freqs = []
    #     for spectrum in np.abs(Zxx.transpose()):  # Transpose to iterate on time frames
    #         max_amp = np.amax(spectrum)
    #         max_freq_idx = np.where(spectrum == max_amp)[0][0]
    #         # max_freq = spectrum[max_freq_idx-1] * bin_size
    #
    #         max_freq = quadratic_interpolation(spectrum, max_freq_idx, freq_band_size, enf_harmonic_n, bin_size)
    #         max_freqs.append(max_freq)
    #
    #     return frames, n_frames, np.expand_dims(np.array(max_freqs), axis=0)

    return frames, n_frames, nperseg, noverlap


def butter_bandpass_filter(data, locut, hicut, synthetic_fs, order):
    """Passes input data through a Butterworth bandpass filter. Code borrowed from
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

    :param data: list of signal sample amplitudes
    :param locut: frequency (in Hz) to start the band at
    :param hicut: frequency (in Hz) to end the band at
    :param synthetic_fs: the sample rate
    :param order: the filter order
    :returns: list of signal sample amplitudes after filtering
    """
    nyq = 0.5 * synthetic_fs
    low = locut / nyq
    high = hicut / nyq
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')

    return signal.sosfilt(sos, data)


# def process_frames(i, augmenter, trg_frames, trg_fs, target_locut, target_hicut):
#     # return print(trg_frames[i])
#     return butter_bandpass_filter(augmenter(samples=trg_frames[i], sample_rate=trg_fs),
#                                   target_locut,
#                                   target_hicut, trg_fs,
#                                   order=10).astype(np.float32)

def bandpass_filter(frame, enf_harmonic_n, signal_fs, freq_band_size=0.1, nominal_freq=50):
    ref_locut = enf_harmonic_n * (nominal_freq - freq_band_size)
    ref_hicut = enf_harmonic_n * (nominal_freq + freq_band_size)
    frame = butter_bandpass_filter(frame, ref_locut, ref_hicut, signal_fs, order=10).astype(np.float32)
    return frame


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def stft(data, synthetic_fs, window_size_seconds, nfft=None):
    nperseg = synthetic_fs * window_size_seconds  # Length of each segment.
    noverlap = synthetic_fs * (window_size_seconds - 1)
    _, _, Zxx = signal.stft(data, synthetic_fs, window='han', nperseg=nperseg, noverlap=noverlap,
                            boundary=None, padded=False, nfft=nfft)

    return np.abs(Zxx).ravel()


def plt_power_spectrum(data, data_filtered, fs):
    N = 15 * len(data)
    X = rfft(data, n=N)
    Y = rfft(data_filtered, n=N)

    f = np.arange(N // 2 + 1) * fs / N

    # fig.set_size_inches((10, 4))
    plt.plot(f, 20 * np.log10(abs(X)), 'r-', label='FFT original signal')
    plt.plot(f, 20 * np.log10(abs(Y)), 'g-', label='FFT filtered signal')
    # plt.xlim(xmax=fs / 2)
    # plt.ylim(ymin=-20)
    plt.ylabel(r'Power Spectrum (dB)', fontsize=8)
    plt.xlabel("frequency (Hz)", fontsize=8)
    plt.grid()
    plt.legend(loc=0)

    plt.tight_layout()

    plt.show()


def unitary_projection(M):
    '''
    Projects M on the rotation manifold
    Parameters
    ----------
    M : array, shape (M, M)
        Input matrix
    '''
    s, u = np.linalg.eigh(np.dot(M, M.T))
    return np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), M)


def check_random_state(seed):
    '''Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    '''
    import numbers
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


import numpy as np
from scipy import signal


def get_dummy_enf(args, n_samples, fs, fft_frames, nfft_scale, desc):
    enf_frames = []

    _enf_size = fs * 10 * 60
    _enf_nperseg = fs * args.window_size_seconds

    enf_phi = fftpack.dct(np.eye(_enf_nperseg), 3, norm='ortho')

    l2_transformer = Normalizer()
    scaler = MinMaxScaler()

    for _ in tqdm(range(n_samples), desc=desc):

        ENF = np.zeros(_enf_size)
        f0 = np.random.randn(_enf_size)
        A = 1 + np.random.randn(_enf_size) * 0.005  # amplitude
        b = [1, -0.99]
        a = 1
        f = signal.lfilter(b, a, f0) * 0.0005 + 50  # instantaneous frequency
        theta = np.random.uniform(0, 2 * np.pi)  # phase

        for n in range(_enf_size):
            ENF[n] = A[n] * np.cos(2 * np.pi / fs * np.sum(f[:n]) + theta)  # clean ENF

        if args.bandpass:
            ENF = bandpass_filter(ENF, enf_harmonic_n=2,
                                  signal_fs=fs,
                                  freq_band_size=args.freq_band_size,
                                  nominal_freq=50)
        elif args.multi_bandpass:
            window = 'kaiser'
            nominal_frequencies = [50., 100., 150.]
            cut_off = []
            for nominal_freq in nominal_frequencies:
                cut_off.extend([nominal_freq - args.freq_band_size, nominal_freq + args.freq_band_size])

            taps_kaiser = bandpass_firwin(n_taps=128,
                                          cut_off=cut_off,
                                          transition_width_hz=args.transition_width_hz,
                                          ripple_db=args.ripple_db,
                                          fs=fs,
                                          window=window)
            b, a = taps_kaiser, [1]
            ENF = filtfilt(b=b, a=1, axis=0, x=ENF,
                           padtype='odd',
                           padlen=3 * (max(len(b), len(a)) - 1))

            # plt_power_spectrum(target_data, target_data, trg_fs)

        _ENF_frames, _ENF_n_frames, _ENF_nperseg, _ = get_wav_frames(ENF,
                                                                     fs,
                                                                     args.window_size_seconds)

        if fft_frames:

            _rfft = False
            if _rfft:
                _nfft = next_power_of_2(nfft_scale * _ENF_nperseg)

                zxx_enf = np.abs(rfft(_ENF_frames, n=_nfft, axis=0))
                zxx_enf = np.log1p(zxx_enf ** 2)

                # trg_spectrum = rfft(_trg_frames, n=trg_nfft, axis=0) / trg_nfft
                # zxx_trg = np.abs(trg_spectrum * np.conj(trg_spectrum))
                # zxx_trg = 20 * np.log1p(zxx_trg)  # scale to db
                # zxx_trg = np.clip(zxx_trg, -40, 200)

            else:
                # _trg_frames = w_trg * _trg_frames
                # _ref_frames = w_ref * _ref_frames
                zxx_enf = np.dot(enf_phi, _ENF_frames) ** 2

            # zxx_trg = scaler.fit_transform(zxx_trg)
            # zxx_ref = scaler.fit_transform(zxx_ref)
            zxx_trg = l2_transformer.fit_transform(zxx_enf.T).T

            enf_frames.append(zxx_trg)

        else:
            _ENF_frames[_ENF_frames < 1e-5] = 0

            # _trg_frames = np.abs(_trg_frames)
            # _ref_frames = np.abs(_ref_frames)

            _ENF_frames = scaler.fit_transform(_ENF_frames)
            _ENF_frames = l2_transformer.fit_transform(_ENF_frames.T).T
            enf_frames.append(_ENF_frames)

    enf_frames = np.hstack(enf_frames).T

    return enf_frames


def convert_to_decibel(arr):
    ref = 1
    if arr != 0:
        return 20 * np.log10(abs(arr) / ref)

    else:
        return -60


def wav_frames(args,
               target_ref_pair,
               dataset_dir,
               target_folder='H1',
               ref_folder='H1_ref',
               window_size_seconds=16,
               trg_fs=8000,
               n_samples=None,
               freq_band_size=0.1,
               nfft_scale=1,
               fft_frames=False,
               noise=False,
               desc=None):
    l2_transformer = Normalizer()
    scaler = MinMaxScaler()

    _trg_nperseg = trg_fs * window_size_seconds
    _ref_nperseg = 400 * window_size_seconds

    # rng = check_random_state(np.random.RandomState(0))
    # trg_phi = unitary_projection(rng.randn(_trg_nperseg, _trg_nperseg))
    # ref_phi = unitary_projection(rng.randn(_ref_nperseg, _ref_nperseg))

    trg_phi = fftpack.dct(np.eye(_trg_nperseg), n=_trg_nperseg, type=3, norm='ortho')
    ref_phi = fftpack.dct(np.eye(_ref_nperseg), n=_ref_nperseg, type=3, norm='ortho')

    # t_trg = np.linspace(0, np.pi, num=_trg_nperseg).reshape(_trg_nperseg, 1)
    # w_trg = np.sin(t_trg)
    #
    # t_ref = np.linspace(0, np.pi, num=_ref_nperseg).reshape(_ref_nperseg, 1)
    # w_ref = np.sin(t_ref)

    target_frames, reference_frames, reference_frames_freq, target_frames_freq = [], [], [], []
    target_frames_id = []
    for target_file_name, ref_file_name in tqdm(target_ref_pair[:n_samples], desc=desc):
        #############################
        # Load reference file sample
        #############################
        reference_data, ref_fs = torchaudio.load(os.path.join(dataset_dir, ref_folder, ref_file_name))

        reference_data = np.squeeze(reference_data.numpy())
        reference_data = reference_data.astype(np.float32)

        # reference_data /= np.max(np.abs(reference_data))
        # reference_data -= np.mean(reference_data)
        # reference_data = (reference_data - np.mean(reference_data)) / np.std(reference_data)

        ##########################
        # Load target file sample
        ##########################
        target_data, _ = torchaudio.load(os.path.join(dataset_dir, target_folder, target_file_name))

        target_data = np.squeeze(target_data.numpy())
        target_data = target_data.astype(np.float32)

        # target_data /= np.max(np.abs(target_data))
        # target_data -= np.mean(target_data)
        # target_data = (target_data - np.mean(target_data)) / np.std(target_data)

        target_data = ssg.resample_poly(target_data, trg_fs, 8000)

        if args.bandpass:
            target_data = bandpass_filter(target_data, enf_harmonic_n=2,
                                          signal_fs=trg_fs,
                                          freq_band_size=freq_band_size,
                                          nominal_freq=50)

            reference_data = bandpass_filter(reference_data, enf_harmonic_n=1,
                                             signal_fs=ref_fs,
                                             freq_band_size=freq_band_size,
                                             nominal_freq=50)
        elif args.multi_bandpass:
            window = 'kaiser'
            nominal_frequencies = [50., 100., 150.]
            cut_off = []
            for nominal_freq in nominal_frequencies:
                cut_off.extend([nominal_freq - args.freq_band_size, nominal_freq + args.freq_band_size])

            trg_taps_kaiser = bandpass_firwin(n_taps=128,
                                              cut_off=cut_off,
                                              transition_width_hz=args.transition_width_hz,
                                              ripple_db=args.ripple_db,
                                              fs=trg_fs,
                                              window=window)
            b, a = trg_taps_kaiser, [1]
            target_data = filtfilt(b=b, a=1, axis=0, x=target_data,
                                   padtype='odd',
                                   padlen=3 * (max(len(b), len(a)) - 1))

            # plt_power_spectrum(target_data, target_data, trg_fs)

            ref_taps_kaiser = bandpass_firwin(n_taps=128,
                                              cut_off=cut_off,
                                              transition_width_hz=args.transition_width_hz,
                                              ripple_db=args.ripple_db,
                                              fs=ref_fs,
                                              window=window)
            b, a = ref_taps_kaiser, [1]
            reference_data = filtfilt(b=b, a=1, axis=0, x=reference_data,
                                      padtype='odd',
                                      padlen=3 * (max(len(b), len(a)) - 1))

            # plt_power_spectrum(reference_data, reference_data_filtered, ref_fs)
        # else:
        #     print('No filtering is applied in H1.')

        _trg_frames, _trg_n_frames, _trg_nperseg, _ = get_wav_frames(target_data,
                                                                     trg_fs,
                                                                     window_size_seconds)

        _ref_frames, _, _ref_nperseg, _ = get_wav_frames(reference_data,
                                                         ref_fs,
                                                         window_size_seconds,
                                                         'boxcar')

        if fft_frames:
            #     zxx_trg = []
            #     for _trg_frame in _trg_frames.T:
            #         _trg_stft = np.abs(librosa.stft(_trg_frame, n_fft=128, hop_length=25)).ravel()
            #         _trg_stft = l2_transformer.fit_transform(_trg_stft[:, None]).squeeze()
            #         zxx_trg.append(_trg_stft)
            #     zxx_trg = np.vstack(zxx_trg).T
            #
            #     zxx_ref = []
            #     for _ref_frame in _ref_frames.T:
            #         _ref_stft = np.abs(librosa.stft(_ref_frame, n_fft=128, hop_length=25)).ravel()
            #         _ref_stft = l2_transformer.fit_transform(_ref_stft[:, None]).squeeze()
            #         zxx_ref.append(_ref_stft)
            #     zxx_ref = np.vstack(zxx_ref).T

            _rfft = True
            if _rfft:
                trg_nfft = next_power_of_2(nfft_scale * _trg_nperseg)
                # _trg_frames = scale(_trg_frames, axis=0)
                # zxx_trg = np.abs(rfft(_trg_frames, n=trg_nfft, axis=0))
                # zxx_trg = np.abs(rfft(_trg_frames, n=trg_nfft, axis=0) / np.sqrt(_trg_nperseg) / 2)

                zxx_trg = np.abs(rfft(_trg_frames, n=trg_nfft, axis=0))
                zxx_trg /= np.linalg.norm(zxx_trg, axis=0, keepdims=True)

                # zxx_trg = np.log1p(zxx_trg**2)

                # trg_spectrum = rfft(_trg_frames, n=trg_nfft, axis=0) / trg_nfft
                # zxx_trg = np.abs(trg_spectrum * np.conj(trg_spectrum))
                # zxx_trg = 20 * np.log1p(zxx_trg)  # scale to db
                # zxx_trg = np.clip(zxx_trg, -40, 200)

                ref_nfft = next_power_of_2(nfft_scale * _ref_nperseg)
                # _ref_frames = scale(_ref_frames, axis=0)
                # zxx_ref = np.abs(rfft(_ref_frames, n=ref_nfft, axis=0) / np.sqrt(_ref_nperseg) / 2)
                zxx_ref = np.abs(rfft(_ref_frames, n=ref_nfft, axis=0))  # / ref_nfft
                zxx_ref /= np.linalg.norm(zxx_ref, axis=0, keepdims=True)

                # zxx_ref = np.log1p(zxx_ref**2)

                # ref_spectrum = rfft(_ref_frames, n=ref_nfft, axis=0) / ref_nfft
                # zxx_ref = np.abs(ref_spectrum * np.conj(ref_spectrum))
                # zxx_ref = 20 * np.log1p(zxx_ref)  # scale to db
                # zxx_ref = np.clip(zxx_ref, -40, 200)

            else:
                # _trg_frames = w_trg * _trg_frames
                # _ref_frames = w_ref * _ref_frames
                # _trg_frames = l2_transformer.fit_transform(_trg_frames.T).T
                # _ref_frames = l2_transformer.fit_transform(_ref_frames.T).T
                # _trg_frames = scale(_trg_frames, axis=0)
                # _ref_frames = scale(_ref_frames, axis=0)

                zxx_trg = np.abs(np.dot(trg_phi.T, _trg_frames)) ** 2
                zxx_ref = np.abs(np.dot(ref_phi.T, _ref_frames)) ** 2

            # zxx_trg = scaler.fit_transform(zxx_trg)
            # zxx_ref = scaler.fit_transform(zxx_ref)
            # zxx_trg /= np.max(zxx_trg, axis=0)
            # zxx_ref /= np.max(zxx_ref, axis=0)

            # zxx_trg = l2_transformer.fit_transform(zxx_trg.T).T
            # zxx_ref = l2_transformer.fit_transform(zxx_ref.T).T

            target_frames.append(zxx_trg)
            reference_frames.append(zxx_ref)

        else:
            # _trg_frames[_trg_frames < 1e-5] = 0
            # _ref_frames[_ref_frames < 1e-5] = 0

            # _trg_frames = np.abs(_trg_frames)
            # _ref_frames = np.abs(_ref_frames)

            # _trg_frames = scaler.fit_transform(_trg_frames)
            # _ref_frames = scaler.fit_transform(_ref_frames)
            #
            # NOISE_FLOOR = 1e-6
            # _trg_frames = -np.vectorize(convert_to_decibel)(_trg_frames + NOISE_FLOOR)
            # _ref_frames = -np.vectorize(convert_to_decibel)(_ref_frames + NOISE_FLOOR)

            # _trg_frames = (1 / 2) * (np.arcsin(_trg_frames) + np.pi / 2)
            # _ref_frames = (1 / 2) * (np.arcsin(_ref_frames) + np.pi / 2)

            # _trg_frames = l2_transformer.fit_transform(_trg_frames.T).T
            # _ref_frames = l2_transformer.fit_transform(_ref_frames.T).T

            # _trg_frames = scale(_trg_frames, axis=0)
            # _ref_frames = scale(_ref_frames, axis=0)

            _trg_frames /= np.max(np.abs(_trg_frames), axis=0)
            _ref_frames /= np.max(np.abs(_ref_frames), axis=0)

            target_frames.append(_trg_frames)
            reference_frames.append(_ref_frames)

        target_frames_id.extend(_trg_n_frames * [target_file_name])

    reference_frames = np.hstack(reference_frames).T
    target_frames = np.hstack(target_frames).T

    return (reference_frames, target_frames), target_frames_id


def get_data(target_folder, ref_folder,
             numpy_seed, snr, noise_type,
             ref_one_day_folder=None,
             window_size_seconds=16,
             nnft=None, n_samples=None):
    target_ref_pair, target_ref_one_day_pair = read_wav_pairs(target_folder,
                                                              ref_folder,
                                                              ref_one_day_folder)

    _spectrum_frames = wav_frames(target_ref_pair,
                                  target_folder,
                                  ref_folder,
                                  numpy_seed, snr, noise_type,
                                  window_size_seconds,
                                  nnft,
                                  n_samples)

    # reference_frames, ref_spectrum_frames, ref_spectrum_enfs, target_frames,
    # target_spectrum_enfs, trg_fs, ref_fs = _spectrum_frames
    return _spectrum_frames


def data_pairs(target_folder, ref_folder, ref_one_day_folder):
    target_files = [Path(f).stem for f in os.listdir(target_folder) if f.endswith('.wav')]
    ref_files = [Path(f).stem for f in os.listdir(ref_folder) if f.endswith('.wav')]
    ref_one_day_files = [Path(f).stem for f in os.listdir(ref_one_day_folder) if f.endswith('.wav')]

    target_ref_pair_dict = {}
    for target_file in target_files:
        try:
            ref_file = [f for f in ref_files if target_file in f].pop()
        except IndexError as e:
            print(f'Error: {e} -- target file {target_file} is not in ref files.')
            continue

        target_ref_pair_dict.update({target_file: ref_file})

    target_ref_one_day_pair_dict = {}
    for target_file in target_files:
        for f in ref_one_day_files:
            try:
                x, y, _ = f.replace('_', '-').split('-')
                x, y = int(x), int(y)
                if x <= int(target_file) <= y:
                    target_ref_one_day_pair_dict.nfftupdate({target_file: f})
            except ValueError:
                x, _ = f.replace('_', '-').split('-')
                x = int(x)
                if x == int(target_file):
                    target_ref_one_day_pair_dict.update({target_file: f})

    target_ref_pair = [(str(k).zfill(3) + '.wav', target_ref_pair_dict[k] + '.wav')
                       for k in sorted(target_ref_pair_dict, key=lambda tt: (int(tt), tt))]

    target_ref_one_day_pair = [(str(k).zfill(3) + '.wav', target_ref_one_day_pair_dict[k] + '.wav')
                               for k in sorted(target_ref_one_day_pair_dict, key=lambda tt: (int(tt), tt))]

    return target_ref_pair, target_ref_one_day_pair
