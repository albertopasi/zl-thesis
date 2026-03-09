import numpy as np
import scipy.io as sio
from io_utils import smooth_moving_average
import os
import h5py
import pickle


def load_srt_raw_newPre(timeLen, timeStep, fs, channel_norm, time_norm, label_type):
    # THU-EP dataset constants
    n_channs = 30       # 32 acquisition channels minus 2 mastoid refs (A1, A2)
    fs = 250            # 250 Hz sampling rate
    n_vids = 28         # 28 emotion-eliciting stimuli
    chn = 30
    sec = 30            # 30-second trials → 7500 time points at 250 Hz
    n_points = fs * sec  # 7500

    # Sub 75 is absent from cl_cs_preprocessed/ (skipped during preprocessing)
    # Subjects 37 and 46 have some corrupted stimuli but are loaded as-is for cl-cs

    data_len = fs * timeLen
    n_segs = int((n_points / fs - timeLen) / timeStep + 1)
    print('n_segs:', n_segs)

    # Pickle files produced by src/thu_ep/preprocessing/preprocess_for_cl_cs.py
    # Path is relative to baselines/cl-cs/ (two levels up to project root, then into data/)
    data_path = '../../data/thu ep/cl_cs_preprocessed'
    data_paths = sorted([p for p in os.listdir(data_path) if p.endswith('.pkl')])

    data = np.zeros((len(data_paths), n_vids, chn, n_points))

    # THU-EP videos are stored in sequential order (0-indexed)
    vid_orders = np.zeros((len(data_paths), n_vids), dtype=int)
    for idx in range(len(data_paths)):
        seq = list(np.arange(n_vids))
        vid_orders[idx, :] = seq

    for idx, path in enumerate(data_paths):
        f = open(os.path.join(data_path, path), 'rb')
        data_sub = pickle.load(f)
        f.close()
        data[idx, vid_orders[idx, :], :, :] = data_sub

    # data shape: (n_subs, n_vids, n_channs, n_points)
    print('data loaded:', data.shape)

    n_subs = data.shape[0]

    # For binary classification: drop neutral stimuli (12-15), keep neg (0-11) + pos (16-27)
    if label_type == 'cls2':
        vid_sel = list(range(12))           # Negative: stimuli 0-11 (Anger, Disgust, Fear, Sadness)
        vid_sel.extend(list(range(16, 28))) # Positive: stimuli 16-27 (Amusement, Inspiration, Joy, Tenderness)
        data = data[:, vid_sel, :, :]       # (n_subs, 24, n_channs, n_points)
        n_videos = 24
    else:
        n_videos = 28

    data = np.transpose(data, (0, 1, 3, 2)).reshape(n_subs, -1, n_channs)

    if channel_norm:
        for i in range(data.shape[0]):
            data[i, :, :] = (data[i, :, :] - np.mean(data[i, :, :], axis=0)) / np.std(data[i, :, :], axis=0)

    if time_norm:
        data = (data - np.tile(np.expand_dims(np.mean(data, axis=2), 2), (1, 1, data.shape[2]))) / np.tile(
            np.expand_dims(np.std(data, axis=2), 2), (1, 1, data.shape[2])
        )

    n_samples = np.ones(n_videos) * n_segs

    # THU-EP label mapping
    # Stimuli 0-2: Anger(0), 3-5: Disgust(1), 6-8: Fear(2), 9-11: Sadness(3)
    # Stimuli 12-15: Neutral(4) [4 stimuli], 16-18: Amusement(5), 19-21: Inspiration(6)
    # Stimuli 22-24: Joy(7), 25-27: Tenderness(8)
    if label_type == 'cls2':
        label = [0] * 12  # 12 negative stimuli
        label.extend([1] * 12)  # 12 positive stimuli
    elif label_type == 'cls9':
        label = [0]*3 + [1]*3 + [2]*3 + [3]*3 + [4]*4 + [5]*3 + [6]*3 + [7]*3 + [8]*3
        print('label', label)
    elif label_type == 'cls3':
        # cls3 is not applicable to THU-EP; kept as stub
        raise ValueError('cls3 is not defined for THU-EP. Use cls2 or cls9.')

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]] * n_segs

    return data, label_repeat, n_samples, n_segs


def load_srt_pretrainFeat(datadir, channel_norm, timeLen, timeStep, isFilt, filtLen, label_type):
    # THU-EP: 24 stimuli for cls2 (neutral dropped), 28 for others; base window count = 30
    if label_type == 'cls2':
        n_samples = np.ones(24).astype(np.int32) * 30
    else:
        n_samples = np.ones(28).astype(np.int32) * 30

    for i in range(len(n_samples)):
        n_samples[i] = int((n_samples[i] - timeLen) / timeStep + 1)

    if datadir[-4:] == '.npy':
        data = np.load(datadir)
        data[data < -10] = -5
    elif datadir[-4:] == '.mat':
        data = sio.loadmat(datadir)['de_lds']
        print('isnan total:', np.sum(np.isnan(data)))
        data[np.isnan(data)] = -8
        # data[data < -8] = -8

    # data_use = data[:, np.max(data, axis=0)>1e-6]
    # data = data.reshape(45, int(np.sum(n_samples)), 256)
    print(data.shape)
    print(np.min(data), np.median(data))

    n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))
    if isFilt:
        print('filtLen', filtLen)
        data = data.transpose(0, 2, 1)
        for i in range(data.shape[0]):
            for vid in range(len(n_samples)):
                data[i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid + 1])] = smooth_moving_average(
                    data[i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid + 1])], filtLen)
        data = data.transpose(0, 2, 1)

    # Normalization for each sub
    if channel_norm:
        print('subtract mean and divided by var')
        for i in range(data.shape[0]):
            # data[i,:,:] = data[i,:,:] - np.mean(data[i,:,:], axis=0)
            data[i, :, :] = (data[i, :, :] - np.mean(data[i, :, :], axis=0)) / (np.std(data[i, :, :], axis=0) + 1e-3)

    # THU-EP label mapping (same as load_srt_raw_newPre)
    if label_type == 'cls2':
        label = [0] * 12  # 12 negative stimuli
        label.extend([1] * 12)  # 12 positive stimuli
    elif label_type == 'cls9':
        # 0=Anger(3), 1=Disgust(3), 2=Fear(3), 3=Sadness(3), 4=Neutral(4),
        # 5=Amusement(3), 6=Inspiration(3), 7=Joy(3), 8=Tenderness(3) → 28 total
        label = [0]*3 + [1]*3 + [2]*3 + [3]*3 + [4]*4 + [5]*3 + [6]*3 + [7]*3 + [8]*3
        print(label)
    elif label_type == 'cls3':
        raise ValueError('cls3 is not defined for THU-EP. Use cls2 or cls9.')

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]] * n_samples[i]
    return data, label_repeat, n_samples
