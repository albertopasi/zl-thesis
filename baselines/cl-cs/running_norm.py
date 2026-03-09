import os
import scipy.io as sio
import numpy as np
from reorder_vids import video_order_load, reorder_vids, reorder_vids_back
import random
import argparse

print('\nrunning_norm.py')
parser = argparse.ArgumentParser(description='Finetune the pretrained model for EEG emotion recognition')
parser.add_argument('--timeLen', default=5, type=int, help='time length in seconds')
parser.add_argument('--use-data', default='pretrained', type=str, help='what data to use')
parser.add_argument('--normTrain', default='yes', type=str, help='whether normTrain')
parser.add_argument('--n-vids', default=15, type=int, help='use how many videos')
parser.add_argument('--randSeed', default=7, type=int, help='random seed')
parser.add_argument('--dataset', default='both', type=str, help='first or second')

parser.add_argument('--cls', default=3, type=int, help='how many cls to use')

args = parser.parse_args()

random.seed(args.randSeed)
np.random.seed(args.randSeed)
use_features = args.use_data
normTrain = args.normTrain

label_type = args.cls

if label_type == 9:
    label_type = 'cls9'
    n_vids = 28  # THU-EP: 28 stimuli
elif label_type == 5:
    label_type = 'cls5'
elif label_type == 2:
    label_type = 'cls2'
    n_vids = 24  # THU-EP cls2: 24 stimuli (neutral 12-15 dropped)
elif label_type == 3:
    label_type = 'cls3'
    n_vids = 28  # cls3 not applicable to THU-EP; kept as stub

n_folds = 10  # THU-EP: 10-fold cross-subject CV

data_dir = './runs_srt/'
if use_features == 'pretrained':
    print(use_features)
    if label_type == 'cls2':
        save_dir = data_dir + 'raw_fold%d_cls2' % n_folds
    elif label_type == 'cls3':
        save_dir = data_dir + 'raw_fold%d_cls3' % n_folds
    else:
        save_dir = data_dir + 'raw_fold%d_cls9' % n_folds

bn_val = 1
# rn_momentum = 0.995
# print(rn_momentum)
# momentum = 0.9

timeLen = 2
timeStep = 1
n_points = 7500  # THU-EP: 30s × 250Hz = 7500 timepoints/trial
fs = 250         # THU-EP: 250 Hz
n_segs = int((n_points / fs - timeLen) / timeStep + 1)
n_total = n_segs * n_vids
n_counters = int(np.ceil(n_total / bn_val))
print('n_total: ', n_total)
print('n_counters: ', n_counters)
dataset = args.dataset
n_subs = 79  # THU-EP: 79 subjects (sub_75 excluded; see docs/excluded_data.md)

n_per = round(n_subs / n_folds)

vid_order = video_order_load(args.dataset, n_vids, n_subs)

for decay_rate in [0.990]:
    print(decay_rate)
    for fold in range(n_folds):
        print('fold: ', fold)

        if (use_features == 'pretrained'):
            if normTrain == 'yes':
                data = sio.loadmat(os.path.join(save_dir, str(fold), 'features1_de_1s_normTrain.mat'))['de']
            else:
                data = sio.loadmat(os.path.join(save_dir, str(fold), 'features1_de_1s.mat'))['de']
        print('data: ', data.shape)

        if fold < n_folds - 1:
            val_sub = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            # val_sub = np.arange(n_per*fold, n_per*(fold+1)-1)
            val_sub = np.arange(n_per * fold, n_subs)
        val_sub = [int(val) for val in val_sub]
        print('val:', val_sub)
        train_sub = list(set(np.arange(n_subs)) - set(val_sub))

        # vid order just need to read one time
        print('data: ', data.shape)
        print('vid_order: ', vid_order.shape)
        data, vid_play_order_new = reorder_vids(data, vid_order, n_segs)
        print('Shape of new order: ', vid_play_order_new.shape)

        data[np.isnan(data)] = -n_segs

        data_mean = np.mean(np.mean(data[train_sub, :, :], axis=1), axis=0)
        data_var = np.mean(np.var(data[train_sub, :, :], axis=1), axis=0)

        data_norm = np.zeros_like(data)
        for sub in range(data.shape[0]):
            running_sum = np.zeros(data.shape[-1])
            running_square = np.zeros(data.shape[-1])
            decay_factor = 1.
            for counter in range(n_counters):
                data_one = data[sub, counter * bn_val: (counter + 1) * bn_val, :]
                running_sum = running_sum + data_one
                running_mean = running_sum / (counter + 1)
                # running_mean = counter / (counter+1) * running_mean + 1/(counter+1) * data_one
                running_square = running_square + data_one ** 2
                running_var = (running_square - 2 * running_mean * running_sum) / (counter + 1) + running_mean ** 2

                # print(decay_factor)
                curr_mean = decay_factor * data_mean + (1 - decay_factor) * running_mean
                curr_var = decay_factor * data_var + (1 - decay_factor) * running_var
                decay_factor = decay_factor * decay_rate

                # print(running_var[:3])
                # if counter >= 2:
                data_one = (data_one - curr_mean) / np.sqrt(curr_var + 1e-5)
                data_norm[sub, counter * bn_val: (counter + 1) * bn_val, :] = data_one

        data_norm = reorder_vids_back(data_norm, vid_play_order_new, n_segs)
        de = {'de': data_norm}
        if (use_features == 'pretrained'):
            if normTrain == 'yes':
                save_file = os.path.join(save_dir, str(fold), 'features1_de_1s_normTrain_rnPreWeighted%.3f_play_order.mat' % decay_rate)
            else:
                save_file = os.path.join(save_dir, str(fold), 'features1_de_1s_rnPreWeighted%.3f_play_order.mat' % decay_rate)
            print(save_file)
            sio.savemat(save_file, de)
