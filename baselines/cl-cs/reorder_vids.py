import scipy.io as sio
from glob import glob
import hdf5storage
import numpy as np
import os
import copy


def video_order_load(dataset, n_vids, n_subs):
    vid_orders = np.zeros((n_subs, n_vids))
    for idx in range(n_subs):
        seq = list(np.array((11, 6, 1, 3, 7, 12, 4, 8, 13, 14, 9, 5, 10, 15, 2)) - 1)
        # np.random.shuffle(seq)
        vid_orders[idx, :] = seq
    print('vid_order: ', vid_orders)
    return vid_orders


def reorder_vids(data, vid_play_order, n_segs):
    # data: (n_subs, n_points, n_feas)
    n_vids = int(data.shape[1] / n_segs)
    n_subs = data.shape[0]
    # Deep copy
    vid_play_order_copy = vid_play_order.copy()

    if n_vids == 24:
        vid_play_order_new = np.zeros((n_subs, n_vids)).astype(np.int32)
        data_reorder = np.zeros_like(data)
        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub, :]
            tmp = tmp[(tmp < 13) | (tmp > 16)]
            tmp[tmp >= 17] = tmp[tmp >= 17] - 4
            tmp = tmp - 1
            vid_play_order_new[sub, :] = tmp

            data_sub = data[sub, :, :]
            data_sub = data_sub.reshape(n_vids, n_segs, data_sub.shape[-1])
            data_sub = data_sub[tmp, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(n_vids * n_segs, data_sub.shape[-1])

    elif n_vids == 28 or n_vids == 15:
        vid_play_order_new = np.zeros((n_subs, n_vids)).astype(np.int32)
        data_reorder = np.zeros_like(data)

        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub, :]
            tmp = tmp - 1
            vid_play_order_new[sub, :] = tmp

            data_sub = data[sub, :, :]
            data_sub = data_sub.reshape(n_vids, n_segs, data_sub.shape[-1])
            # Error occurs saying that the elements of tmp is not int
            tmp = [int(i) for i in tmp]
            data_sub = data_sub[tmp, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(n_vids * n_segs, data_sub.shape[-1])

    return data_reorder, vid_play_order_new


def reorder_vids_back(data, vid_play_order_new, n_segs):
    # data: (n_subs, n_points, n_feas)
    n_vids = int(data.shape[1] / n_segs)
    n_subs = data.shape[0]

    data_back = np.zeros((n_subs, n_vids, n_segs, data.shape[-1]))

    for sub in range(n_subs):
        data_sub = data[sub, :, :].reshape(n_vids, n_segs, data.shape[-1])
        data_back[sub, vid_play_order_new[sub, :], :, :] = data_sub
    data_back = data_back.reshape(n_subs, n_vids * n_segs, data.shape[-1])
    return data_back
