from __future__ import division
import glob
import os

# import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter

from wyrm import processing as proc
from wyrm import plot
from wyrm import io


data_folder = '/home/haydar/BCI/VPkk_08_08_14'
vhdr_files = glob.glob(os.path.join(data_folder, "*.vhdr"))
print(vhdr_files[0])
'''
data_0 = io.load_brain_vision_data(vhdr_files[0])

print(data_0.names)
print(data_0)
'''

cnt = io.load_brain_vision_data(vhdr_files[0])
# print(cnt.markers)

# remove unneeded channels
cnt = proc.remove_channels(cnt, ['EMG.*', 'EOG.*'])

# band pass data
print("filtering data")
fn = cnt.fs / 2
b, a = butter(4, [10 / fn, 14 / fn], btype='band')
cnt = proc.filtfilt(cnt, b, a)
print(cnt.data.shape)
# subsampling
print("subsampling")
cnt = proc.subsample(cnt, 100)
print(cnt.data.shape)
print("Making epochs")
# for mi we use ~750-3500ms range
mrk_def = {'class 1': ['S  1'],
           'class 2': ['S  2'],
           'class 3': ['S  3']
           }
epo = proc.segment_dat(cnt, mrk_def, [-100, 3500])
epo = proc.correct_for_baseline(epo, [-100, 0])
print(epo.data.shape)

# rectify channels
print("rectifying channels")
epo2 = proc.rectify_channels(epo)
print(epo2.data.shape)

epo_avg = proc.calculate_classwise_average(epo2)
'''
for i, e in enumerate(epo_avg.class_names):
    plot.plot_channels(proc.select_epochs(epo_avg, [i]))
plt.show()
'''

# select interval
print("selecting interval")
epo = proc.select_ival(epo, [750, 3500])
print(epo.data.shape)
print(epo.axes)

# calculate csp
print("calculating csp")

# data_left = proc.select_classes(epo, [0]).data
# data_right = proc.select_classes(epo, [1]).data
w, a, d = proc.calculate_csp(epo, [0, 1])
print(a)

# the interesting channels are usually c3, c4 and cz
# TODO plot_scalp doesn't subplot correctly
for ii, i in enumerate([0, 1, 2, -3, -2, -1]):
    plt.subplot(2, 3, ii+1)
    plot.plot_scalp(a[:, i], epo.axes[-1])
    plt.title(i)
plt.show()
