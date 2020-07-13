import os
import json
import matplotlib.pyplot as plt

from utils.tools import get_time_series_dataset

show_setup = False
show_filename = False
show_setting = '-1'


def plot_file(filename, title=''):
    ds = get_time_series_dataset(filename=filename, sep=';', col='DT')
    if ds is None:
        raise ValueError('Impossible read train file')

    fig, ax = plt.subplots(figsize=(20, 10))
    ds.plot(ax=ax)
    if title:
        ax.set_title(title)
    else:
        filename = os.path.basename(filename)
        ax.set_title(filename)
    plt.show()


# normal_list = [
#     '14', '17', '25', '27', '34', '35', '36', '37', '38', '39', '41', '42', '43', '44', '45', '46', '48', '50', '51',
#     '56', '57', '58', '62', '63', '64', '67', '69'
# ]

filename = '/Users/delbu/Projects/Notebooks/PHM/Dataset/Dataset_1/LOG1HZ\File (35).CSV'

if show_filename:
    plot_file(filename)

with open('../results/setup_files.json') as json_file:
    setup_files = json.load(json_file)

if show_setup:
    for file in setup_files:
        plot_file(file)

with open('../results/settings_map.json') as json_file:
    settings_map = json.load(json_file)

if show_setting in settings_map:
    for file in settings_map[show_setting]:
        print('show: {}'.format(file))
        title = os.path.basename(file)

        if file in setup_files:
            title = 'anomaly_' + title
        else:
            title = 'normal_' + title

        plot_file(file, title=title)
