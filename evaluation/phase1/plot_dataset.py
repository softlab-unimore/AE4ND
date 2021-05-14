import os
import matplotlib.pyplot as plt

from timely.utils.tools import get_time_series_dataset

show_dir = False
show_filename = True


def plot_file(filename, title=''):
    ds = get_time_series_dataset(filename=filename, sep=';', col='DT')
    if ds is None:
        raise ValueError('Impossible read train file')

    fig, ax = plt.subplots(figsize=(20, 10))
    ds.plot(ax=ax)
    ax.set_title(title)
    plt.show()


normal_list = [
    '14', '17', '25', '27', '34', '35', '36', '37', '38',
    '39', '41', '42', '43', '44', '45', '46', '48', '50',
    '51', '56', '57', '58', '62', '63', '64', '67', '69'
]

data_dir = '/Users/delbu/Projects/Notebooks/PHM/Dataset/Dataset_1/LOG1HZ'
filename = 'File (35).CSV'

if show_filename:
    plot_file(os.path.join(data_dir, filename))

if show_dir:
    files = [name for name in os.listdir(data_dir) if name.endswith('.CSV')]
    for name in files:
        print('Show: {}'.format(name))
        code = int(name.split('(')[-1].split(')')[0])
        title = name
        if code in normal_list:
            title = 'NORMAL ' + title
        else:
            title = 'ANOMALY ' + title

        plot_file(os.path.join(data_dir, name), title=title)
