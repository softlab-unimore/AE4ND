import os
import pandas as pd

from utils.tools import get_files

global_dir = r'\Users\delbu\Documents\Dataset\SBDIO\Dataset 6\LOG1HZ'


def main():
    # Get all .CSV files in global folder
    files = get_files(global_dir, ext='.CSV')
    print('Found {} files'.format(len(files)))

    normal_file_list = ["File ({}).CSV".format(x) for x in range(1, len(files) + 1)]

    name_list = []
    mean_list = []
    std_list = []
    prev = None

    for file in normal_file_list:
        path = os.path.join(global_dir, file)
        if not os.path.isfile(path):
            print('The following file doesn\'t exist: {}'.format(file))
            continue

        ds = pd.read_csv(path, sep=';', index_col='DT', encoding='latin1')
        if ds is None:
            print('Impossible read: {}'.format(file))
            return

        if prev is not None and prev > ds.index.min():
            print('ERROR prev {} > ds {}'.format(prev, ds.index.min()))

        mean_list.append(ds.mean(axis=0))
        std_list.append(ds.std(axis=0))
        name_list.append(file)

        prev = ds.index.min()

    mean_ds = pd.DataFrame(mean_list)
    std_ds = pd.DataFrame(std_list)

    mean_ds.to_csv('../results/mean.CSV', sep=';', index=False)
    std_ds.to_csv('../results/std.CSV', sep=';', index=False)


if __name__ == '__main__':
    main()
