import os
import pandas as pd


def get_files(dir_name, ext='CSV'):
    """get file list in the given folder"""
    files = []
    file_list = os.listdir(dir_name)
    file_list = sorted(file_list)
    for file in file_list:
        if file.endswith(ext):
            files.append(os.path.join(dir_name, file))
    return files


def read_ds_lvm(filename, get_header=True):
    if not os.path.isfile(filename):
        raise ValueError('wrong filename')

    with open(filename, 'r') as f:
        lines = f.readlines()

    lines = [x.strip() for x in lines]
    end_cmd_list = [i for i, x in enumerate(lines) if x == '***End_of_Header***']

    headers = []
    prev = 0
    for idx in end_cmd_list:
        headers.append('\n'.join(lines[prev:idx]))
        prev = idx + 1

    del lines

    ds = pd.read_csv(filename, sep='\t', decimal=',', skiprows=prev)
    if get_header:
        return ds, headers
    else:
        return ds
