# import os
# import pandas as pd
# import numpy as np
#
# from utils.tools import get_files
# from utils.manage_file import read_ds_lvm
#
#
# def main():
#     dirname = r'C:\Users\delbu\Projects\Dataset\Anomaly Detection\TEST 6  fail motore'
#     files = get_files(dirname, 'lvm')
#
#     print('\nRead: ', dirname)
#     for filename in files:
#         print('\n', os.path.basename(filename))
#
#         ds, _ = read_ds_lvm(filename)
#         print(ds.columns)
#
#
# if __name__ == '__main__':
#     main()
