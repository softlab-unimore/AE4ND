import os
import argparse
import pandas as pd


def get_arguments():
    parser = argparse.ArgumentParser(description='Run "setting_extraction"')
    parser.add_argument('--file',
                        type=str,
                        required=True,
                        help='file to convert into settings')

    parser.add_argument('--sep',
                        type=str,
                        default=';',
                        help='table separator')

    parser.add_argument('--complete',
                        action='store_true',
                        help='insert if you want the complete version of settings '
                             'without daily compression and with all columns')

    args = parser.parse_args()
    if not os.path.isfile(args.file):
        raise ValueError('{} is not a valid file'.format(args.file))
    return args


def get_selected_columns():
    """extract relevant columns for setting defined in params/columns.txt"""
    try:
        with open('../../params/columns.txt', 'r') as f:
            columns = f.readlines()
    except FileNotFoundError:
        return []
    columns = [x.strip() for x in columns]
    return columns


def get_states_flow(ds: pd.DataFrame, keep_na: bool = False, propagate: bool = True) -> pd.DataFrame:
    """transform ds DataFrame (DT-VARIABLE-VALUE)
    into settings dataset (timestamp/row x variables/columns)"""

    if 'DT' not in ds.columns or 'VARIABLE' not in ds.columns or 'VALUE' not in ds.columns:
        raise ValueError('ds doesn\'t contain the right columns {}'.format(ds.columns))

    # get features
    states = sorted(ds['VARIABLE'].unique())
    # group by datetime
    index = ds.groupby('DT').apply(list).index
    # transpose setting dataset
    res = pd.DataFrame(index=index, columns=states)
    res.sort_index(inplace=True)

    for idx, row in ds.groupby('DT').agg(lambda x: list(x)).iterrows():
        res.loc[idx, row['VARIABLE']] = row['VALUE']

    if not keep_na:
        if propagate:
            res.fillna(method='ffill', inplace=True)
        else:
            res.fillna(0, inplace=True)

    # transform index into datetime index
    res.index = pd.to_datetime(res.index)
    res.sort_index(inplace=True)
    return res


# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
def time_compress(ds, freq, get_conflict=False):
    """compress input DataFrame on freq"""
    times = ds.index.to_period(freq=freq)
    columns = ds.columns
    index = times.unique()

    columns = list(columns) + ['ltime', 'rtime']

    print("initial length: {}".format(len(times)))
    print("result length: {}".format(len(index)))

    res = pd.DataFrame(index=index, columns=columns)

    conflict_map = {}

    for time in index:
        conflicts = []
        for col in ds.columns:
            data = ds.loc[times == time, col].dropna()
            if not data.empty:
                # get last value
                res.loc[time, col] = data[-1]
                # save conflict
                if len(data.unique()) > 1 and get_conflict:
                    conflicts.append(col)

        # save conflict matching
        if conflicts:
            # keep only conflicting columns
            conflict_map[time] = ds.loc[times == time, conflicts].dropna(how='all')

        # save setting insertion timely window
        tw = ds.loc[times == time, :].index
        res.loc[time, 'ltime'] = tw[0]
        res.loc[time, 'rtime'] = tw[-1]

    if get_conflict:
        return res, conflict_map

    return res


def main():
    print('setting_extraction')
    args = get_arguments()
    # args = {'file': 'data/Ricette.CSV', 'sep': ';', 'complete': False}
    print('read file...')
    ds = pd.read_csv(args.file, args.sep)
    print('transpose dataset...')
    settings = get_states_flow(ds, propagate=True)

    output_dir = '../results'
    filename = os.path.basename(args.file)
    filename = os.path.splitext(filename)[0]
    filename = os.path.join(output_dir, 'settings_' + filename + '.CSV')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if args.complete:
        print('save complete setting version: {}'.format(filename))
        settings.to_csv(filename, sep=args.sep)
        return

    print('compress settings...')
    selected_columns = get_selected_columns()
    settings = settings[selected_columns]
    settings = time_compress(settings, freq='1D', get_conflict=False)

    print('save compressed setting version: {}'.format(filename))
    settings.to_csv(filename, sep=args.sep)
    return


if __name__ == '__main__':
    main()
