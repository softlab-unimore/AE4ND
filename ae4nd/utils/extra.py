import pandas as pd


def create_triplet_time_series(ts: pd.Series, with_support: bool = False):
    """
    create triplet ae4nd series encoding
    withSupport if return the number of compressed records
    """
    res = []
    start = -1
    prev = -1
    support = 0
    for k, val in ts.iteritems():
        support += 1
        if start == -1 and val > 0:
            start = k
            support = 0
        elif start >= 0 and val == 0:
            x = {
                'feature': ts.name,
                'start': start,
                'end': prev
            }
            if with_support:
                x['support'] = support
            res.append(x)
            start = -1

        prev = k

    if start != -1:
        x = {
            'feature': ts.name,
            'start': start,
            'end': prev
        }
        if with_support:
            x['support'] = support
        res.append(x)

    return res
