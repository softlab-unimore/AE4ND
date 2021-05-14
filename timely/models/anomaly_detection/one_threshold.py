import numpy as np
from scipy import stats


class OneThreshold:

    def __init__(self, th, size, margin):
        self.th = th
        self.size = size
        self.margin = margin

    def predict(self, s, start):
        s = s[s.index >= start]
        start = s.index.min()

        if s.empty:
            print('s is empty')
            return None

        if np.std(s.values) == 0:
            print('s std is zero')
            return None

        # z-normalization on s
        s[:] = stats.zscore(s.values)

        # keep only not nan value
        cond = np.isnan(s).any()

        if cond:
            print('nan value in series')
            return start, start

        size_support = self.size
        margin_support = self.margin
        time = None
        reset = True

        for idx, val in s.iteritems():
            if reset:
                size_support = self.size
                margin_support = self.margin
                time = idx
                reset = False

            if abs(val) <= self.th:
                size_support -= 1
            elif margin_support > 0:
                size_support -= 1
                margin_support -= 1
            else:
                reset = True

            if size_support == 0:
                break

        return start, time
