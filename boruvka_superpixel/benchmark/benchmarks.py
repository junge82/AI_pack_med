# python3

# numpy family
import numpy as np

def asa(supix, segment):
    assert supix.shape == segment.shape, ('supix and segment has to have '
            'identical shape')
    supix = supix.flatten()
    segment = segment.flatten()
    isect = np.zeros((supix.max() + 1, segment.max() + 1), dtype=int)
    for p in range(len(supix)):
        isect[supix[p], segment[p]] += 1
    return isect.max(axis=1).sum() / supix.shape[0]


# vim: set sw=4 sts=4 expandtab :
