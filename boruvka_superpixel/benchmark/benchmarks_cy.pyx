import numpy as np
cimport numpy as np

def asa(supix, segment):
    assert supix.shape == segment.shape, ('supix and segment has to have '
            'identical shape')
    supix = supix.flatten().astype(np.int32)
    segment = segment.flatten().astype(np.int32)
    isect = np.zeros((supix.max() + 1, segment.max() + 1), dtype=np.int32)
    cdef np.int32_t [:] supix_mv = supix    # memview
    cdef np.int32_t [:] segment_mv = segment 
    cdef np.int32_t [:,:] isect_mv = isect
    cdef int p
    for p in range(supix_mv.shape[0]):
        isect_mv[supix_mv[p], segment_mv[p]] += 1
    return isect.max(axis=1).sum() / supix.shape[0]




# vim: set sw=4 sts=4 expandtab :
