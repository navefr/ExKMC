# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.stdlib cimport malloc, free

ctypedef np.int32_t NP_INT_t
ctypedef np.float64_t NP_FLOAT_t

cdef extern from "<math.h>" nogil:
    const float INFINITY


cdef extern from "<limits.h>":
    const int INT_MIN
    const int INT_MAX


cdef struct IMM_Cut:
    int col
    float threshold


@cython.boundscheck(False)
@cython.wraparound(False)
def get_min_mistakes_cut(NP_FLOAT_t[:,:] X, NP_INT_t[:] y, NP_FLOAT_t[:,:] centers, NP_INT_t[:] valid_centers, NP_INT_t[:] valid_cols, int njobs):
    cdef int n = X.shape[0]
    cdef int k = centers.shape[0]
    cdef int d = valid_cols.shape[0]
    cdef int *centers_count = <int *> malloc(k * sizeof(int))
    cdef NP_FLOAT_t *cols_thresholds = <NP_FLOAT_t *> malloc(d * sizeof(NP_FLOAT_t))
    cdef int *cols_mistakes = <int *> malloc(d * sizeof(int))
    cdef int col
    cdef int best_col = -1
    cdef NP_FLOAT_t best_threshold
    cdef int min_mistakes = INT_MAX

    # Count the number of data points for each center.
    # This information will be helpful for fast mistakes calculation, once a threshold pass a center.
    for i in range(k):
        centers_count[i] = 0
    for i in range(n):
        centers_count[y[i]] += 1

    if njobs is None or njobs <= 1:
        # Iterate over valid coordinates
        for col in range(d):
            if valid_cols[col] == 1:
                update_col_min_mistakes_cut(X, y, centers, valid_centers, centers_count, cols_thresholds, cols_mistakes, col, n, d, k)
    else:
        # Iterate over valid coordinates
        for col in prange(d, nogil=True, num_threads=njobs):
            if valid_cols[col] == 1:
                update_col_min_mistakes_cut(X, y, centers, valid_centers, centers_count, cols_thresholds, cols_mistakes, col, n, d, k)

    for col in range(d):
        # This is a valid column
        if valid_cols[col] == 1:
            # We found a valid split
            if cols_mistakes[col] != -1:
                # This is a better cut
                if cols_mistakes[col] < min_mistakes:
                    best_col = col
                    min_mistakes = cols_mistakes[col]

    if best_col != -1:
        best_threshold = cols_thresholds[best_col]

    free(cols_thresholds)
    free(cols_mistakes)
    free(centers_count)

    if best_col == -1:
        return None
    else:
        return IMM_Cut(best_col, best_threshold)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_col_min_mistakes_cut(NP_FLOAT_t[:,:] X, NP_INT_t[:] y, NP_FLOAT_t[:,:] centers, NP_INT_t[:] valid_centers, int* centers_count, NP_FLOAT_t *cols_thresholds, int *cols_mistakes, int col, int n, int d, int k) nogil:
    cdef int i
    cdef int ix
    cdef int ic
    cdef int mistakes
    cdef NP_FLOAT_t prev_threshold
    cdef NP_FLOAT_t threshold
    cdef NP_FLOAT_t max_val
    cdef np.int64_t[:] data_order
    cdef np.int64_t[:] centers_order
    cdef int *left_centers_count = <int *> malloc(k * sizeof(int))
    cdef int curr_center_idx
    cdef bint valid_found = 0
    cdef bint is_center_threshold
    cdef int min_mistakes = INT_MAX
    cdef NP_FLOAT_t best_threshold

    # Sort data points and centers
    with gil:
        data_order = np.asarray(X[:,col]).argsort()
        centers_order = np.asarray(centers[:, col]).argsort()

    # Find maximal value of valid centers. Possible threshold must be strictly smaller than that.
    max_val = -INFINITY
    for i in range(k):
        if valid_centers[i] == 1:
            if centers[i, col] > max_val:
                max_val = centers[i, col]

    # For each center, count number of data points associated with it that are smaller than the current threshold.
    # This information will be helpful for fast mistakes calculation, once a threshold pass a center.
    for i in range(k):
        left_centers_count[i] = 0

    # Initialize pointers to data points and centers sorted lists.
    ix = 0
    ic = 0

    # Initialize number of mistakes.
    mistakes = 0

    # Advance center index to the first valid one
    while valid_centers[centers_order[ic]] == 0:
        ic += 1

    # The first threshold is the value of the first valid center.
    threshold = centers[centers_order[ic], col]

    # The first threshold is a center.
    # All data points that are smaller or equal to the threshold will be accumulated prior to the main loop.
    is_center_threshold = 1

    # Advance data point index to the first valid center (which is the first valid threshold).
    # Count mistakes of points smaller than the first valid threshold
    while ix < n and X[data_order[ix], col] <= threshold:
        curr_center_idx = y[data_order[ix]] # Center of the current data point
        left_centers_count[curr_center_idx] += 1
        if centers[curr_center_idx, col] >= threshold:
            mistakes += 1
        ix += 1

    # Corner case.
    # Exactly n - 1 data points are on the left of the first center - this is a valid cut, but we won't enter the main loop.
    if ix == n - 1:
        # Recalculate the number of mistakes.
        # In this corner case all points except one are to the left of the current threshold.
        mistakes = 0
        ic = 0
        # Go over all valid centers.
        while ic < k:
            if valid_centers[ic] != 0:
                # If a center is to the right of the current threshold, then all of its point are considered as mistakes.
                # (Perhaps except to the last point that will be corrected later).
                if centers[ic, col] > threshold:
                    mistakes += centers_count[ic]
            ic += 1
        # Find the center of the single point that is on the right of the threshold.
        # If the center is also to the right of the current threshold, then remove one mistake.
        ic = y[data_order[n - 1]]
        if centers[ic, col] > threshold:
            mistakes -= 1
        # Update best cut.
        if mistakes < min_mistakes:
            valid_found = 1
            best_col = col
            best_threshold = threshold
            min_mistakes = mistakes

    # Main loop
    while ix < n - 1 and ic < k:

        # If threshold reached to the last valid center, the loop should end.
        if threshold >= max_val:
            break

        # In case this threshold is associated to a data point
        if is_center_threshold == 0:
            # Find data point center
            curr_center_idx = y[data_order[ix]]
            # Increase the count of points smaller than the threshold associated with the center
            left_centers_count[curr_center_idx] += 1

            # Update the mistakes count
            if centers[curr_center_idx, col] >= threshold:
                mistakes += 1
            elif centers[curr_center_idx, col] < threshold:
                mistakes -= 1

            # Move to the next data point index
            ix += 1

        # In case this threshold is associated to a center
        else:
            # Update mistakes count
            # left points are no longer mistakes
            # right points (which equal to total points - left points) are now mistakes
            mistakes += centers_count[centers_order[ic]] - 2 * left_centers_count[centers_order[ic]]

            # Move to the next center index
            ic += 1
            while valid_centers[centers_order[ic]] == 0 and ic < k:
                ic += 1

        prev_threshold = threshold

        # Find next threshold
        # in case of equality, data points arrive before centers in order to correctly find left count
        if X[data_order[ix], col] <= centers[centers_order[ic], col]:
            threshold = X[data_order[ix], col]
            is_center_threshold = 0
        else:
            threshold = centers[centers_order[ic], col]
            is_center_threshold = 1

        # Update best cut (only if the next threshold is not equal to the current one)
        if (prev_threshold != threshold) and (mistakes < min_mistakes):
            valid_found = 1
            best_threshold = prev_threshold
            min_mistakes = mistakes

    free(left_centers_count)

    if valid_found == 1:
        cols_thresholds[col] = best_threshold
        cols_mistakes[col] = min_mistakes
    else:
        cols_thresholds[col] = -1.0
        cols_mistakes[col] = -1


cdef struct Surrogate_Cut:
    int col
    float threshold
    NP_FLOAT_t cost
    int center_left
    int center_right


@cython.boundscheck(False)
@cython.wraparound(False)
def get_min_surrogate_cut(NP_FLOAT_t[:,:] X, NP_FLOAT_t[:,:] X_center_dot, NP_FLOAT_t[:] X_sum_all_center_dot, NP_FLOAT_t[:] centers_norm_sqr, int njobs):
    cdef int n = X.shape[0]
    cdef int k = X_center_dot.shape[1]
    cdef int d = X.shape[1]
    cdef int col
    cdef NP_FLOAT_t *thresholds = <NP_FLOAT_t *> malloc(d * sizeof(NP_FLOAT_t))
    cdef NP_FLOAT_t *costs = <NP_FLOAT_t *> malloc(d * sizeof(NP_FLOAT_t))
    cdef int *left_centers = <int *> malloc(d * sizeof(int))
    cdef int *right_centers = <int *> malloc(d * sizeof(int))
    cdef (NP_FLOAT_t, NP_FLOAT_t, int, int) cut
    cdef int best_col = -1
    cdef NP_FLOAT_t best_cost = INFINITY
    cdef NP_FLOAT_t best_threshold
    cdef int best_center_left
    cdef int best_center_right


    if njobs is None or njobs <= 1:
        # Iterate over valid coordinates
        for col in range(d):
            update_col_surrogate_cut(X, X_center_dot, centers_norm_sqr, X_sum_all_center_dot, n, d, k, col, thresholds, costs, left_centers, right_centers)
    else:
        # Iterate over valid coordinates
        for col in prange(d, nogil=True, num_threads=njobs):
            update_col_surrogate_cut(X, X_center_dot, centers_norm_sqr, X_sum_all_center_dot, n, d, k, col, thresholds, costs, left_centers, right_centers)

    for col in range(d):
        # This is a valid cut
        if left_centers[col] != -1:
            # This is a better cut
            if costs[col] < best_cost:
                best_col = col
                best_cost = costs[col]


    if best_col != -1:
        best_threshold = thresholds[best_col]
        best_center_left = left_centers[best_col]
        best_center_right = right_centers[best_col]

    free(thresholds)
    free(costs)
    free(left_centers)
    free(right_centers)

    if best_col == -1:
        return None
    else:
        return Surrogate_Cut(best_col, best_threshold, best_cost, best_center_left, best_center_right)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_col_surrogate_cut(NP_FLOAT_t[:,:] X, NP_FLOAT_t[:,:] X_center_dot, NP_FLOAT_t[:] centers_norm_sqr, NP_FLOAT_t[:] X_sum_all_center_dot, int n, int d, int k, int col, NP_FLOAT_t *thresholds, NP_FLOAT_t *costs, int *left_centers, int *right_centers) nogil:
    cdef int i
    cdef int ix
    cdef int ic
    cdef int n_left
    cdef int n_right
    cdef NP_FLOAT_t[:] curr_X_center_dot
    cdef NP_FLOAT_t *X_sum_left_center_dot = <NP_FLOAT_t *> malloc(k * sizeof(NP_FLOAT_t))
    cdef NP_FLOAT_t *X_sum_right_center_dot = <NP_FLOAT_t *> malloc(k * sizeof(NP_FLOAT_t))
    cdef NP_FLOAT_t cost
    cdef NP_FLOAT_t left_cost
    cdef NP_FLOAT_t right_cost
    cdef int left_center
    cdef int right_center
    cdef NP_FLOAT_t cur_total_cost
    cdef NP_FLOAT_t prev_threshold
    cdef NP_FLOAT_t threshold
    cdef np.int64_t[:] data_order
    cdef bint valid_found = 0
    cdef NP_FLOAT_t best_cost = INFINITY
    cdef NP_FLOAT_t best_threshold
    cdef int best_center_left
    cdef int best_center_right

    # Sort data points
    with gil:
        data_order = np.asarray(X[:,col]).argsort()

    ix = 0

    curr_X_center_dot = X_center_dot[data_order[0]]
    for i in range(k):
        X_sum_left_center_dot[i] = curr_X_center_dot[i]
        X_sum_right_center_dot[i] = X_sum_all_center_dot[i]
    for i in range(k):
        X_sum_right_center_dot[i] -= curr_X_center_dot[i]
    n_left = 1
    n_right = n - 1

    threshold = X[data_order[0], col]

    while ix < n - 1:

        ix += 1

        prev_threshold = threshold
        threshold = X[data_order[ix], col]

        if prev_threshold != threshold:

            left_cost = INFINITY
            right_cost = INFINITY
            for ic in range(k):
                cost = n_left * centers_norm_sqr[ic] - 2 * X_sum_left_center_dot[ic]
                if cost < left_cost:
                    left_cost = cost
                    left_center = ic
                cost = n_right * centers_norm_sqr[ic] - 2 * X_sum_right_center_dot[ic]
                if cost < right_cost:
                    right_cost = cost
                    right_center = ic

            cur_total_cost = left_cost +  right_cost

        # Add dot product of current vector and each center to the left
        # Subtract dot product of current vector and each center to the right
        curr_X_center_dot = X_center_dot[data_order[ix]]
        for i in range(k):
            X_sum_left_center_dot[i] = X_sum_left_center_dot[i] + curr_X_center_dot[i]
            X_sum_right_center_dot[i] = X_sum_right_center_dot[i] - curr_X_center_dot[i]

        n_left += 1
        n_right -= 1

        if (prev_threshold != threshold) and (cur_total_cost < best_cost):
            valid_found = 1
            best_threshold = prev_threshold
            best_cost = cur_total_cost
            best_center_left = left_center
            best_center_right = right_center


    free(X_sum_left_center_dot)
    free(X_sum_right_center_dot)

    if valid_found == 1:
        thresholds[col] = best_threshold
        costs[col] = best_cost
        left_centers[col] = best_center_left
        right_centers[col] = best_center_right
    else:
        thresholds[col] = -1.0
        costs[col] = -1.0
        left_centers[col] = -1
        right_centers[col] = -1
