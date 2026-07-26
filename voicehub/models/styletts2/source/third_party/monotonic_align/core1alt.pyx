"""
Adapted for readability from: [VITS](https://github.com/jaywalnut310/vits/blob/main/monotonic_align/core.pyx).
"""
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_each(
  int[:,::1] path,
  float[:,::1] value,
  int n_symbols,
  int n_frames,
  float max_neg_val
) nogil:
  cdef int n
  cdef int t
  cdef float v_prev
  cdef float v_cur

  # Let the log-likelihood be stored in an NxT maxtrix, where
  #   N is the number of symbols, and
  #   T is the number of frames.
  # We will loop over the row entries over a column
  # and update the accumulated score into the matrix
  # (in-place, to save space).
  #   n: symbol index. n \in {0, 1, ..., N-1}
  #   t: frame index.  t \in {0, 1, ..., T-1}
  # Apply the transition rules to the entries except for two impassible areas:
  #   1. below n - t > 0 ... left_boarder
  #   2. above n < t - (T - N) ... right_boarder
  for t in range(n_frames):

    right_boarder = t - (n_frames - n_symbols)
    for n in range(n_symbols):
      if n <= t and n >= right_boarder:
        # determine the score of the current state at t - 1
        if n == t:
          v_cur = max_neg_val
        else:
          v_cur = value[n, t-1]

        # determine the score of the previous state at t - 1
        if n == 0:
          if t == 0:
            v_prev = 0.
          else:
            v_prev = max_neg_val
        else:
          v_prev = value[n-1, t-1]

        # update the accumulated score (in-place)
        value[n, t] = max(v_cur, v_prev) + value[n, t]

      else:
        value[n, t] = max_neg_val

  backtrack(path, value, n_symbols - 1, n_frames - 1)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void backtrack(
  int[:, ::1] path,
  float[:, ::1] value,
  int n_last,
  int t_last,
) nogil:
  # If `t` is declared as a `cdef` integer type,
  # it will optimise this into a pure C loop.
  cdef int t
  for t in range(t_last, -1, -1):
    path[n_last, t] = 1
    if n_last != 0 and (
      n_last == t or
      value[n_last, t-1] < value[n_last-1, t-1]
    ):
      n_last = n_last - 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c1alt(
  int[:,:,::1] paths,
  float[:,:,::1] values,
  int[::1] t_xs,
  int[::1] t_ys,
  float max_neg_val=-1e9
) nogil:
  cdef int b = values.shape[0]

  cdef int i
  for i in prange(b, nogil=True):
    maximum_path_each(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)
