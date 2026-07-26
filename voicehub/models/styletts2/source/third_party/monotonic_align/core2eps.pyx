cimport cython
from cython.parallel import prange
# import numpy as np
# cimport numpy as np
# from numpy.math cimport logaddexp


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_each2(int[:,::1] path, float[:,::1] value, int n_symbols, int n_frames, float max_neg_val) nogil:
  cdef int x
  cdef int y

  cdef float v_prev_symbol
  cdef float v_prev_state
  cdef float v_cur
  cdef float tmp
  cdef int index = n_symbols - 1

  cdef int frame_index
  cdef int symbol_index
  cdef float stay = 0.0
  cdef float unit_step = 0.0
  cdef float skip_eps_transition = 0.0
  cdef float max_ = 0.0
  cdef float max_val = 0.0

  cdef float step_val = 0.0



  # dynamic programming
  for frame_index in range(n_frames):
    # lower = max(0, n_symbols + frame_index - n_frames)
    # higher = min(n_symbols, frame_index + 1)
    # head =
    trail = frame_index - (n_frames - n_symbols)  # i >= trail: is valid
    for symbol_index in range(n_symbols):
      # if lower <= symbol_index < symbol_index:
      if symbol_index <= frame_index and symbol_index >= trail:

        # w/o an auxiliary matrix (one for accumulation and one for raw score value)
        # the number of visited symbols must be < frame index
        if symbol_index >= frame_index:
          v_cur = max_neg_val
        else:
          v_cur = value[symbol_index, frame_index - 1]

        # Corner cases at the start
          # score of the 1st row is 0, meaning that
          # a valid path can start from any frame at the 1st symbol.
          # score of all other symbols cannot be the start. (score = -inf)
        # ordination cases
        if symbol_index == 0:
          if frame_index == 0:
            v_prev_state = 0.
          else:
            v_prev_state = max_neg_val
        else:
          v_prev_state = value[symbol_index-1, frame_index-1]


        # Extra: v_prev_symbol
        # regular/initial state of a phone
        # epsilon state
        if symbol_index >= 2 and symbol_index % 2 == 0:
          v_prev_symbol = value[symbol_index - 2, frame_index - 1]
        else:
          v_prev_symbol = max_neg_val


        # step_val = logaddexp(v_prev_symbol, v_prev_symbol)

        # value after transition
        step_val = value[symbol_index, frame_index]
        stay = v_cur + step_val
        unit_step = v_prev_state + step_val
        max_ = max(stay, unit_step)
        if symbol_index >= 2:
          skip_eps_transition = v_prev_symbol + step_val

          if skip_eps_transition > max_ and symbol_index % 2 == 0:
            value[symbol_index, frame_index] = skip_eps_transition
          else:
            if unit_step > stay:
              value[symbol_index, frame_index] = unit_step
            else:
              value[symbol_index, frame_index] = stay
        else:
          if unit_step > stay:
            value[symbol_index, frame_index] = unit_step
          else:
            value[symbol_index, frame_index] = stay

      else:
        value[symbol_index, frame_index] = max_neg_val

      # if unit_step > stay:
      #   value[symbol_index, frame_index] = unit_step
      # else:
      #   value[symbol_index, frame_index] = stay



      #   value[symbol_index, frame_index] = value[symbol_index, frame_index] + max(max_, v_prev_symbol / nrm5)

      # else:
      #   value[symbol_index, frame_index] = value[symbol_index, frame_index] + max_


  # backtracking (indicator matrix)
  for frame_index in range(n_frames - 1, -1, -1):
    path[index, frame_index] = 1

    # must transition due to monotonic constraint
    if index == frame_index:
      index = index - 1

    if index != 0:
      if frame_index > 0:
        stay = value[index, frame_index - 1]
        unit_step = value[index - 1, frame_index - 1]

        max_ = max(stay, unit_step) # / nrm2)
        if (index >= 2) and (index % 2 == 0):
          skip_eps_transition = value[index - 2, frame_index - 1] #/ nrm5
          max_val = max(max_, skip_eps_transition)
          if skip_eps_transition == max_val:
            index = index - 2
          else:
            if unit_step == max_:
              index = index - 1
        else:
          if unit_step == max_:
            index = index - 1

        # if unit_step == max_:
        #   index = index - 1



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c2eps(int[:,:,::1] paths, float[:,:,::1] values, int[::1] t_xs, int[::1] t_ys, float max_neg_val=-1e9) nogil:
  cdef int b = values.shape[0]

  cdef int i
  for i in prange(b, nogil=True):
    maximum_path_each2(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)
