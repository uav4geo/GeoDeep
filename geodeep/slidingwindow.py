# Based on https://github.dev/adamrehn/slidingwindow
# MIT (c) 2017 Adam Rehn
# This version has been modified and is released under AGPL
import math
from rasterio.windows import Window

def generate_for_size(width, height, max_window_size, overlap_percent, clip=True):
	# Create square windows unless an explicit width or height has been specified
    window_size_x = max_window_size
    window_size_y = max_window_size

    # If the input data is smaller than the specified window size,
    # clip the window size to the input size on both dimensions
    if clip:
        window_size_x = min(window_size_x, width)
        window_size_y = min(window_size_y, height)

    # Compute the window overlap and step size
    window_overlap_x = int(math.floor(window_size_x * overlap_percent))
    window_overlap_y = int(math.floor(window_size_y * overlap_percent))
    step_size_x = window_size_x - window_overlap_x
    step_size_y = window_size_y - window_overlap_y

    # Determine how many windows we will need in order to cover the input data
    last_x = width - window_size_x
    last_y = height - window_size_y
    x_offsets = list(range(0, last_x + 1, step_size_x))
    y_offsets = list(range(0, last_y + 1, step_size_y))

    # Unless the input data dimensions are exact multiples of the step size,
    # we will need one additional row and column of windows to get 100% coverage
    if len(x_offsets) == 0 or x_offsets[-1] != last_x:
        x_offsets.append(last_x)
    if len(y_offsets) == 0 or y_offsets[-1] != last_y:
        y_offsets.append(last_y)

    # Generate the list of windows
    windows = []
    for x_offset in x_offsets:
        for y_offset in y_offsets:
                windows.append(Window(
                    x_offset,
                    y_offset,
                    window_size_x,
                    window_size_y,
                ))

    return windows
