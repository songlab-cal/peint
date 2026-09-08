import numpy as np

def read_distance_map(
    distance_map_path: str,
) -> np.array:
    return np.loadtxt(distance_map_path, dtype=float)


def write_distance_map(
    distance_map: np.array,
    distance_map_path: str,
) -> None:
    np.savetxt(distance_map_path, distance_map, fmt="%.3f") # set floating point precision to 3 decimal points
