import numpy as np

def is_square_ordered_matrix(m: np.ndarray) -> bool:
    if len(m.shape) != 2:
        return False
    
    if m.shape[0] != m.shape[1]:
        return False
    
    min_val = (np.zeros(m.shape[0]) + m.min() - 1).reshape(1, m.shape[0])
    up_m = np.concatenate([m[1:,:], min_val], axis=0)
    left_m = np.concatenate([m[:,1:], min_val.transpose()], axis=1)

    return (m>up_m).all() and (m>left_m).all()

def diamond(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(1).reshape((1,1))
    
    straight_diagonal = divmod(np.eye(n) + 1, 2)[1]
    reversed_diagonal = np.flip(straight_diagonal, axis=0)
    upper_half = np.concatenate([reversed_diagonal[:,0:-1], straight_diagonal], axis=1)
    lower_half = np.concatenate([straight_diagonal[:, 0:-1], reversed_diagonal], axis=1)
    return np.concatenate([upper_half[0:-1,:], lower_half], axis=0)

def multiplication_table(n: int) -> np.ndarray:
    m = np.arange(n)
    cols_m = np.tile(m, (n, 1))
    rows_m = np.tile(m.reshape((n, 1)), (n))
    return cols_m * rows_m
