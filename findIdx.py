import numpy as np

def findIdx(curr_state, ZETA, XI, A):
    matched = False
    idx_match = None
    for a in A[1:]:
        min_dist = np.inf
        for idx_candidate, radius in zip(a['i'], a['ri']):
            center = ZETA[idx_candidate, :]
            dist = np.linalg.norm(center - curr_state)
            if dist <= radius and dist < min_dist:
                matched = True
                min_dist = abs(XI[idx_candidate,0])
                idx_match = idx_candidate
        if matched:
            break
    return matched, idx_match

def findNear(curr_state, ZETA):
    min_dist = np.inf
    for idx_candidate in range(ZETA.shape[0]):
        center = ZETA[idx_candidate, :]
        dist = np.linalg.norm(center - curr_state)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = idx_candidate
    return nearest_idx