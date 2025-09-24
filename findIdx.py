import numpy as np

def findIdx(curr_state, ZETA, XI, A):
    # --- Flag indicating whether a match is found ---
    matched = False   
    # --- Index of the matching candidate ---    
    idx_match = None  

    # --- Loop over all A_\delta^j and find idx
    for a in A[1:]:
        # --- Track the best match distance
        min_dist = np.inf  

        # --- Compare distance between curr_state and center with radius
        for idx_candidate, radius in zip(a['i'], a['ri']):
            center = ZETA[idx_candidate, :]        
            dist = np.linalg.norm(center - curr_state)  

            # --- Check if current state lies inside this ball ---
            if dist <= radius and dist < min_dist:
                matched = True
                min_dist = abs(XI[idx_candidate, 0])
                idx_match = idx_candidate

        # --- If a match was found in this set, stop searching
        if matched:
            break

    return matched, idx_match
