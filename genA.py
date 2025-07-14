import numpy as np
from scipy.optimize import root_scalar

def genA(delta, XI, U, kbar, y_ref, gammaInv):
    # XI : [y(k+1), y(k), y(k-1) u(k-1)] = [y_i^+, zeta_i]
    ZETAPLUS = XI[1:, 1:4]  
    ZETA     = XI[:-1, 1:4] 
    n = ZETA.shape[0]
    A = []

    # === Step 1: Initial reachable set A_delta^0 ===
    I0 = {'i': [], 'ri': []}

    for idx in range(n):
        yplus = XI[idx, 0]  
        dist_to_ref = np.linalg.norm(yplus - y_ref)
        if dist_to_ref < delta:
            I0['i'].append(idx)
            I0['ri'].append(delta - dist_to_ref)
    A.append(I0)

    # === Step 2: A_delta^1
    I1 = {'i': [], 'ri': []}
    for idxx in range(len(I0['i'])):
        idx = I0['i'][idxx]
        ri = gammaInv(I0['ri'][idxx])
        I1['i'].append(idx)
        I1['ri'].append(ri)
    A.append(I1)


    # === Step 3: k \ge 2
    for k in range(2, kbar + 1):
        I = {'i': [], 'ri': []}
        IPast = A[k - 1]
        nPast = len(IPast['i'])
        if nPast == 0:
            break
        for idx in range(n):
            zeta_next = ZETAPLUS[idx, :]
            ri = None

            for idxx in range(nPast):  # Loop over previous reachable points
                idx_past = IPast['i'][idxx]
                ri_past = IPast['ri'][idxx]
                zeta_past = ZETA[idx_past, :]

                dist = np.linalg.norm(zeta_next - zeta_past)

                if dist < ri_past:
                    diff = ri_past - dist
                    ri_candidate = gammaInv(diff)

                    if ri is None or ri_candidate > ri:
                        ri = ri_candidate

            if ri is not None:
                I['i'].append(idx)
                I['ri'].append(ri)
        A.append(I)

    return A