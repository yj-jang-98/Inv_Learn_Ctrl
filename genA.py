import numpy as np
from scipy.optimize import root_scalar

def genA(delta, XI, U, kbar, y_ref, gammaInv):
    # --- Zeta shifted forward (zeta_{i+1}) ---
    ZETAPLUS = XI[1:, 1:]   
    # --- Zeta shifted forward (zeta_{i}) ---
    ZETA     = XI[:-1, 1:]
    n = ZETA.shape[0]

    A = [] 

    # === Step 1: Generate A_delta^0 ===
    A0 = {'i': [], 'ri': []}
    for idx in range(n):
        yplus = XI[idx, 0]                        
        dist_to_ref = np.linalg.norm(yplus - y_ref)
        if dist_to_ref < delta:
            A0['i'].append(idx)
            A0['ri'].append(delta - dist_to_ref)  
    A.append(A0)

    # === Step 2: Generate A_delta^1 ===
    A1 = {'i': [], 'ri': []}
    for idx, r0 in zip(A0['i'], A0['ri']):
        r1 = gammaInv(r0)   
        A1['i'].append(idx)
        A1['ri'].append(r1)
    A.append(A1)

    # === Step 3: Generate A_delta^j, j >= 2 ===
    for k in range(2, kbar + 1):
        Ak = {'i': [], 'ri': []}
        A_prev = A[k - 1]
        if not A_prev['i']:  
            break

        # --- Test reachability for each Zeta ---
        for idx in range(n):
            zeta_next = ZETAPLUS[idx, :]
            best_radius = None

            for idx_prev, r_prev in zip(A_prev['i'], A_prev['ri']):
                zeta_prev = ZETA[idx_prev, :]
                dist = np.linalg.norm(zeta_next - zeta_prev)

                if dist < r_prev:
                    diff = r_prev - dist
                    r_candidate = gammaInv(diff)

                    # --- Keep the largest radius if multiple apply ---
                    if best_radius is None or r_candidate > best_radius:
                        best_radius = r_candidate

            # --- If reachable, record candidate ---
            if best_radius is not None:
                Ak['i'].append(idx)
                Ak['ri'].append(best_radius)

        A.append(Ak)

    return A