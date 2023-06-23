from numpy import diag_indices_from, clip, empty, inf

def d_metzler (A, separate=True)  :
    diag = diag_indices_from(A)
    Am = clip(A, 0, inf); Am[diag] = A[diag]
    An = A - Am
    if separate :
        return Am, An
    else :
        n = A.shape[0]
        ret = empty((2*n,2*n))
        ret[:n,:n] = Am; ret[n:,n:] = Am
        ret[:n,n:] = An; ret[n:,:n] = An
        return ret

def d_positive (B, separate=True) :
    Bp = clip(B, 0, inf); Bn = clip(B, -inf, 0)
    if separate :
        return Bp, Bn
    else :
        n,m = B.shape
        ret = empty((2*n,2*m))
        ret[:n,:m] = Bp; ret[n:,m:] = Bp
        ret[:n,m:] = Bn; ret[n:,:m] = Bn
        return ret