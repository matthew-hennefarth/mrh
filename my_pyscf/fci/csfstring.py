import numpy as np
import sys, os, time
import ctypes
from mrh.my_pyscf.fci import csdstring
from pyscf.fci import cistring
from pyscf.fci.spin_op import spin_square0
from pyscf.lib import numpy_helper
from scipy import special, linalg
from mrh.util.io import prettyprint_ndarray
from functools import reduce
from mrh.lib.helper import load_library
libcsf = load_library ('libcsf')

def check_spinstate_norm (detarr, norb, neleca, nelecb, smult, csd_mask=None):
    ''' Calculate the norm of the given CI vector projected onto spin-state smult (= 2S+1) '''
    return transform_civec_det2csf (detarr, norb, neleca, nelecb, smult, csd_mask=csd_mask)[1]

def project_civec_csf (detarr, norb, neleca, nelecb, smult, csd_mask=None):
    ''' Project the total spin = s [= (smult-1) / 2] component of a CI vector using CSFs 

    Args
    detarr: 2d ndarray of shape (ndeta,ndetb)
        ndeta = norb choose neleca
        ndetb = norb choose nelecb
    norb, neleca, nelecb, smult: ints

    Returns
    detarr: ndarray of unchanged shape
        Normalized CI vector in terms of determinants
    detnorm: float
    '''

    #detarr = transform_civec_det2csf (detarr, norb, neleca, nelecb, smult, csd_mask=csd_mask)
    #detarr = transform_civec_csf2det (detarr, norb, neleca, nelecb, smult, csd_mask=csd_mask)
    #return detarr
    ndeta = special.comb (norb, neleca, exact=True)
    ndetb = special.comb (norb, nelecb, exact=True)
    ndet = ndeta*ndetb
    assert (detarr.shape == tuple((ndeta,ndetb)) or detarr.shape == tuple((ndet,)))
    detarr = np.ravel (detarr, order='C')
        
    detarr = _transform_detcsf_vec_or_mat (detarr, norb, neleca, nelecb, smult, reverse=False, op_matrix=False, csd_mask=csd_mask, project=True)
    try:
        detnorm = linalg.norm (detarr)
    except Exception as ex:
        assert (detarr.shape == tuple((1,))), "{} {}".format (detarr.shape, ex)
        detnorm = detarr[0];
    '''
    if np.isclose (detnorm, 0):
        raise RuntimeWarning (('CI vector projected into CSF space (norb, na, nb, s = {}, {}, {}, {})'
            ' has zero norm; did you project onto a different spin before?').format (norb, neleca, nelecb, (smult-1)/2))
    '''
    return detarr / detnorm, detnorm

def transform_civec_det2csf (detarr, norb, neleca, nelecb, smult, csd_mask=None):
    ''' Express CI vector in terms of CSFs for spin s

    Args
    detarr: ndarray of shape 
        ndet = (norb choose neleca) * (norb choose nelecb)
    norb, neleca, nelecb, smult: ints

    Returns
    csfarr: contiguous ndarray of shape (ncsf) or (ncsf, ncsf) where ncsf < ndet
        Normalized CI vector in terms of CSFs
    csfnorm: float
    '''
 
    ndeta = special.comb (norb, neleca, exact=True)
    ndetb = special.comb (norb, nelecb, exact=True)
    ndet = ndeta*ndetb
    assert (detarr.shape == tuple((ndeta,ndetb)) or detarr.shape == tuple((ndet,)))
    detarr = np.ravel (detarr, order='C')
        
    csfarr = _transform_detcsf_vec_or_mat (detarr, norb, neleca, nelecb, smult, reverse=False, op_matrix=False, csd_mask=csd_mask, project=False)
    try:
        csfnorm = linalg.norm (csfarr)
    except Exception as ex:
        if csfarr.shape == (0,):
            return np.zeros (0, dtype=detarr.dtype), 0
        assert (csfarr.shape == tuple((1,))), "{} {}".format (csfarr.shape, ex)
        csfnorm = csfarr[0];
    '''
    if np.isclose (csfnorm, 0):
        raise RuntimeWarning (('CI vector projected into CSF space (norb, na, nb, s = {}, {}, {}, {})'
            ' has zero norm; did you project onto a different spin before?').format (norb, neleca, nelecb, (smult-1)/2))
    '''
    return csfarr / csfnorm, csfnorm

def transform_civec_csf2det (csfarr, norb, neleca, nelecb, smult, csd_mask=None):
    ''' Transform CI vector in terms of CSFs back into determinants

    Args
    csfarr: ndarray of shape (ncsf) or (ncsf, ncsf)
    norb, neleca, nelecb, smult: ints

    Returns
    detarr: contiguous ndarray of shape (ndeta,ndetb)
        Normalized CI vector in terms of determinants
    detnorm: float
    '''

    ndeta = special.comb (norb, neleca, exact=True) 
    ndetb = special.comb (norb, nelecb, exact=True)
    if csfarr.shape == (0,):
        return np.zeros ((ndeta, ndetb), dtype=csfarr.dtype), 0
    detarr = _transform_detcsf_vec_or_mat (csfarr, norb, neleca, nelecb, smult, reverse=True, op_matrix=False, csd_mask=csd_mask, project=False)
    detnorm = linalg.norm (detarr)
    return detarr.reshape (ndeta, ndetb) / detnorm, detnorm

def transform_opmat_det2csf (detarr, norb, neleca, nelecb, smult, csd_mask=None):
    ''' Express operator matrix in terms of CSFs for spin s

    Args
    detarr: ndarray of shape (ndet, ndet) or (ndet**2,)
        ndet = (norb choose neleca) * (norb choose nelecb)
    norb, neleca, nelecb, smult: ints

    Returns
    csfarr: contiguous ndarray of shape (ncsf, ncsf) where ncsf < ndet
        Operator matrix in terms of csfs
    '''

    ndeta = special.comb (norb, neleca, exact=True)
    ndetb = special.comb (norb, nelecb, exact=True)
    ndet = ndeta*ndetb
    assert (detarr.shape == tuple((ndet,ndet)) or detarr.shape == tuple((ndet**2,)))
    csfarr = _transform_detcsf_vec_or_mat (detarr, norb, neleca, nelecb, smult, reverse=False, op_matrix=True, csd_mask=csd_mask, project=False)
    return csfarr

def _transform_detcsf_vec_or_mat (arr, norb, neleca, nelecb, smult, reverse=False, op_matrix=False, csd_mask=None, project=False):
    ''' Wrapper to manipulate array into correct shape and transform both dimensions if an operator matrix '''

    ndeta = special.comb (norb, neleca, exact=True) 
    ndetb = special.comb (norb, nelecb, exact=True)
    ndet_all = ndeta*ndetb
    ncsf_all = count_all_csfs (norb, neleca, nelecb, smult)

    ncol = ncsf_all if reverse else ndet_all
    if op_matrix:
        assert (arr.shape == tuple((ncol, ncol))), "array shape should be ({0}, {0}); is {1}".format (ncol, arr.shape)
    else:
        assert (arr.shape == tuple((ncol, ))), "array shape should be ({0},); is {1}".format (ncol, arr.shape)
        arr = arr[np.newaxis,:]

    arr = _transform_det2csf (arr, norb, neleca, nelecb, smult, reverse=reverse, csd_mask=csd_mask, project=project)
    if op_matrix:
        arr = arr.T
        arr = _transform_det2csf (arr, norb, neleca, nelecb, smult, reverse=reverse, csd_mask=csd_mask, project=project)
        arr = numpy_helper.transpose (arr, inplace=True)

    ncol = ndet_all if (reverse or project) else ncsf_all
    if op_matrix:
        assert (arr.shape == tuple((ncol, ncol))), "array shape should be ({0}, {0}); is {1}".format (ncol, arr.shape)
    else:
        assert (arr.shape == tuple((1, ncol))), "array shape should be (1,{0}); is {1}".format (ncol, arr.shape)
        arr = arr[0,:]


    return arr

    
def _transform_det2csf (inparr, norb, neleca, nelecb, smult, reverse=False, csd_mask=None, project=False):
    ''' Must take an array of shape (*, ndet) or (*, ncsf) '''
    t_start = time.time ()
    time_umat = 0
    time_mult = 0
    time_getdet = 0
    size_umat = 0
    s = (smult - 1) / 2
    ms = (neleca - nelecb) / 2

    min_npair, npair_csd_offset, npair_dconf_size, npair_sconf_size, npair_sdet_size = csdstring.get_csdaddrs_shape (norb, neleca, nelecb)
    _, npair_csf_offset, _, _, npair_csf_size = get_csfvec_shape (norb, neleca, nelecb, smult)
    nrow = inparr.shape[0]
    ndeta_all = special.comb (norb, neleca, exact=True)
    ndetb_all = special.comb (norb, nelecb, exact=True)
    ndet_all = ndeta_all * ndetb_all
    ncsf_all = count_all_csfs (norb, neleca, nelecb, smult)

    ncol_out = (ncsf_all, ndet_all)[reverse or project]
    ncol_in = (ncsf_all, ndet_all)[~reverse or project]
    if not project:
        outarr = np.ascontiguousarray (np.zeros ((nrow, ncol_out), dtype=np.float_))
        csf_addrs = np.zeros (ncsf_all, dtype=np.bool_)
    # Initialization is necessary because not all determinants have a csf for all spin states

    #max_npair = min (nelecb, (neleca + nelecb - int (round (2*s))) // 2)
    max_npair = nelecb
    for npair in range (min_npair, max_npair+1):
        ipair = npair - min_npair
        ncsf = npair_csf_size[ipair]
        nspin = neleca + nelecb - 2*npair
        nconf = npair_dconf_size[ipair] * npair_sconf_size[ipair]
        ndet = npair_sdet_size[ipair]
        csf_offset = npair_csf_offset[ipair]
        csd_offset = npair_csd_offset[ipair]
        if (ncsf == 0) and not project:
            continue
        if not project:
            csf_addrs[:] = False
            csf_addrs_ipair = csf_addrs[csf_offset:][:nconf*ncsf].reshape (nconf, ncsf) # Note: this is a view, i.e., a pointer

        t_ref = time.time ()
        if csd_mask is None:
            det_addrs = csdstring.get_nspin_dets (norb, neleca, nelecb, nspin)
        else:
            det_addrs = csd_mask[csd_offset:][:nconf*ndet].reshape (nconf, ndet, order='C')
        assert (det_addrs.shape[0] == nconf)
        assert (det_addrs.shape[1] == ndet)
        time_getdet += time.time () - t_ref

        if (ncsf == 0):
            inparr[:,det_addrs] = 0 
            continue

        t_ref = time.time ()
        umat = np.asarray_chkfinite (get_spin_evecs (nspin, neleca, nelecb, smult))
        size_umat = max (size_umat, umat.nbytes)
        ncsf_blk = ncsf # later on I can use this variable to implement a generator form of get_spin_evecs to save memory when there are too many csfs
        assert (umat.shape[0] == ndet)
        assert (umat.shape[1] == ncsf_blk)
        if project:
            Pmat = np.dot (umat, umat.T)
        time_umat += time.time () - t_ref

        if not project:
            csf_addrs_ipair[:,:ncsf_blk] = True # Note: edits csf_addrs

        # The elements of csf_addrs and det_addrs are addresses for the flattened vectors and matrices (inparr.flat and outarr.flat)
        # Passing them unflattened as indices of the flattened arrays should result in a 3-dimensional array if I understand numpy's indexing rules correctly
        # For the lvalues, I think it's necessary to flatten csf_addrs and det_addrs to avoid an exception
        # Hopefully this is parallel under the hood, and hopefully the OpenMP reduction epsilon doesn't ruin the spin eigenvectors
        t_ref = time.time ()
        if project:
            inparr[:,det_addrs] = np.tensordot (inparr[:,det_addrs], Pmat, axes=1)
        elif not reverse:
            outarr[:,csf_addrs] = np.tensordot (inparr[:,det_addrs], umat, axes=1).reshape (nrow, ncsf_blk*nconf)
        else:
            outarr[:,det_addrs] = np.tensordot (inparr[:,csf_addrs].reshape (nrow, nconf, ncsf_blk), umat, axes=((2,),(1,)))
        time_mult += time.time () - t_ref

    if project:
        outarr = inparr
    else:
        outarr = outarr.reshape (nrow, ncol_out)
    d = ['determinants','csfs']
    '''
    print (('Transforming {} into {} summary: {:.2f} seconds to get determinants,'
            ' {:.2f} seconds to build umat, {:.2f} seconds matrix-vector multiplication,'
            ' {:.2f} MB largest umat').format (d[reverse], d[~reverse], time_getdet, time_umat,
            time_mult, size_umat / 1e6))
    print ('Total time spend in _transform_det2csf: {:.2f} seconds'.format (time.time () - t_start))
    '''
    return outarr


    
def csf_gentable (nspin, smult):
    ''' Example of a genealogical coupling table for 8 spins and s = 1 (triplet), counting from the final state
        back to the null state:

           28 28 19 10  4  1  .
            |  9  9  6  3  1  .
            |  |  3  3  2  1  .
            |  |  |  1  1  t  .
                        .  .  .
                           .  .
                              .

        Top left (0,0) is the null state (nspin = 0, s = 0).
        Position (3,5) (1/2 nspin - s, 1/2 nspin + s) is the target state (nspin=8, s=1).
        Numbers count paths from that position to the target state, moving only down or to the right; gen(0,0)=gen(0,1) is the total number of CSFs with this spin state
        for any electron configuration with 8 unpaired electrons.
        Moving left corresponds to bit=1; moving down corresponds to bit=0.
        Vertical lines are not defined (s<0) but stored as zero [so array has shape=(1/2 nspin - s + 1 , 1/2 nspin + s + 1)].
        Dots are not stored but are defined as zero.
        Rotate 45 degrees counter clockwise and nspins is on the horizontal from left to right, and spin is on the vertical from bottom to top.
        To compute the address from a string, sum the numbers which appear above and to the right of every coordinate reached
        from above. For example, the string 01101101 turns down in the second, fifth, and eighth places (counting from right to left) so its
        address is 19 (0,2 [0+2=2]) + 3 (1,4 [1+4=5]) + 0 (2,6[2+6=8]) = 22.
        To compute the string from the address, find the largest number on each row that's less than the address, subtract it,
        set every unassignd bit up to that column to 1, set the bit in that column to 0, go down a row, reset the column index to zero, and repeat.
        For example, address 15 is 10 (0,3) + 3 (1,4) + 2 (2,4), so the string is 11001011
        so the string is 1100101, again indexing the bits from right to left.
    '''
        
    assert ((smult - 1) % 2 == nspin % 2), "{} {}".format (smult, nspin)
    if smult > nspin+1:
        return np.zeros ((1,1), dtype=np.int32)

    n0 = (nspin - (smult - 1)) // 2
    n1 = (nspin + (smult - 1)) // 2

    gentable = np.zeros ((n0+1,n1+1), dtype=np.int32)
    gentable[n0,n0:] = 1
    for i0 in range (n0, 0, -1):
        row = gentable[i0,i0:]
        row = [sum (row[i1:]) for i1 in range (len (row))]
        row = [row[0]] + list (row)
        gentable[i0-1,i0-1:] = row
    return gentable

def count_csfs (nspin, smult):
    return csf_gentable (nspin, smult)[0,0]

def count_all_csfs (norb, neleca, nelecb, smult):
    a,b,c = get_csfvec_shape (norb, neleca, nelecb, smult)[-3:]
    return np.sum (a*b*c) 

def get_csfvec_shape (norb, neleca, nelecb, smult):
    ''' For a system of neleca + nelecb electrons with MS = (neleca - nelecb) occupying norb orbitals,
        get shape information about the irregular CI vector array in terms of csfs (number of pairs, pair config, unpair config, coupling string)

        Args:
        norb, neleca, nelecb are integers

        Returns:
        min_npair, integer, the lowest possible number of electron pairs
        npair_offset, 1d ndarray of integers
            npair_offset[i] points to the first determinant of a csdaddrs-sorted CI vector with i+min_npair electron pairs
        npair_dconf_size, 1d ndarray of integers
            npair_dconf_size[i] = number of pair configurations with i+min_npair electron pairs
        npair_sconf_size, 1d ndarray of integers
            npair_sconf_size[i] = number of unpaired electron configurations for a system of neleca+nelecb electrons with npair paired
        npair_csf_size, 1d ndarray of integers
            npair_csf_size[i] = number of coupling vectors leading to spin = s for neleca+nelecb - 2*npair spins
    '''
    s = (smult - 1) / 2
    ms = (neleca - nelecb) / 2
    assert (neleca >= nelecb)
    assert (neleca - nelecb <= smult - 1), "Impossible quantum numbers: s = {}; ms = {}".format (s, ms)
    min_npair = max (0, neleca + nelecb - norb)
    nspins = [neleca + nelecb - 2*npair for npair in range (min_npair, nelecb+1)]
    nfreeorbs = [norb - npair for npair in range (min_npair, nelecb+1)]
    nas = [(nspin + neleca - nelecb) // 2 for nspin in nspins]
    for nspin in nspins:
        assert ((nspin + neleca - nelecb) % 2 == 0)

    npair_dconf_size = np.asarray ([special.comb (norb, npair, exact=True) for npair in range (min_npair, nelecb+1)], dtype=np.int32)
    npair_sconf_size = np.asarray ([special.comb (nfreeorb, nspin, exact=True) for nfreeorb, nspin in zip (nfreeorbs, nspins)], dtype=np.int32)
    npair_csf_size = np.asarray ([count_csfs (nspin, smult) for nspin in nspins]) 

    npair_sizes = np.asarray ([0] + [i * j * k for i,j,k in zip (npair_dconf_size, npair_sconf_size, npair_csf_size)], dtype=np.int32)
    npair_offset = np.asarray ([np.sum (npair_sizes[:i+1]) for i in range (len (npair_sizes))], dtype=np.int32)
    ndeta, ndetb = (special.comb (norb, n, exact=True) for n in (neleca, nelecb))
    assert (npair_offset[-1] <= ndeta*ndetb), "{} determinants and {} csfs".format (ndeta*ndetb, npair_offset[-1])

    return min_npair, npair_offset[:-1], npair_dconf_size, npair_sconf_size, npair_csf_size

def get_spin_evecs (nspin, neleca, nelecb, smult):
    ms = (neleca - nelecb) / 2
    s = (smult - 1) / 2
    assert (neleca >= nelecb)
    assert (neleca - nelecb <= smult - 1)
    assert (neleca - nelecb <= nspin)
    assert ((neleca - nelecb) % 2 == (smult-1) % 2)
    assert ((neleca - nelecb) % 2 == nspin % 2)

    na = (nspin + neleca - nelecb) // 2
    ndet = special.comb (nspin, na, exact=True)
    ncsf = count_csfs (nspin, smult)

    t_start = time.time ()
    spinstrs = cistring.addrs2str (nspin, na, list (range (ndet)))
    assert (len (spinstrs) == ndet), "should have {} spin strings; have {} (nspin={}, ms={}".format (ndet, len (spinstrs), nspin, ms)

    t_start = time.time ()
    scstrs = addrs2str (nspin, smult, list (range (ncsf)))
    assert (len (scstrs) == ncsf), "should have {} coupling strings; have {} (nspin={}, s={})".format (ncsf, len (scstrs), nspin, s)

    umat = np.ones ((ndet, ncsf), dtype=np.float_)
    twoS = smult-1
    twoMS = neleca - nelecb
    
    t_start = time.time ()
    libcsf.FCICSFmakecsf (umat.ctypes.data_as (ctypes.c_void_p),
                        spinstrs.ctypes.data_as (ctypes.c_void_p),
                        scstrs.ctypes.data_as (ctypes.c_void_p),
                        ctypes.c_int (nspin),
                        ctypes.c_int (ndet),
                        ctypes.c_int (ncsf),
                        ctypes.c_int (twoS),
                        ctypes.c_int (twoMS))

    return umat

def test_spin_evecs (nspin, neleca, nelecb, smult, S2mat=None):
    s = (smult - 1) / 2
    ms = (neleca - nelecb) / 2
    assert (ms <= s)
    assert (smult-1 <= nspin)
    assert (nspin >= neleca + nelecb)

    na = (nspin + neleca - nelecb) // 2
    ndet = special.comb (nspin, na, exact=True)
    ncsf = count_csfs (nspin, smult)

    spinstrs = cistring.addrs2str (nspin, na, list (range (ndet)))

    if S2mat is None:
        S2mat = np.zeros ((ndet, ndet), dtype=np.float_)
        twoS = smult - 1
        twoMS = int (round (2 * ms))

        t_start = time.time ()    
        libcsf.FCICSFmakeS2mat (S2mat.ctypes.data_as (ctypes.c_void_p),
                             spinstrs.ctypes.data_as (ctypes.c_void_p),
                             ctypes.c_int (ndet),
                             ctypes.c_int (nspin),
                             ctypes.c_int (twoS),
                             ctypes.c_int (twoMS))
        print ("TIME: {} seconds to make S2mat for {} spins with s={}, ms={}".format (
            time.time() - t_start, nspin, (smult-1)/2, ms))
        print ("MEMORY: {} MB for {}-spin S2 matrix with s={}, ms={}".format (S2mat.nbytes / 1e6,
            nspin, (smult-1)/2, ms))

    umat = get_spin_evecs (nspin, neleca, nelecb, smult)
    print ("MEMORY: {} MB for {}-spin csfs with s={}, ms={}".format (umat.nbytes / 1e6,
        nspin, (smult-1)/2, ms))
    assert (umat.shape == tuple((ndet, ncsf))), "umat shape should be ({},{}); is {}".format (ndet, ncsf, umat.shape)
    
    s = (smult-1)/2
    t_start = time.time ()
    isorth = np.allclose (np.dot (umat.T, umat), np.eye (umat.shape[1]))
    ortherr = linalg.norm (np.dot (umat.T, umat) - np.eye (umat.shape[1]))
    S2mat_csf = reduce (np.dot, (umat.T, S2mat, umat))
    S2mat_csf_comp = s * (s+1) * np.eye (umat.shape[1])
    S2mat_csf_err = linalg.norm (S2mat_csf - S2mat_csf_comp)
    diagsS2 = np.allclose (S2mat_csf, S2mat_csf_comp)
    passed = isorth and diagsS2
    print ("TIME: {} seconds to analyze umat for {} spins with s={}, ms={}".format (
        time.time() - t_start, nspin, s, ms))
    

    print (('For a system of {} spins with total spin {} and spin projection {}'
            ', {} CSFs found from {} determinants by Clebsch-Gordan algorithm').format (
            nspin, s, ms, umat.shape[1], len (spinstrs)))
    print ('Did the Clebsch-Gordan algorithm give orthonormal eigenvectors? {}'.format (
        ('NO (err = {})'.format (ortherr), 'Yes')[isorth]))
    print ('Did the Clebsch-Gordan algorithm diagonalize the S2 matrix with the correct eigenvalues? {}'.format (
        ('NO (err = {})'.format (S2mat_csf_err), 'Yes')[diagsS2]))
    print ('nspin = {}, S = {}, MS = {}: {}'.format (nspin, s, ms, ('FAILED','Passed')[passed]))
    sys.stdout.flush ()

    return umat, S2mat

def get_scstrs (nspin, smult):
    ''' This is not a great way to do this, but I seriously can't think of any straightforward way to put the coupling strings in order... '''
    if (smult >= nspin):
        return np.ones ((0), dtype=np.int64)
    elif (nspin == 0):
        return np.zeros ((1), dtype=np.int64)
    assert (int (round (smult + nspin)) % 2 == 1), "npsin = {}; 2S+1 = {}".format (nspin, smult)
    nup = (nspin + smult - 1) // 2
    ndet = int (special.comb (nspin, nup))
    scstrs = cistring.addrs2str (nspin, nup, list (range (ndet)))
    mask = np.ones (len (scstrs), dtype=np.bool_)

    libcsf.FCICSFgetscstrs (scstrs.ctypes.data_as (ctypes.c_void_p),
                            mask.ctypes.data_as (ctypes.c_void_p),
                            ctypes.c_int (len (scstrs)),
                            ctypes.c_int (nspin))

    return np.ascontiguousarray (scstrs[mask], dtype=np.int64)


def addrs2str (nspin, smult, addrs):
    addrs = np.ascontiguousarray (addrs, dtype=np.int32)
    nstr = len (addrs)
    strs = np.ascontiguousarray (np.zeros (nstr, dtype=np.int64))
    gentable = np.ravel (csf_gentable (nspin, smult))
    twoS = smult - 1
    libcsf.FCICSFaddrs2str (strs.ctypes.data_as (ctypes.c_void_p),
                            addrs.ctypes.data_as (ctypes.c_void_p),
                            ctypes.c_int (nstr),
                            gentable.ctypes.data_as (ctypes.c_void_p),
                            ctypes.c_int (nspin),
                            ctypes.c_int (twoS));

    return strs

def strs2addr (nspin, smult, strs):
    strs = np.ascontiguousarray (strs, dtype=np.int64)
    nstr = len (strs)
    addrs = np.zeros (nstr, dtype=np.int32)
    gentable = np.ravel (csf_gentable (nspin, smult))
    twoS = smult - 1
    libcsf.FCICSFstrs2addr (addrs.ctypes.data_as (ctypes.c_void_p),
                            strs.ctypes.data_as (ctypes.c_void_p),
                            ctypes.c_int (nstr),
                            gentable.ctypes.data_as (ctypes.c_void_p),
                            ctypes.c_int (nspin),
                            ctypes.c_int (twoS));
    return addrs


def check_all_umat_size (norb, neleca, nelecb):
    ''' Calcluate the number of elements of the unitary matrix between all possible determinants
    and all possible CSFs for (neleca, nelecb) electrons in norb orbitals '''
    min_smult = neleca - nelecb + 1
    min_npair = max (0, neleca + nelecb - norb)
    max_smult = neleca + nelecb - 2*min_npair + 1
    return sum ([check_tot_umat_size (norb, neleca, nelecb, smult) for smult in range (min_smult, max_smult, 2)])

def check_tot_umat_size (norb, neleca, nelecb, smult):
    ''' Calculate the number of elements of the unitary matrix between all possible determinants
    and CSFs of a given spin state for (neleca, nelecb) electrons in norb orbitals '''
    min_npair = max (0, neleca + nelecb - norb)
    return sum ([check_umat_size (norb, neleca, nelecb, npair, smult) for npair in range (min_npair, nelecb+1)])

def check_umat_size (norb, neleca, nelecb, npair, smult):
    ''' Calculate the number of elements of the unitary matrix between determinants with npair electron pairs
    and CSFs of a given spin state for (neleca, nelecb) electrons in norb orbitals '''
    nspin = neleca + nelecb - 2*npair
    ms = (neleca - nelecb) / 2
    na = (nspin + neleca - nelecb) // 2
    ndet = special.binom (nspin, na)
    ncsf = count_csfs (nspin, smult)
    return ndet * ncsf


if __name__ == '__main__':
    for nspin in range (21):
        for s in np.arange ((nspin%2)/2, (nspin/2)+0.1, 1):
            ncsf = count_csfs (nspin, s)
            print ("Supposedly {} csfs of {} spins with overall s = {}".format (ncsf, nspin, s));
            rand_addrs = np.random.randint (0, high=ncsf, size=min (ncsf, 5), dtype=np.int32)
            rand_strs = addrs2str (nspin, s, rand_addrs)
            rand_addrs_2 = strs2addr (nspin, s, rand_strs)
            assert (np.all (rand_addrs == rand_addrs_2))
    
    for nspin in range (15):
        for ms in np.arange ((nspin%2)/2, (nspin/2)+0.1, 1):
            evals = []
            evecs = []
            S2mat = None
            for s in np.arange (abs (ms), (nspin/2)+0.1, 1):
                umat, S2mat = test_spin_evecs (nspin, ms, s, S2mat=S2mat)
                evecs.append (umat)
                evals.append (s*(s+1)*np.ones (umat.shape[1]))
            print ("COLLECTIVE RESULTS:")
            t_start = time.time ()
            evals = np.concatenate (evals)
            evecs = np.concatenate (evecs, axis=-1)
            issquare = np.all (evecs.shape == S2mat.shape)
            print (('Is the final CSF vector matrix square with correct dimension?'
                    ' {}').format (("NO ({0}-by-{1} vs {2}-by-{3})".format (*evecs.shape, *S2mat.shape), "Yes")[issquare]))
            if not issquare:
                print ("{} spins, {} projected spin overall: FAILED".format (nspin, ms))
                continue
            ovlp = np.dot (evecs.T, evecs)
            S2op = reduce (np.dot, (evecs.T, S2mat, evecs))
            ovlperr = linalg.norm (ovlp - np.eye (evecs.shape[1]))
            diagerr = linalg.norm (S2op - np.diag (evals))
            isorthnorm = ovlperr < 1e-8
            diagsS2 = diagerr < 1e-8
            print ("TIME: {} seconds to analyze umat for {} spins with ms={} and all s".format (
                time.time() - t_start, nspin, ms))
            print ("Is the final CSF vector matrix unitary? {}".format (("NO (err = {})".format (ovlperr), "Yes")[isorthnorm]))
            print (('Does the final CSF vector matrix correctly diagonalize S2?'
                    ' {}').format (('NO (err = {})'.format (diagerr), 'Yes')[diagsS2]))
            print ("{} spins, {} projected spin overall: {}".format (nspin, ms, ("FAILED", "Passed")[isorthnorm and diagsS2]))
            sys.stdout.flush ()
    
    