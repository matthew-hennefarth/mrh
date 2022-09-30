#!/usr/bin/env python

from functools import reduce
import numpy as np

def h_for_cas(mc, mo_coeff=None, ncas=None, ncore=None):
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ncas is None: ncas = mc.ncas
    if ncore is None: ncore = mc.ncore

    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:ncore+ncas]

    hcore = mc.get_hcore()
    
    core_dm = np.dot(mo_core, mo_core.conj().T) * 2
    energy_core = np.einsum('ij,ji', core_dm, hcore).real
    
    hcas = reduce(np.dot, (mo_cas.conj().T, hcore, mo_cas))

    # Should return h_tu (hcas) and 2 sum_i h_ii
    return hcas, energy_core

def get_linear_pdft_ham(mc, mo_coeff=None, ci=None):
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff

    if ci is None:
        ci = mc.ci

    hcas, energy_core = h_for_cas(mc, mo_coeff=mo_coeff)

    print(hcas)
    print(energy_core)



if __name__ == "__main__":
    from pyscf import gto, scf, mcscf
    mol = gto.M(atom='''Li   0 0 0
                       H 1.5 0 0''',
                       basis='sto3g',
                       verbose=5,
                       spin=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.fix_spin_(ss=0)

    N_STATES = 2

    mc = mc.state_average([1.0/float(N_STATES),]*N_STATES)
    mc.kernel()
    
    get_linear_pdft_ham(mc)

