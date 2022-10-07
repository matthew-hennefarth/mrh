#!/usr/bin/env python

from functools import reduce
import numpy as np

from pyscf import ao2mo
from mrh.my_pyscf.mcpdft import _dms

# def h_for_cas(mc, mo_coeff=None, ncas=None, ncore=None):
    # if mo_coeff is None: mo_coeff = mc.mo_coeff
    # if ncas is None: ncas = mc.ncas
    # if ncore is None: ncore = mc.ncore

    # mo_core = mo_coeff[:,:ncore]
    # mo_cas = mo_coeff[:,ncore:ncore+ncas]

    # hcore = mc.get_hcore()
    
    # core_dm = np.dot(mo_core, mo_core.conj().T) * 2
    # energy_core = np.einsum('ij,ji', core_dm, hcore).real

    # hcas = reduce(np.dot, (mo_cas.conj().T, hcore, mo_cas))

    # h = reduce(np.dot, (mo_coeff.conj().T, hcore, mo_coeff))

    # # Should return h_tu (hcas) and 2 sum_i h_ii
    # # interms of active-orbitals
    # return hcas, energy_core

def get_h(mc, mo_coeff=None):
    if mo_coeff is None: mo_coeff = mc.mo_coeff

    return reduce(np.dot, (mo_coeff.conj().T, mc.get_hcore(), mo_coeff))
    

def get_g(mc, mo_coeff=None):
    if mo_coeff is None: mo_coeff = mc.mo_coeff

    return ao2mo.restore(1, ao2mo.get_ao_eri(mc.mol), mo_coeff.shape[0])

def get_D0(mc, ci=None):
    if ci is None:
        ci = mc.ci

    nstates = len(ci)
    for root in nstates:
        print(root)

def get_linear_pdft_ham(mc, mo_coeff=None, ci=None, ncas=None, ncore=None, nelecas=None):
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff

    if ci is None:
        ci = mc.ci

    if ncas is None:
        ncas = mc.ncas

    if ncore is None:
        ncore = mc.ncore

    # the h_pq in MO basis
    hmo = get_h(mc, mo_coeff=mo_coeff)
    
    # the d_pqrs in MO basis
    gmo = get_g(mc, mo_coeff=mo_coeff)

    fcisolver, _, nelecas = _dms._get_fcisolver(mc, ci)

    #
    #casdm1_transit, casdm2_transit = fcisolver.trans_rdm12(ci[1], ci[0], ncas, nelecas)
    #print(casdm1_transit)
    #print(casdm2_transit.shape)
    get_D0(mc)

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

