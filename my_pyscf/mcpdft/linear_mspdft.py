#!/usr/bin/env python

from functools import reduce
import numpy as np
from scipy import linalg

from pyscf import ao2mo
from pyscf.fci import direct_spin1
from mrh.my_pyscf.mcpdft import _dms


def get_expansion_densities(mc, ci=None, weights=None, ncas=None):
    if ci is None:
        ci = mc.ci
    
    if weights is None:
        weights = mc.fcisolver.weights

    if ncas is None:
        ncas = mc.ncas

    fcisolver, _, nelecas = _dms._get_fcisolver(mc, ci)

    casdm1s_all = np.array([np.array(fcisolver.make_rdm1s(c, ncas, nelecas)) for c in ci])
    casdm2_all = [fcisolver.make_rdm2(c, ncas, nelecas) for c in ci]

    assert(len(ci) == len(weights)) 

    casdm1s_a_0 = np.einsum("i,i...->...", weights, casdm1s_all[:,0])
    casdm1s_b_0 = np.einsum("i,i...->...", weights, casdm1s_all[:,1])
    casdm2_0 = np.einsum("i,i...->...", weights, casdm2_all)

    return (casdm1s_a_0, casdm1s_b_0), casdm2_0


def get_transformed_h1eff(mc, veff1_0, veff2_0, casdm1s_0, mo_coeff=None, ncas=None, ncore=None):
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ncas is None: ncas = mc.ncas
    if ncore is None: ncore = mc.ncore

    nocc = ncore + ncas

    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:nocc]

    # Active space Density matrix for the expansion term
    casdm1_0 = casdm1s_0[0] + casdm1s_0[1]

    # h_pq + V_pq in atomic integrals
    hcore_eff = mc.get_hcore() + veff1_0
    energy_core = mc.energy_nuc()
    
    # Gets the 2 electron integrals in MO
    eris = ao2mo.restore(1, ao2mo.get_mo_eri(mc.mol, mo_coeff), len(mo_coeff))
    # Partition for summations
    eris_core = eris[:, :, :ncore,:ncore]
    eris_cas = eris[:, :, ncore:nocc, ncore:nocc]
    
    # The 2\sum_j g_pqjj + g_pqvw D_vw^0 term (all in MO basis)
    reduced_coulomb = 2*np.einsum("pqjj->pq", eris_core) + np.einsum("pqvw,vw->pq", eris_cas, casdm1_0)
   
    if mo_core.size != 0:
        core_dm = np.dot(mo_core, mo_core.conj().T) * 2
        energy_core += veff2_0.energy_core 
        energy_core += np.einsum('ij,ji', core_dm, hcore_eff).real
        energy_core += 2*np.trace(reduced_coulomb[:ncore, :ncore])

    h1eff = reduce(np.dot, (mo_cas.conj().T, hcore_eff, mo_cas)) + veff2_0.vhf_c[ncore:nocc,ncore:nocc]
     
    h1eff += reduced_coulomb[ncore:nocc, ncore:nocc]

    return h1eff, energy_core


def get_linear_pdft_ham(mc, mo_coeff=None, ci=None, ot=None, ncore=None, ncas=None):
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff

    if ci is None:
        ci = mc.ci
    
    if ot is None:
        ot = ot=mc.otfnal
    ot.reset(mol=mc.mol)

    if ncore is None:
        ncore = mc.ncore

    if ncas is None:
        ncas = mc.ncas

    nocc = ncore + ncas

    casdm1s_0, casdm2_0 = get_expansion_densities(mc, ci=ci)    

    Eot_0 = mc.energy_dft(ot=ot, mo_coeff=mo_coeff, casdm1s=casdm1s_0, casdm2=casdm2_0)
    veff1_0, veff2_0 = mc.get_pdft_veff(mo=mo_coeff, casdm1s=casdm1s_0, casdm2=casdm2_0)    


    h1, h0 = get_transformed_h1eff(mc, veff1_0, veff2_0, casdm1s_0)
    h0 += Eot_0 # This really doesn't have much...information since it is missing the zeroth density constant terms
    # ie the classical coulomb for the zeroth order as well as the veff1_0 and veff2_0 contractions
    h2 = veff2_0.papa[ncore:nocc, : , ncore:nocc, : ]

    h2eff = direct_spin1.absorb_h1e(h1, h2, ncas, mc.nelecas, 0.5)
    hc_all = [direct_spin1.contract_2e(h2eff, c, ncas, mc.nelecas) for c in ci]
    heff = np.tensordot(ci, hc_all, axes=((1,2),(1,2)))
    idx = np.diag_indices_from(heff)
    heff[idx] += h0
    return heff


def get_linear_pdft_ham_offdiag(mc):
    linear_ham = get_linear_pdft_ham(mc)
    idx = np.diag_indices_from(linear_ham)
    linear_ham[idx] = 0.0
    return linear_ham

def get_heff_linpdft(mc):
    linear_ham = get_linear_pdft_ham(mc) 
    idx = np.diag_indices_from(linear_ham)
    linear_ham[idx] = mc.e_states
    return linear_ham

def pseudo_linpdft(mc):
    heff = get_heff_linpdft(mc)
    print(heff)
    return linalg.eigh(heff)

if __name__ == "__main__":
    from pyscf import gto, scf, mcscf
    from mrh.my_pyscf import mcpdft
    mol = gto.M(atom='''Li   0 0 0
                       H 4 0 0''',
                       basis='6-31G**',
                       verbose=5,
                       spin=0)
    mf = scf.RHF(mol).run()
    
    mc = mcpdft.CASSCF(mf, 'tPBE', 2, 2, grids_level=6)
    mc.fix_spin_(ss=0)

    N_STATES = 3

    mc = mc.state_average([1.0/float(N_STATES),]*N_STATES)
    mc.kernel()
    print(mc.e_states)    
    

    initial_energies = mc.e_states.copy()
    initial_energies.sort()
    e_states, si_pdft = pseudo_linpdft(mc)
    print(initial_energies) 
    print(e_states)
    print(e_states-initial_energies)

