#!/usr/bin/env python

from functools import reduce
import numpy as np
from scipy import linalg

from pyscf import ao2mo
from pyscf.fci import direct_spin1
from mrh.my_pyscf.mcpdft import _dms


def weighted_average_densities(mc, ci=None, weights=None, ncas=None):
    '''Compute the weighted average 1- and 2-electron CAS densities. 
    1-electron CAS is returned as spin-separated.
    
    Args:
        mc : instance of class _PDFT
        ci : list of ndarrays of length nroots
            CI vectors should be from a converged CASSCF/CASCI calculation
        weights : list of floats of length nroots
            Weight for each state. If none, uses weights from SA-CASSCF
            calculation
        ncas : float
            Number of active space MOs

    Returns:
        A tuple, the first is casdm1s and the second is casdm2 where they are 
        weighted averages where the weights are given.
    '''
    if ci is None: ci = mc.ci
    if weights is None: weights = mc.fcisolver.weights
    if ncas is None: ncas = mc.ncas

    # There might be a better way to construct all of them, but this should be 
    # more cost-effective then what is currently in the _dms file.
    fcisolver, _, nelecas = _dms._get_fcisolver(mc, ci)

    casdm1s_all = np.array([np.array(fcisolver.make_rdm1s(c, ncas, nelecas))
                            for c in ci])
    casdm2_all = [fcisolver.make_rdm2(c, ncas, nelecas) for c in ci]

    casdm1s_a_0 = np.einsum("i,i...->...", weights, casdm1s_all[:, 0])
    casdm1s_b_0 = np.einsum("i,i...->...", weights, casdm1s_all[:, 1])
    casdm2_0 = np.einsum("i,i...->...", weights, casdm2_all)

    return (casdm1s_a_0, casdm1s_b_0), casdm2_0


def get_effhcore(mc, Eot_0, veff1_0, veff2_0, casdm1s_0, casdm2_0, ncas=None,
                 ncore=None):
    # This should be correct now where we are
    # returning the h_const term in my derivations
    mo_coeff = mc.mo_coeff
    if ncas is None: ncas = mc.ncas
    if ncore is None: ncore = mc.ncore

    nocc = ncore + ncas

    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:nocc]
    # Active space Density matrix for the expansion term
    casdm1_0 = casdm1s_0[0] + casdm1s_0[1]

    # h_nuc + Eot
    energy_core = mc.energy_nuc() + Eot_0

    # the 1/2 g_pqrs D_pq D_rs around zeroth states
    dm1s = _dms.casdm1s_to_dm1s(mc, casdm1s=casdm1s_0)
    dm1 = dm1s[0] + dm1s[1]
    vj = mc._scf.get_j(dm=dm1)
    coulomb = 0.5 * np.tensordot(vj, dm1)
    energy_core -= coulomb

    if mo_core.size != 0:
        core_dm = np.dot(mo_core, mo_core.conj().T) * 2
        energy_core -= veff2_0.energy_core
        energy_core -= np.einsum('ij,ji', core_dm, veff1_0).real

    veff1_0_cas = reduce(np.dot,
                         (mo_cas.conj().T, veff1_0, mo_cas)) + veff2_0.vhf_c[
                                                               ncore:nocc,
                                                               ncore:nocc]
    energy_core -= np.einsum('vw,vw', veff1_0_cas, casdm1_0)

    veff2_0_cas = get_transformed_h2eff_for_cas(mc, veff2_0)
    energy_core -= np.einsum('vwxy,vwxy', veff2_0_cas, casdm2_0)

    return energy_core


def transformed_h1e_for_cas(mc, Eot_0, veff1_0, veff2_0, casdm1s_0, casdm2_0,
                            mo_coeff=None, ncas=None, ncore=None):
    '''CAS Space one-electron Linear PDFT Hamiltonian

    Args:
        mc : instance of a _PDFT object

        Eot_0 : float
            On-top energy for the expansion density.

        veff1_0 : ndarray with shape (nao, nao)
            1-body effective potential in the AO basis.
            Should not include classical Coulomb potential term.
            Generated from expansion density.

        veff2_0 : pyscf.mcscf.mc_ao2mo._ERIS instance
            Relevant 2-body effecive potential in the MO basis.
            Generated from expansion density.

        casdm1s_0 : ndarray of shape (2,ncas,ncas)
            Spin-separated 1-RDM in the active space generated
            from expansion density

        mo_coeff : ndarray of shape (nao,nmo)
            A full set of molecular orbital coefficients. Taken from
            self if not provided.

        ncas : int
            Number of active space molecular orbitals

        ncore : int
            Number of core molecular orbitals

    Returns:
        A tuple, the first is the effective one-electron linear PDFT Hamiltonian
        defined in CAS space, the second is the core energy.
    '''

    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ncas is None: ncas = mc.ncas
    if ncore is None: ncore = mc.ncore

    nocc = ncore + ncas

    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:nocc]

    # Active space Density matrix for the expansion term
    casdm1_0 = casdm1s_0[0] + casdm1s_0[1]

    hcore_eff = mc.get_hcore() + veff1_0
    energy_core = get_effhcore(mc, Eot_0, veff1_0, veff2_0, casdm1s_0,
                               casdm2_0)

    # Gets the 2 electron integrals in MO
    eris = ao2mo.restore(1, ao2mo.get_mo_eri(mc.mol, mo_coeff), len(mo_coeff))
    # Partition for summations
    eris_core = eris[:, :, :ncore, :ncore]
    eris_cas = eris[:, :, ncore:nocc, ncore:nocc]

    # The 2\sum_j g_pqjj + g_pqvw D_vw^0 term (all in MO basis)
    reduced_coulomb = 2 * np.einsum("pqjj->pq", eris_core) + np.einsum(
        "pqvw,vw->pq", eris_cas, casdm1_0)

    if mo_core.size != 0:
        core_dm = np.dot(mo_core, mo_core.conj().T) * 2
        energy_core += veff2_0.energy_core
        energy_core += np.einsum('ij,ji', core_dm, hcore_eff).real
        energy_core += 2 * np.trace(reduced_coulomb[:ncore, :ncore])

    h1eff = reduce(np.dot,
                   (mo_cas.conj().T, hcore_eff, mo_cas)) + veff2_0.vhf_c[
                                                           ncore:nocc,
                                                           ncore:nocc]

    h1eff += reduced_coulomb[ncore:nocc, ncore:nocc]
    return h1eff, energy_core


def get_transformed_h2eff_for_cas(mc, veff2_0, ncore=None, ncas=None):
    if ncore is None: ncore = mc.ncore
    if ncas is None: ncas = mc.ncas

    nocc = ncore + ncas
    return veff2_0.papa[ncore:nocc, :, ncore:nocc, :]


def make_heff_lin(mc, mo_coeff=None, ci=None, ot=None):
    if mo_coeff is None: mo_coeff = mc.mo_coeff

    if ci is None: ci = mc.ci

    if ot is None: ot = ot = mc.otfnal
    ot.reset(mol=mc.mol)

    ncas = mc.ncas

    casdm1s_0, casdm2_0 = weighted_average_densities(mc)

    Eot_0 = mc.energy_dft(ot=ot, mo_coeff=mo_coeff, casdm1s=casdm1s_0,
                          casdm2=casdm2_0)
    veff1_0, veff2_0 = mc.get_pdft_veff(mo=mo_coeff, casdm1s=casdm1s_0,
                                        casdm2=casdm2_0)

    h1, h0 = transformed_h1e_for_cas(mc, Eot_0, veff1_0, veff2_0, casdm1s_0,
                                     casdm2_0)

    h2 = get_transformed_h2eff_for_cas(mc, veff2_0)

    h2eff = direct_spin1.absorb_h1e(h1, h2, ncas, mc.nelecas, 0.5)
    hc_all = [direct_spin1.contract_2e(h2eff, c, ncas, mc.nelecas) for c in ci]
    heff = np.tensordot(ci, hc_all, axes=((1, 2), (1, 2)))
    idx = np.diag_indices_from(heff)
    heff[idx] += h0
    return heff


class _QLPDFT:

    def __init__(self, mc):
        self.__dict__.update(mc.__dict__)

    @property
    def e_states (self):
        if self._in_mcscf_env:
            return self.fcisolver.e_states
        else:
            return self._e_states
    @e_states.setter
    def e_states (self, x):
        self._e_states = x

    make_heff_lin = make_heff_lin

    def get_heff_pdft(self):
        idx = np.diag_indices_from(self.heff_lin)
        heff_pdft = self.heff_lin.copy()
        heff_pdft[idx] = self.hdiag_pdft
        return heff_pdft

    def get_heff_offdiag(self):
        idx = np.diag_indices_from(self.heff_lin)
        heff_offdiag = self.heff_lin.copy()
        heff_offdiag[idx] = 0.0
        return heff_offdiag

    def kernel(self, mo_coeff=None, ci0=None, otxc=None, grids_level=None,
                grids_attr=None, **kwargs):
        self.otfnal.reset(mol=self.mol)  # scanner mode safety
        if ci0 is None and isinstance(getattr(self, 'ci', None), list):
            ci0 = [c.copy() for c in self.ci]

        self.optimize_mcscf_(mo_coeff=mo_coeff, ci0=ci0)
        self.heff_lin = self.make_heff_lin()

        self.hdiag_pdft = self.compute_pdft_energy_(
            otxc=otxc, grids_level=grids_level, grids_attr=grids_attr)[-1]

        self.e_states, self.si_pdft = self._eig_si(self.get_heff_pdft())
        self.e_tot = np.dot(self.e_states, self.weights)

        return (self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy)

    def _eig_si (self, heff):
        return linalg.eigh (heff)

def qlpdft(mc):
    mcbase_class = mc.__class__
    class QLPDFT(_QLPDFT, mcbase_class):
        pass

    return QLPDFT(mc)

if __name__ == "__main__":
    from pyscf import gto, scf, mcscf
    from mrh.my_pyscf import mcpdft
    from mrh.my_pyscf.fci import csf_solver

    mol = gto.M(atom='''Li 0 0 0
                        H 1.5 0 0''',
                basis='6-31G**',
                verbose=5,
                spin=0)
    mf = scf.RHF(mol).run()

    mc = mcpdft.CASSCF(mf, 'tPBE', 2, 2, grids_level=6)
    mc.fcisolver = csf_solver (mol, smult = 1)

    N_STATES = 2

    mc = mc.state_average([1.0 / float(N_STATES), ] * N_STATES)

    test = qlpdft(mc)
    test.kernel()

    print(test.e_states)
    print(test.hdiag_pdft)
    print(test.heff_lin)
    print(test.e_mcscf)

