#!/usr/bin/env python

from functools import reduce
import numpy as np
from scipy import linalg

from pyscf import ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf.fci import direct_spin1

from mrh.my_pyscf.mcpdft import _dms


def weighted_average_densities(mc, ci=None, weights=None, ncas=None):
    '''Compute the weighted average 1- and 2-electron CAS densities. 
    1-electron CAS is returned as spin-separated.
    
    Args:
        mc : instance of class _PDFT

        ci : list of ndarrays of length nroots
            CI vectors should be from a converged CASSCF/CASCI calculation

        weights : ndarray of length nroots
            Weight for each state. If none, uses weights from SA-CASSCF
            calculation

        ncas : float
            Number of active space MOs

    Returns:
        A tuple, the first is casdm1s and the second is casdm2 where they are 
        weighted averages where the weights are given.
    '''
    if ci is None:
        ci = mc.ci
    
    if weights is None: 
        weights = mc.weights
    
    if ncas is None: 
        ncas = mc.ncas

    # There might be a better way to construct all of them, but this should be
    # more cost-effective than what is currently in the _dms file.
    fcisolver, _, nelecas = _dms._get_fcisolver(mc, ci)
    casdm1s_all = [fcisolver.make_rdm1s(c, ncas, nelecas) for c in ci]
    casdm2_all = [fcisolver.make_rdm2(c, ncas, nelecas) for c in ci]

    casdm1s_0 = np.tensordot(weights, casdm1s_all, axes=1)
    casdm2_0 = np.tensordot(weights, casdm2_all, axes=1)

    return tuple(casdm1s_0), casdm2_0


def get_effhconst(mc, veff1_0, veff2_0, casdm1s_0, casdm2_0, mo_coeff=None,
                  ot=None, ncas=None, ncore=None):
    ''' Compute h_const for the linear PDFT Hamiltonian

    Args:
        mc : instance of class _PDFT
        
        veff1_0 : ndarray with shape (nao, nao)
            1-body effective potential in the AO basis.
            Should not include classical Coulomb potential term.
            Generated from expansion density

        veff2_0 : pyscf.mcscf.mc_ao2mo._ERIS instance
            Relevant 2-body effective potential in the MO basis.
            Generated from expansion density.

        casdm1s_0 : ndarray of shape (2, ncas, ncas)
            Spin-separated 1-RDM in the active space generated 
            from expansion density.

        casdm2_0 : ndarray of shape (ncas, ncas, ncas, ncas)
            Spin-summed 2-RDM in the active space generated
            from expansion density.

        mo_coeff : ndarray of shape (nao, nmo)
            A full set of molecular orbital coefficients. Taken from
            self if not provided.

        ot : an instance of on-top functional class - see otfnal.py

        ncas : float
            Number of active space MOs

        ncore: float
            Number of core MOs

    Returns:
        Constant term h_const for the expansion term.
    '''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff

    if ot is None:
        ot = mc.otfnal

    if ncas is None:
        ncas = mc.ncas

    if ncore is None:
        ncore = mc.ncore

    nocc = ncore + ncas

    # Get the 1-RDM matrices
    casdm1_0 = casdm1s_0[0] + casdm1s_0[1]
    dm1s = _dms.casdm1s_to_dm1s(mc, casdm1s=casdm1s_0)
    dm1 = dm1s[0] + dm1s[1]

    # Eot for zeroth order state
    Eot_0 = mc.energy_dft(ot=ot, mo_coeff=mo_coeff, casdm1s=casdm1s_0,
                          casdm2=casdm2_0)

    # Coulomb energy for zeroth order state
    vj = mc._scf.get_j(dm=dm1)
    E_j = np.tensordot(vj, dm1) / 2

    # One-electron on-top potential energy
    E_veff1 = np.tensordot(veff1_0, dm1)

    # Deal with 2-electron on-top potential energy
    E_veff2 = veff2_0.energy_core
    E_veff2 += np.tensordot(veff2_0.vhf_c[ncore:nocc, ncore:nocc], casdm1_0)
    E_veff2 += 0.5*np.tensordot(mc.get_h2eff_lin(veff2_0), casdm2_0, axes=4)

    # h_nuc + Eot - 1/2 g_pqrs D_pq D_rs - V_pq D_pq - 1/2 v_pqrs d_pqrs
    energy_core = mc.energy_nuc() + Eot_0 - E_j - E_veff1 - E_veff2
    return energy_core


def transformed_h1e_for_cas(mc, veff1_0, veff2_0, casdm1s_0, casdm2_0,
                            mo_coeff=None, ncas=None, ncore=None, ot=None):
    '''Compute the CAS one-particle linear PDFT Hamiltonian

    Args:
        mc : instance of a _PDFT object

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

        casdm2_0 : ndarray of shape (ncas,ncas,ncas,ncas)
            Spin-summed 2-RDM in the active space generated
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
        defined in CAS space, the second is the modified core energy.
    '''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff

    if ncas is None:
        ncas = mc.ncas

    if ncore is None:
        ncore = mc.ncore

    if ot is None:
        ot = mc.otfnal

    nocc = ncore + ncas
    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:nocc]

    dm1s = _dms.casdm1s_to_dm1s(mc, casdm1s=casdm1s_0)
    dm1 = dm1s[0] + dm1s[1]
    v_j = mc._scf.get_j(dm=dm1)

    # h_pq + V_pq + J_pq all in AO integrals
    hcore_eff = mc.get_hcore() + veff1_0 + v_j
    energy_core = mc.get_effhconst(veff1_0, veff2_0, casdm1s_0,
                                   casdm2_0, ot=ot)

    if mo_core.size != 0:
        core_dm = np.dot(mo_core, mo_core.conj().T) * 2
        # This is precomputed in MRH's ERIS object
        energy_core += veff2_0.energy_core
        energy_core += np.einsum('ij,ji', core_dm, hcore_eff).real

    h1eff = reduce(np.dot, (mo_cas.conj().T, hcore_eff, mo_cas))
    # Add in the 2-electron portion that acts as a 1-electron operator
    h1eff += veff2_0.vhf_c[ncore:nocc, ncore:nocc]

    return h1eff, energy_core


def get_transformed_h2eff_for_cas(mc, veff2_0, ncore=None, ncas=None):
    '''Compute the CAS two-particle linear PDFT Hamiltonian

    Args:
        veff2_0 : pyscf.mcscf.mc_ao2mo._ERIS instance
            Relevant 2-body effecive potential in the MO basis.
            Generated from expansion density.

        ncore : int
            Number of core MOs

        ncas : int
            Number of active space MOs

    Returns:
        ndarray of shape (ncas,ncas,ncas,ncas) which contain v_vwxy
    '''
    if ncore is None:
        ncore = mc.ncore

    if ncas is None:
        ncas = mc.ncas

    nocc = ncore + ncas

    return veff2_0.papa[ncore:nocc, :, ncore:nocc, :]


def make_heff_lin_(mc, mo_coeff=None, ci=None, ot=None):
    '''Compute the linear PDFT Hamiltonian

    Args:
        mo_coeff : ndarray of shape (nao, nmo)
            A full set of molecular orbital coefficients. Taken from
            self if not provided.

        ci : list of ndarrays of length nroots
            CI vectors should be from a converged CASSCF/CASCI calculation

        ot : an instance of on-top functional class - see otfnal.py

    Returns:
        heff_lin : ndarray of shape (nroots, nroots)
            Linear approximation to the MC-PDFT energy expressed as a
            hamiltonian in the basis provided by the CI vectors.
    '''

    if mo_coeff is None:
        mo_coeff = mc.mo_coeff

    if ci is None:
        ci = mc.ci

    if ot is None:
        ot = mc.otfnal

    ot.reset(mol=mc.mol)

    ncas = mc.ncas
    casdm1s_0, casdm2_0 = mc.get_casdm12_0()

    veff1_0, veff2_0 = mc.get_pdft_veff(mo=mo_coeff, casdm1s=casdm1s_0,
                                        casdm2=casdm2_0)

    # This is all standard procedure for generating the hamiltonian in PySCF
    h1, h0 = mc.get_h1eff_lin(veff1_0, veff2_0, casdm1s_0, casdm2_0, ot=ot)
    h2 = mc.get_h2eff_lin(veff2_0)
    h2eff = direct_spin1.absorb_h1e(h1, h2, ncas, mc.nelecas, 0.5)
    hc_all = [direct_spin1.contract_2e(h2eff, c, ncas, mc.nelecas) for c in ci]
    heff = np.tensordot(ci, hc_all, axes=((1, 2), (1, 2)))
    idx = np.diag_indices_from(heff)
    heff[idx] += h0

    return heff

def kernel(mc, mo_coeff=None, ci0=None, otxc=None, grids_level=None,
               grids_attr=None, verbose=logger.NOTE):
    if otxc is None:
        otxc = mc.otfnal

    if mo_coeff is None:
        mo_coeff = mc.mo_coeff

    log = logger.new_logger(mc, verbose)
    mc.optimize_mcscf_(mo_coeff=mo_coeff, ci0=ci0)
    mc.heff_lin = mc.make_heff_lin_()

    mc.hdiag_pdft = mc.compute_pdft_energy_(
        otxc=otxc, grids_level=grids_level, grids_attr=grids_attr)[-1]

    mc.e_states, mc.si_pdft = mc._eig_si(mc.get_heff_pdft())
    mc.e_tot = np.dot(mc.e_states, mc.weights)

    return (
        mc.e_tot, mc.e_ot, mc.e_mcscf, mc.e_cas, mc.ci,
        mc.mo_coeff, mc.mo_energy)


class _QLPDFT:

    def __init__(self, mc):
        self.__dict__.update(mc.__dict__)

    @property
    def e_states(self):
        if self._in_mcscf_env:
            return self.fcisolver.e_states

        else:
            return self._e_states

    @e_states.setter
    def e_states(self, x):
        self._e_states = x

    make_heff_lin_ = make_heff_lin_
    make_heff_lin_.__doc__ = make_heff_lin_.__doc__

    get_effhconst = get_effhconst
    get_effhconst.__doc__ = get_effhconst.__doc__

    get_h1eff_lin = transformed_h1e_for_cas
    get_h1eff_lin.__doc__ = transformed_h1e_for_cas.__doc__

    get_h2eff_lin = get_transformed_h2eff_for_cas
    get_h2eff_lin.__doc__ = get_transformed_h2eff_for_cas.__doc__

    get_casdm12_0 = weighted_average_densities
    get_casdm12_0.__doc__ = weighted_average_densities.__doc__

    def get_heff_pdft(self):
        '''The QL-PDFT effective Hamiltonian matrix
            ( EPDFT_0    H_10*^lin  ...)
            ( H_10^lin   EPDFT_1    ...)
            ( ...        ...        ...)

        Returns:
            heff_pdft : ndarray of shape (nroots, nroots)
                Contains the linear PDFT Hamiltonian on the off-diagonals
                and PDFT energies on the diagonals
        '''
        idx = np.diag_indices_from(self.heff_lin)
        heff_pdft = self.heff_lin.copy()
        heff_pdft[idx] = self.hdiag_pdft
        return heff_pdft

    def get_heff_offdiag(self):
        '''Off-diagonal elements of the QL-PDFT effective Hamiltonian matrix
            ( 0          H_10*^lin  ...)
            ( H_10^lin   0          ...)
            ( ...        ...        ...)

        Returns:
            heff_offdiag : ndarray of shape (nroots, nroots)
                Contains the linear PDFT Hamiltonian on the off-diagonals
                and 0 on the diagonals
        '''
        idx = np.diag_indices_from(self.heff_lin)
        heff_offdiag = self.heff_lin.copy()
        heff_offdiag[idx] = 0.0
        return heff_offdiag

    def get_heff_diag(self):
        '''Diagonal elements of the linear PDFT Hamiltonian matrix
            (H_00^lin, H_11^lin, H_22^lin, ...)

        Returns:
            hlin_diag : ndarray of shape (nroots)
                Contains the linear approximation to the MC-PDFT energy. These
                are also the diagonal elements of the linear PDFT Hamiltonian
                matrix.
        '''
        idx = np.diag_indices_from(self.heff_lin)
        heff_diag = self.heff_lin.copy()
        return heff_diag[idx]

    def kernel(self, mo_coeff=None, ci0=None, otxc=None, grids_level=None,
               grids_attr=None, verbose=None, **kwargs):
        '''
        Returns:
            7 elements, they are
            total energy,
            on-top energy,
            the MCSCF energies,
            the active space CI energy,
            the active space FCI wave function coefficients,
            the MCSCF canonical orbital coefficients,
            the MCSCF canonical orbital energies

        They are attributes of the QLPDFT object, which can be accessed by
        .e_tot, .e_ot, .e_mcscf, .e_cas, .ci, .mo_coeff, .mo_energy
        '''
        self.otfnal.reset(mol=self.mol)  # scanner mode safety
        if otxc is None:
            otxc = self.otfnal

        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        else:
            self.mo_coeff = mo_coeff

        log = logger.new_logger(self, verbose)

        if ci0 is None and isinstance(getattr(self, 'ci', None), list):
            ci0 = [c.copy() for c in self.ci]

        kernel(self, mo_coeff, ci0, otxc, grids_level, grids_attr, verbose=log)
        self._finalize_ql()
        return (
            self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci,
            self.mo_coeff, self.mo_energy)

    def hybrid_kernel(self, lam=0):
        #self.heff_hyb = (1.0-lam) * self.get_heff_pdft()
        self.heff_hyb = (1.0-lam) * self.heff_lin
        idx = np.diag_indices_from(self.heff_hyb)
        self.heff_hyb[idx] += lam * self.e_mcscf
        self.e_hyb_states, self.si_hyb_pdft = self._eig_si(self.heff_hyb)
    
        return self.e_hyb_states, self.si_hyb_pdft

        

    def _finalize_ql(self):
        log = logger.Logger(self.stdout, self.verbose)
        nroots = len(self.e_states)
        log.note("%s (final) states:", self.__class__.__name__)
        if log.verbose >= logger.NOTE and getattr(self.fcisolver, 'spin_square', None):
            ci = np.tensordot(self.si_pdft, np.asarray(self.ci), axes=1)
            ss = self.fcisolver.states_spin_square(ci, self.ncas, self.nelecas)[0]

            for i in range(nroots):
                log.note('  State %d weight %g  EMSPDFT = %.15g  S^2 = %.7f',
                         i, self.weights[i], self.e_states[i], ss[i])

        else:
            for i in range(nroots):
                log.note('  State %d weight %g  EMSPDFT = %.15g', i,
                         self.weights[i], self.e_states[i])

    def _eig_si(self, heff):
        return linalg.eigh(heff)


def qlpdft(mc):
    mcbase_class = mc.__class__

    class QLPDFT(_QLPDFT, mcbase_class):
        pass

    return QLPDFT(mc)


if __name__ == "__main__":
    from pyscf import gto, scf
    from mrh.my_pyscf import mcpdft
    from mrh.my_pyscf.fci import csf_solver

    mol = gto.M(atom='''H 0 0 0
                       H 1.5 0 0''',
                basis='sto-3g',
                verbose=5,
                spin=0,
                unit="AU",
                symmetry=True)

    mf = scf.RHF(mol).run()

    mc = mcpdft.CASSCF(mf, 'tPBE', 2, 2, grids_level=6)
    mc.fcisolver = csf_solver(mol, smult=1)

    N_STATES = 2

    mc = mc.state_average([1.0 / float(N_STATES), ] * N_STATES)

    sc = qlpdft(mc)
    sc.kernel()
