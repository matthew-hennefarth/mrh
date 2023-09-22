import numpy as np
from pyscf.lib import logger
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.mcscf.productstate import ProductStateFCISolver
from mrh.my_pyscf.lassi.excitations import ExcitationPSFCISolver
from mrh.my_pyscf.lassi.states import spin_shuffle, spin_shuffle_ci
from mrh.my_pyscf.lassi.states import all_single_excitations, SingleLASRootspace
from mrh.my_pyscf.lassi.lassi import LASSI

def prepare_states (lsi, nmax_charge=0):
    las = lsi._las
    las1 = spin_shuffle (las)
    las1.ci = spin_shuffle_ci (las1, las1.ci)
    # TODO: make states_energy_elec capable of handling lroots and address inconsistency
    # between definition of e_states array for neutral and charge-separated rootspaces
    las1.e_states = las1.energy_nuc () + np.array (las1.states_energy_elec ())
    las2 = all_single_excitations (las1)
    las2.ci, las2.e_states = single_excitations_ci (lsi, las2, las1, nmax_charge=nmax_charge)
    return las2

def single_excitations_ci (lsi, las2, las1, nmax_charge=0):
    log = logger.new_logger (lsi, lsi.verbose)
    mol = lsi.mol
    nfrags = lsi.nfrags
    e_roots = np.append (las1.e_states, np.zeros (las2.nroots-las1.nroots))
    psrefs = []
    ci = [[ci_ij for ci_ij in ci_i] for ci_i in las2.ci]
    for j in range (las1.nroots):
        solvers = [b.fcisolvers[j] for b in las1.fciboxes]
        psrefs.append (ProductStateFCISolver (solvers, stdout=mol.stdout, verbose=mol.verbose))
    spaces = [SingleLASRootspace (las2, m, s, c, 0, ci=[c[ix] for c in ci])
              for ix, (c, m, s, w) in enumerate (zip (*get_space_info (las2)))]
    ncsf = las2.get_ugg ().ncsf_sub
    if isinstance (nmax_charge, np.ndarray): nmax_charge=nmax_charge[None,:]
    lroots = np.minimum (1+nmax_charge, ncsf)
    h0, h1, h2 = lsi.ham_2q ()
    t0 = (logger.process_clock (), logger.perf_counter ())
    for i in range (las1.nroots, las2.nroots):
        psref = []
        ciref = [[] for j in range (nfrags)]
        excfrags = set ()
        for j in range (las1.nroots):
            if not spaces[i].is_single_excitation_of (spaces[j]): continue
            excfrags = excfrags.union (spaces[i].list_excited_fragments (spaces[j]))
            psref.append (psrefs[j])
            for k in range (nfrags):
                ciref[k].append (las1.ci[k][j])
        psexc = ExcitationPSFCISolver (psref, ciref, las2.ncas_sub, las2.nelecas_sub,
                                       stdout=mol.stdout, verbose=mol.verbose)
        neleca = spaces[i].neleca
        nelecb = spaces[i].nelecb
        smults = spaces[i].smults
        for k in excfrags:
            weights = np.ones (lroots[k,i]) / lroots[k,i]
            psexc.set_excited_fragment_(k, (neleca[k],nelecb[k]), smults[k], weights=weights)
        conv, e_roots[i], ci1 = psexc.kernel (h1, h2, ecore=h0,
                                              max_cycle_macro=las2.max_cycle_macro,
                                              conv_tol_self=1)
        if not conv: log.warn ("CI vectors for charge-separated rootspace %d not converged", i)
        for k in range (nfrags):
            ci[k][i] = ci1[k]
        t0 = log.timer ("Space {} excitations".format (i), *t0)
    return ci, e_roots

class LASSIS (LASSI):
    def __init__(self, las, nmax_charge=None, **kwargs):
        self.nmax_charge = nmax_charge
        LASSI.__init__(self, las, **kwargs)
    def kernel (self, nmax_charge=None, **kwargs):
        if nmax_charge is None: nmax_charge = self.nmax_charge
        las = prepare_states (self, nmax_charge=nmax_charge)
        self.__dict__.update(las.__dict__)
        return LASSI.kernel (self, **kwargs)

