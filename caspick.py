'''
robert.anderson@kcl.ac.uk
Dec 2020

Improved CAS picker based on orbital energies and overlap with meta-Lowdin AOs
'''

from pyscf import M, mrpt, mcscf, scf
from definitions import data as defs
from functools import reduce
from matplotlib import pyplot as plt
import numpy as np
import string

ND_TOL_DEFAULT = 1e-3

def gather_near_degeneracies(mo_energies, mask=None, tol=ND_TOL_DEFAULT):
    rnd = lambda i: int(mo_energies[i]/tol)*tol
    if mask is None: mask = np.ones(len(mo_energies), dtype=bool)
    else: assert len(mo_energies)==len(mask)
    sets = {}
    last = rnd(0)
    for i in range(1, len(mo_energies)):
        if not mask[i]: continue
        if last==rnd(i):
            if last in sets: sets[last].append(i+1)
            else: sets[last] = [i, i+1]
        last = rnd(i)
    return sets

def meta_lowdin(scf_obj):
    """
    returns minimally-orthogonalized AOs, whose names are obtained by:
        scf_obj.mol.ao_labels()
    """
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    ovlp_ao = scf_obj.get_ovlp()
    orth_coeff = orth.orth_ao(scf_obj.mol, 'meta_lowdin', s=ovlp_ao)
    return reduce(np.dot, (orth_coeff.T, ovlp_ao, scf_obj.mo_coeff))

class CasPicker:
    show_nd_labels = True
    occ_nd_select = []
    vrt_nd_select = []

    def __init__(self, scf_obj, corr_ao_labels, ewindow=(-np.inf, np.inf), nd_tol=ND_TOL_DEFAULT):
        assert isinstance(scf_obj, scf.hf.SCF)
        self.mo_energy = scf_obj.mo_energy
        assert all(isinstance(i, str) for i in corr_ao_labels)
        self.corr_ao_labels = corr_ao_labels
        assert ewindow[0]<ewindow[1]
        '''
        scf_obj.analyze() with verbose > 4 prints the entire MO coeff matrix
        with rows corresponding to the meta-Lowdin AOs and the columns to the
        molecular orbitals
        
        we're only interested in a subset of the rows (the "correlated AOs") 
        and a subset of the columns (MO energy window)
        '''
        nao = len(scf_obj.mol.ao_labels())
        
        '''
        construct the row mask
        '''
        ao_label_map = {scf_obj.mol.ao_labels()[i].strip():i for i in range(nao)}
        corr_ao_ids = [ao_label_map[name] for name in corr_ao_labels]
        
        '''
        construct the column mask
        '''
        e_min, e_max = ewindow
        e_mask = np.logical_and(scf_obj.mo_energy>e_min, scf_obj.mo_energy<e_max)
        occ_mask = np.logical_and(e_mask, scf_obj.mo_occ>0)
        vrt_mask = np.logical_and(e_mask, scf_obj.mo_occ==0)
        mo_ids = np.where(e_mask)[0]

        '''
        construct the meta-Lowdin AOs
        '''
        ml_coeff = meta_lowdin(scf_obj)


        '''
        apply masks to select the occupied and virtual submatrices of interest
        '''
        occ_ml_coeff = ml_coeff[:,occ_mask][corr_ao_ids,:]
        vrt_ml_coeff = ml_coeff[:,vrt_mask][corr_ao_ids,:]

        assert occ_ml_coeff.shape[0] == vrt_ml_coeff.shape[0]
        assert occ_ml_coeff.shape[1] + vrt_ml_coeff.shape[1] == len(mo_ids)
        self.nocc = occ_ml_coeff.shape[1]
        self.nvrt = vrt_ml_coeff.shape[1]

        '''
        compute total norm of each submatrix column (i.e. the combined weight of 
        all "correlated AOs")
        '''
        self.occ_norms = [np.linalg.norm(occ_ml_coeff[:,i]) for i in range(self.nocc)]
        self.vrt_norms = [np.linalg.norm(vrt_ml_coeff[:,i]) for i in range(self.nvrt)]
        self.occ_ml_coeff = occ_ml_coeff
        self.vrt_ml_coeff = vrt_ml_coeff

        occ_inds = np.where(occ_mask)[0]
        assert len(occ_inds), 'lower bound of energy window exceeds HOMO'
        self.occ_offset = occ_inds[0]
        vrt_inds = np.where(vrt_mask)[0]
        assert len(vrt_inds), 'LUMO exceeds upper bound of energy window'
        self.vrt_offset = vrt_inds[0]

        '''
        nearly-degenerate sets of orbitals are selected together
        '''
        self.nd_sets = gather_near_degeneracies(scf_obj.mo_energy, e_mask, nd_tol)

    def ind_in_nd_set(self, i1):
        '''
        returns index from back of nearly-degenerate set
        '''
        for k, v in self.nd_sets.items():
            try: return len(v)-v.index(i1)
            except ValueError: continue
        return 0

    def plot(self, kind):
        assert kind in ('occ', 'vrt')
        fig, ax1 = plt.subplots()
        ax1.set_title('Meta-Lowdin composition of {} MOs'.format('occupied' if kind=='occ' else 'virtual'))
        ml_coeff = self.occ_ml_coeff if kind=='occ' else self.vrt_ml_coeff
        offset = self.occ_offset if kind=='occ' else self.vrt_offset
        ncol = self.nocc if kind=='occ' else self.nvrt
        ax1.axhline(1, color='gray', linestyle='--')

        inds = range(ncol)

        total = np.zeros_like(ml_coeff[0,:])
        for i, label in enumerate(self.corr_ao_labels):
            ax1.bar(inds, abs(ml_coeff[i,:])**2, 1, label=label, bottom=total)
            total+=abs(ml_coeff[i,:])**2

        ax1.set_ylabel('Total norm')
        ax1.set_xlabel('1-based MO index (near-degenerate set label)')
        ax1.legend()

        ax2 = ax1.twinx()
        energy = self.mo_energy[offset:offset+ncol]
        ax2.plot(inds, energy, color='k', linewidth=2, label='Energy')
        ax2.set_ylabel('MO energy / au')
        ax2.set_xticks(inds)

        ticklabels = []
        nset=0
        for i in inds:
            i0 = i+offset
            i1 = i0+1

            ticklabels.append('#{}'.format(i1))
            if not self.show_nd_labels: continue
            index = self.ind_in_nd_set(i1)
            if index > 0:
                # this orbital is part of a nearly-degenerate set
                ticklabels[-1]+='[{}]'.format(string.ascii_lowercase[nset])
            if index == 1:
                # this was the last orbital in the degenerate set
                nset+=1
        ax2.set_xticklabels(ticklabels)

        fig.tight_layout()
        plt.show()
