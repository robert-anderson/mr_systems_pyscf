'''
A demonstration that N-1 electron WFs created by the application of the annihilation
operator of an arbitrary CAS spinorbital are not spatially-symmetric.
'''

import pyscf
import numpy as np
from pyscf.fci import addons

mol = pyscf.M(atom = 'N 0 0 0; N 0 0 1.4', basis = 'ccpvdz', symmetry=1, symmetry_subgroup='D2h', spin = 0)

myhf = mol.RHF().run()
'''
These labels are unchanged in the CASCI object.
see: https://sunqm.github.io/pyscf/mcscf.html
"In the MCSCF calculation and the canonicalization, the irreducible representation label of the orbitals 
will not be changed. They are always the same to the symmetry labels of the input orbitals."
'''
labels = myhf.get_orbsym()

no, ne = 6, 6
na, nb = 3, 3

mycas = myhf.CASCI(no, ne)
mycas.kernel()

ci = mycas.fcisolver.ci
# the fcisolver sets the value of the irrep label id to which the solution belongs
ci_symd = addons.symmetrize_wfn(ci, 6, (na, nb), labels[mycas.ncore:], wfnsym=mycas.fcisolver.wfnsym)
# if that wfnsym label is indeed correct, the above call should only have zeroed identically zero elements in the ci array
assert np.allclose(ci, ci_symd) # this passes
print('N electron WF has symmetry label', mycas.fcisolver.wfnsym)

# now apply the annihilator for an arbitrary MO (0-indexed within CAS)
iorb_des = 3
ci_n_minus_1 = addons.des_a(ci, 6, (na, nb), iorb_des)
na -= 1

symfound = False
for ilabel in range(max(labels)):
    ci_n_minus_1_symd = addons.symmetrize_wfn(ci_n_minus_1, 6, (na, nb), labels[mycas.ncore:], wfnsym=ilabel)
    if np.allclose(ci_n_minus_1, ci_n_minus_1_symd):
        print("N-1 electron WF has symmetry label", ilabel)
        symfound = True
        break

if not symfound: print("N-1 electron WF belongs to no point group")
