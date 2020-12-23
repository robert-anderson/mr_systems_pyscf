from pyscf import M, mrpt, mcscf, scf
import sys, os
sys.path.append(os.path.abspath('.'))
from definitions import data as defs

name = 'N2'
nelecas = 6

casorbs = defs[name]['occ_orbs']+defs[name]['vrt_orbs']
ncas = len(casorbs)

mol = M(**defs[name]['mol_kwargs'])

myhf = scf.ROHF(mol)
myhf.kernel()

'''
do CASSCF+NEVPT2 on neutral molecule
'''
mc = mcscf.CASSCF(myhf, ncas, nelecas)
mc.canonicalization = True
casorbs = defs[name]['occ_orbs']+defs[name]['vrt_orbs']
mc.sort_mo(casorbs)

mc.kernel()
mrpt.NEVPT(mc).kernel()


'''
do same for all point group sectors of the anion
'''
anion_spin = 1

kw = defs[name]['mol_kwargs'].copy()
kw['charge']+=1
kw['spin']+=anion_spin
nelecas-=1
mol = M(**kw)
myhf = scf.ROHF(mol)
myhf.kernel()

results = {}
for irrep in mol.irrep_name:
    print('Performing CASSCF+NEVPT2 for fermion irrep {}'.format(irrep))
    mc2 = mcscf.CASSCF(myhf, ncas, nelecas)
    mc2.canonicalization = True
    mc2.fcisolver.wfnsym = irrep
    casorbs = defs[name]['occ_orbs']+defs[name]['vrt_orbs']
    mc2.sort_mo(casorbs)
    
    # do CASSCF using optimized orbs of neutral system as initial guess
    ecasscf, _, _, _, _ = mc2.kernel(mc.mo_coeff)
    enevpt2 = mrpt.NEVPT(mc2).kernel()

    results[irrep] = {'casscf': ecasscf, 'nevpt2': enevpt2}

best = None
for k, v in results.items():
    tot = v['casscf']+v['nevpt2']
    if best is None or best[1]>tot:
        best = (k, tot)
    print(k, v)

print('Anion ground state spatial symmetry for spin={}: {}'.format(anion_spin, best[0]))
