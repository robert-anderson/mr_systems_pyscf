from pyscf import M, mrpt, mcscf, scf, fci, symm
import sys, os
sys.path.append(os.path.abspath('.'))
from definitions import data as defs
import numpy as np
import pandas as pd

name = 'N2'
nelecas = 6

casorbs = defs[name]['occ_orbs']+defs[name]['vrt_orbs']
ncas = len(casorbs)

mol = M(**defs[name]['mol_kwargs'])
nelecb = (nelecas-mol.spin)//2
neleca = nelecas - nelecb

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

# target anion (-1) or cation (1)
tgt_charge = 1
# target a particular spin sector of the ion
tgt_spin = 1
tgt_spin_channel = int(tgt_spin>0)

assert tgt_charge in (-1, 1)
sqops = (fci.addons.des_a, fci.addons.des_b) if tgt_charge==1 else (fci.addons.cre_a, fci.addons.cre_b)

symids = myhf.get_orbsym()
cas_symids = symids[mc.ncore:mc.ncore+mc.ncas]


'''
Prepare ROHF solutions for each irrep
the irrep must be occupied in the tgt_charge==1 case
'''
def prepare_ion_rohf_solns():
    solns = dict()
    for symid in set(cas_symids):
        kw = defs[name]['mol_kwargs'].copy()
        kw['charge']+=tgt_charge
        kw['spin'] = tgt_spin
        mol = M(**kw)
        myhf_ion = scf.ROHF(mol)
        # irrep_nelec uses labels not symids as keys
        label = symm.irrep_id2name(mol.groupname, symid)
        myhf_ion.irrep_nelec = myhf.get_irrep_nelec()

        if tgt_charge==1:
            if label not in myhf_ion.irrep_nelec or myhf_ion.irrep_nelec[label][tgt_spin_channel]==0: continue

        # make mutable
        myhf_ion.irrep_nelec[label] = list(myhf_ion.irrep_nelec[label])
        myhf_ion.irrep_nelec[label][tgt_spin_channel] -= tgt_charge
        # make immutable again
        myhf_ion.irrep_nelec[label] = tuple(myhf_ion.irrep_nelec[label])

        myhf_ion.kernel(verbose=0)
        solns[symid] = myhf_ion
    return solns

ion_rohf_solns = prepare_ion_rohf_solns()


def do_mr_from_ci0(civec, na_nb, symid):
    mc = mcscf.CASSCF(ion_rohf_solns[symid], ncas, na_nb)
    mc.fcisolver.wfnsym = symid
    ecasscf, _, _, _, _ = mc.kernel(ci0=civec)
    enevpt2 = mrpt.NEVPT(mc).kernel()
    return ecasscf, enevpt2


ci = mc.fcisolver.ci
sqop = sqops[int(tgt_spin<0)]


nelecas_ion = [neleca, nelecb]
nelecas_ion[tgt_spin_channel]-=tgt_charge
nelecas_ion = tuple(nelecas_ion)

assert sum(nelecas_ion)==nelecas-tgt_charge


columns = ['iorb', 'symid', 'irrep', 'ecasscf', 'enevpt2', 'etot']

data = {column:[] for column in columns}

for irow, iorb in enumerate(range(mc.ncas)):
    symid = cas_symids[iorb]
    if symid not in ion_rohf_solns: continue
    ci_ion = sqop(ci, mc.ncas, nelecas_ion, iorb)
    ci_ion/=np.linalg.norm(ci_ion)
    emr = do_mr_from_ci0(ci_ion, nelecas_ion, symid)
    data['iorb'].append(iorb)
    data['symid'].append(symid)
    data['irrep'].append(symm.irrep_id2name(mol.groupname, symid))
    data['ecasscf'].append(emr[0])
    data['enevpt2'].append(emr[1])
    data['etot'].append(sum(emr))

df = pd.DataFrame(data)
df.sort_values(by=['etot'], inplace=True)
print('\nResults sorted by total energy:')
print(df)
