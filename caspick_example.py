from caspick import CasPicker, np
from pyscf import M, scf
from definitions import data as defs

corr_ao_labels = [
   '0 Cr 3dxy', '0 Cr 3dyz', '0 Cr 3dxz', '0 Cr 3dz^2', '0 Cr 3dx2-y2',
   '1 Cr 3dxy', '1 Cr 3dyz', '1 Cr 3dxz', '1 Cr 3dz^2', '1 Cr 3dx2-y2',
   '0 Cr 4dxy', '0 Cr 4dyz', '0 Cr 4dxz', '0 Cr 4dz^2', '0 Cr 4dx2-y2',
   '1 Cr 4dxy', '1 Cr 4dyz', '1 Cr 4dxz', '1 Cr 4dz^2', '1 Cr 4dx2-y2']

system_name = 'Cr2'

mol = M(**defs[system_name]['mol_kwargs'])
rhf = scf.RHF(mol)
rhf.kernel()

cp = CasPicker(rhf, corr_ao_labels, (-2.5, 0.4))

cp.plot('occ')
cp.plot('vrt')
