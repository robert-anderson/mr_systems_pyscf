'''
robert.anderson@kcl.ac.uk
Dec 2020

Definitions for a selection of molecular systems.
Each element contains three members: 
    doi, mol_kwargs, occ_orbs, and vrt_orbs.
doi points to the ref in which the parameters are detailed
mol_kwargs is contructed as a dict such that its ** expansion forms a
valid set of keyword arguments to the pyscf.M convenience method

the orbital index members can be integer (CAS not selected), or tuple
(CAS selected)
'''

data = {
    'N2': {
        'doi': None,
        'mol_kwargs': {
            'atom':
            '''
            N 0.000000 0.000000 0.000000
            N 0.000000 0.000000 1.400000
            ''',
            'basis': {'N': 'cc-pvdz'},
            'symmetry': True,
            'verbose': 4,
            'charge': 0,
            'spin': 0
        },
        # remember: occupied and virtual orbital indices are 1-based!
        'occ_orbs': (4, 5, 6),
        'vrt_orbs': (7, 8, 9),
    },
    'Cr2': {
        'doi': 'https://doi.org/10.1021/acs.jctc.6b00034',
        'mol_kwargs': {
            'atom':
            '''
            Cr 0.000000 0.000000 0.000000
            Cr 0.000000 0.000000 1.678800
            ''',
            'basis': {'Cr': 'aug-cc-pvtz'},
            'symmetry': True,
            'verbose': 4,
            'charge': 0,
            'spin': 0
        },
        # remember: occupied and virtual orbital indices are 1-based!
        'occ_orbs': (28, 29, 34, 35, 36, 37, 38, 39, 40, 41, 42, 46),
        'vrt_orbs': (47, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 87),
    },
    'Cu2O2(NH3)2 (peroxo)': {
        'doi': 'https://doi.org/10.1021/acs.jctc.6b00714',
        'mol_kwargs': {
            'atom':
            '''
            Cu 0.000000 1.800000 0.000000
            Cu 0.000000 -1.800000 0.000000
            O 0.000000 0.000000 0.700000
            O 0.000000 0.000000 -0.700000
            N 0.000000 3.800000 0.000000
            N 0.000000 -3.800000 0.000000
            H -0.939693 4.142020 0.000000
            H 0.939693 -4.142020 0.000000
            H 0.469846 4.142020 0.813798
            H -0.469846 -4.142020 -0.813798
            H 0.469846 4.142020 -0.813798
            H -0.469846 -4.142020 0.813798
            ''',
            'basis': {'Cu': 'ccpvtz', 'O': 'ccpvdz', 'N': 'ccpvdz', 'H': 'ccpvdz'},
            'symmetry': True,
            'verbose': 4,
            'charge': 2,
            'spin': 0
        },
        # remember: occupied and virtual orbital indices are 1-based!
        'occ_orbs': (28, 29, 34, 35, 36, 37, 38, 39, 40, 41, 42, 46),
        'vrt_orbs': (47, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 87),
    },
    'Cu2O2(NH3)2 (bis-mu-oxo)': {
        'doi': 'https://doi.org/10.1021/acs.jctc.6b00714',
        'mol_kwargs': {
            'atom':
            '''
            Cu 0.000000 1.400000 0.000000
            Cu 0.000000 -1.400000 0.000000
            O 0.000000 0.000000 1.150000
            O 0.000000 0.000000 -1.150000
            N 0.000000 3.400000 0.000000
            N 0.000000 -3.400000 0.000000
            H -0.939693 3.742020 0.000000
            H 0.939693 -3.742020 0.000000
            H 0.469846 3.742020 0.813798
            H -0.469846 -3.742020 -0.813798
            H 0.469846 3.742020 -0.813798
            H -0.469846 -3.742020 0.813798
            ''',
            'basis': {'Cu': 'ccpvtz', 'O': 'ccpvdz', 'N': 'ccpvdz', 'H': 'ccpvdz'},
            'symmetry': True,
            'verbose': 4,
            'charge': 2,
            'spin': 0
        },
        'occ_orbs': (27, 28, 30, 31, 32, 33, 34, 39, 43, 44, 45, 46),
        'vrt_orbs': (47, 48, 69, 70, 71, 72, 73, 78, 83, 86, 87, 100),
    },
    'Fe(P) 1A1g' : {
        'doi': 'https://doi.org/10.1021/acs.jctc.6b00714',
        'mol_kwargs': {
            'atom': 
            '''
            Fe 0.000000 0.000000 0.000000
            N 1.406594 1.406594 0.000000
            N -1.406594 1.406594 0.000000
            N 1.406594 -1.406594 0.000000
            N -1.406594 -1.406594 0.000000
            C 0.000000 3.398725 0.000000
            C 0.000000 -3.398725 0.000000
            C 3.398725 0.000000 0.000000
            C -3.398725 0.000000 0.000000
            C 2.485268 3.441695 0.000000
            C -2.485268 3.441695 0.000000
            C 2.485268 -3.441695 0.000000
            C -2.485268 -3.441695 0.000000
            C 3.441695 2.485268 0.000000
            C -3.441695 2.485268 0.000000
            C 3.441695 -2.485268 0.000000
            C -3.441695 -2.485268 0.000000
            C 1.223292 2.760499 0.000000
            C -1.223292 2.760499 0.000000
            C 1.223292 -2.760499 0.000000
            C -1.223292 -2.760499 0.000000
            C 2.760499 1.223292 0.000000
            C -2.760499 1.223292 0.000000
            C 2.760499 -1.223292 0.000000
            C -2.760499 -1.223292 0.000000
            H 0.000000 4.481408 0.000000
            H 0.000000 -4.481408 0.000000
            H 4.481408 0.000000 0.000000
            H -4.481408 0.000000 0.000000
            H 4.514620 2.604530 0.000000
            H -4.514620 2.604530 0.000000
            H 4.514620 -2.604530 0.000000
            H -4.514620 -2.604530 0.000000
            H 2.604530 4.514620 0.000000
            H -2.604530 4.514620 0.000000
            H 2.604530 -4.514620 0.000000
            H -2.604530 -4.514620 0.000000
            ''',
            'basis': {'Fe': 'ccpvtz', 'N': 'ccpvdz', 'H': 'ccpvdz', 'C': 'ccpvdz'},
            'symmetry': True,
            'verbose': 4,
            'charge': 0,
            'spin': 0
        },
        'occ_orbs': (49, 50, 52, 53, 54, 55, 56, 58, 59, 60, 82, 83),
        'vrt_orbs': (94, 95, 96, 125, 126, 127, 128, 133, 134, 141, 142, 214),
    },
}
