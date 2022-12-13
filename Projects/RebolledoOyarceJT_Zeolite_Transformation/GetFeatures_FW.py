import numpy as np

from ase.io import read, write
from ase.calculators.vasp import Vasp

from dscribe.descriptors import CoulombMatrix, SineMatrix, EwaldSumMatrix, SOAP


def get_SOAPDescriptors(atoms, rcut = 6.0, nmax = 8, lmax = 6, periodic = True):
    """Generate Smooth Overlap of Atomic Positions Descriptor 
        
        Parameters
        ----------
        atoms : object
            atoms is a collection of atoms in ASE format
            
        positions : list
            Positions where to calculate SOAP. Can be provided as cartesian positions or atomic indices. 
            If no positions are defined, the SOAP output will be created for all atoms in the system. 
            When calculating SOAP for multiple systems, provide the positions as a list for each system.
            
        rcut : float
            A cutoff for local region in angstroms. Should be bigger than 1 angstrom.
            
        nmax : int
            The number of radial basis functions.
            
        lmax : int
            The maximum degree of spherical harmonics.
            
        periodic : bool
            Set to true if you want the descriptor output to respect the periodicity of the atomic systems 
            (see the pbc-parameter in the constructor of ase.Atoms).
            
            
        Returns
        -------
        ewald : ndarray
            SOAP Descriptors, (n, m) narray where n is the number of atoms and m is the number of descriptors.            
        """
    
    # Define Chemical Species in the Structure
    
    species = list(set(atoms.get_chemical_symbols()))

#     n_atoms = atoms.get_positions()
#     n_atoms = n_atoms.shape[0]
    
#     print(n_atoms)
    # Setting up the SOAP descriptor
    soap = SOAP(
                species=species,
                periodic=periodic,
                rcut=rcut,
                nmax=nmax,
                lmax=lmax,
               )
    
    
    
    # Create SOAP output for the system
    soap_descriptor = soap.create(atoms)
    

    # Return
    return soap_descriptor

