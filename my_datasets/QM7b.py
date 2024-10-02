from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import add_self_loops, remove_self_loops, is_undirected

from rdkit import Chem
from rdkit.Chem import AllChem

class QM7b(InMemoryDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.mat'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'qm7.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_url(self.url, self.raw_dir) 
    
    def process(self) -> None:
        from scipy.io import loadmat
        atom_mapping = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4}
        reverse_atom_mapping = {v: k for k, v in atom_mapping.items()}

        data_general = loadmat(self.raw_paths[0])
        coulomb_matrix = torch.from_numpy(data_general['X'])
        target = torch.from_numpy(data_general['T'])
        atomic_numbers = torch.from_numpy(data_general['Z'])
        coordinates = torch.from_numpy(data_general['R'])

        data_list = []
        for i in range(target.shape[1]):
            edge_index = coulomb_matrix[i].nonzero(as_tuple=False).t().contiguous()
            edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]].unsqueeze(1)
            edge_attr = torch.cat((edge_attr, torch.zeros(edge_attr.size(0), 3)), dim=1)

            y = target[:, i].view(1, -1)
            atoms_i = atomic_numbers[i, :].view(-1).int()
            valid_mask = (atoms_i != 0)
            atoms_i_filtered = atoms_i[valid_mask]

            mapped_atoms_i = torch.tensor([atom_mapping[int(a.item())] for a in atoms_i_filtered])
            coordinates_filtered = coordinates[i, :, :][valid_mask, :]

            x_i = F.one_hot(mapped_atoms_i.long(), num_classes=len(atom_mapping)).float()

            # Create RDKit molecule and generate SMILES
            atomic_numbers_list = [reverse_atom_mapping[int(a.item())] for a in mapped_atoms_i]
            coordinates_list = coordinates_filtered.tolist()
            smiles = self._generate_smiles(atomic_numbers_list, coordinates_list)

            data = Data(
                x=x_i,
                atomic_numbers=atoms_i_filtered,
                pos=coordinates_filtered,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                natoms=atoms_i_filtered.size(0),
                smiles=smiles  # Add SMILES to the data object
            )

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

    def _generate_smiles(self, atomic_numbers, coordinates):
        """
        Helper function to generate SMILES from atomic numbers and coordinates.
        """
        mol = Chem.RWMol()

        # Add atoms
        for atomic_num in atomic_numbers:
            atom = Chem.Atom(atomic_num)
            mol.AddAtom(atom)
        
        # Add 3D coordinates (optional, SMILES doesn't require 3D information)
        conformer = Chem.Conformer(len(atomic_numbers))
        for i, coord in enumerate(coordinates):
            conformer.SetAtomPosition(i, coord)
        
        mol.AddConformer(conformer)

        # Optionally sanitize molecule and generate SMILES
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)

        return smiles
