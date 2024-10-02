from tqdm import tqdm
import os
import os.path as osp
import pickle
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

import torch
from torch_geometric.data import InMemoryDataset, Data


class Molecule_Net(InMemoryDataset):
    """
    A dataset loader for the GEOM dataset, utilizing RDKit to handle conformers 
    and molecular structures. This script follows the structure of the QM9 dataset 
    handler but adapted for the GEOM data format.
    """

    def __init__(self, root: str, transform=None, pre_transform=None):
        super(Molecule_Net, self).__init__(root, transform, pre_transform)
        # self.raw_dir = ['scratch/ssd004/scratch/username']
        # self.process()
        # self.processed_paths = ['/scratch/ssd004/scratch/username/molecule_net']
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Specify the tar.gz file name that contains the raw data
        return ['molecule_net.tar.gz']

    @property
    def processed_file_names(self):
        # Name of the file after processing the data
        return ['molecule_net_data.pt']

    def download(self):
        # Download the tar file from the specified URL
        # Placeholder: You need to add your own link
        pass

    def process(self):
        data_list = []

        # Extract and load the pickle files containing RDKit mol objects
        # rdkit_folder = osp.join(self.raw_dir, 'molecule_net')
        rdkit_folder = '/scratch/ssd004/scratch/username/molecule_net'
        for subdir, _, files in os.walk(rdkit_folder):
            for file in tqdm(files):
                if file.endswith('.pickle'):
                    # Load the pickle file which contains conformer data as RDKit mol objects
                    with open(osp.join(subdir, file), 'rb') as f:
                        mol_data = pickle.load(f)

                        # for _ in mol_data['conformers']:
                        #     mol = mol_data['rd_mol']
                        #     energy = mol_data['totalenergy']

                        #     # Process the RDKit molecule and convert to torch_geometric Data
                        #     data = self.mol_to_data(mol, energy)
                        #     data_list.append(data)
                    
                    mol = mol_data['conformers'][0]['rd_mol']
                    energy = mol_data['conformers'][0]['totalenergy']

                    # Process the RDKit molecule and convert to torch_geometric Data
                    data = self.mol_to_data(mol, energy)
                    data_list.append(data)

        # Convert the list of Data objects into PyTorch Geometric format
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def mol_to_data(self, mol, energy):
        """
        Converts an RDKit mol object into a torch_geometric Data object.
        This includes atom and bond features as well as 3D conformer coordinates.
        """
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        atom_features = []
        pos = []

        for atom in mol.GetAtoms():
            atom_features.append(atom.GetAtomicNum())
            pos.append(list(mol.GetConformer().GetAtomPosition(atom.GetIdx())))

        atom_mapping = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4}
        atom_features = torch.tensor(atom_features, dtype=torch.int64)
        num_nodes = len(atom_features)
        mapped_features = torch.full_like(atom_features, -1)
        for atomic_number, mapped_index in atom_mapping.items():
            mapped_features[atom_features == atomic_number] = mapped_index
        num_classes = len(atom_mapping)
        x_i = F.one_hot(mapped_features.clamp(min=0), num_classes=num_classes).float()
        x_i[mapped_features == -1] = 0

        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            row += [i, j]
            col += [j, i]
            edge_type += 2*[bonds[bond.GetBondType()]]

        pos = torch.tensor(pos, dtype=torch.float)
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

        smiles = Chem.MolToSmiles(mol)

        data = Data(
            x=x_i,
            atomic_numbers=atom_features,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr, 
            smiles=smiles, 
            y=energy,
            natoms=num_nodes
        )
        
        return data
