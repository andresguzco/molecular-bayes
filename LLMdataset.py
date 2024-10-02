import torch
import pandas as pd
from tqdm import tqdm
from my_datasets.QM9 import QM9
from my_datasets.QM7b import QM7b
from my_datasets.GEOM import GEOM
from my_datasets.Molecule_Net import Molecule_Net

def save_smiles_and_energy(dataset, save_path):
    data_list = []
    
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        smiles = data.smiles
        if isinstance(dataset, QM9):
            energy = data.y[:, 4].item() if torch.is_tensor(data.y) else data.y
        else:
            energy = data.y.item() if torch.is_tensor(data.y) else data.y
        data_list.append({"SMILES": smiles, "energy": energy})
    
    df = pd.DataFrame(data_list)
    df.to_csv(save_path, index=False)
    print(f"Saved CSV file at {save_path}")


def process_datasets():

    dataset_paths = {
        # "rdkit_folder": "my_datasets/rdkit_folder",  
        # "moelcule_net": "my_datasets/moelcule_net",
        # "qm7b": "my_datasets/qm7b", 
        "qm9": "my_datasets/qm9", 
    }

    for name, data_path in dataset_paths.items():

        print(f"Processing {name}...")

        if name == "qm7b":
            dataset = QM7b(data_path)
        elif name == "qm9":
            dataset = QM9(data_path, split="train")
        elif name == "rdkit_folder":
            dataset = GEOM(data_path)
        elif name == "moelcule_net":
            dataset = Molecule_Net(data_path)
        else:
            raise ValueError(f"Dataset {name} not found.")  
        
        csv_save_path = f"my_datasets/{name}/smiles_energy.csv"
        save_smiles_and_energy(dataset, csv_save_path)

if __name__ == "__main__":
    process_datasets()