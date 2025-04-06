from rdkit import Chem
import json
import argparse
from autoencoder import autoencoder
import os
import pandas as pd

def encode(smiles_file, output_latent_file_path=None, encoder=None):
    model = autoencoder.load_model(model_version=encoder)

    # Input SMILES
    df = pd.read_csv(smiles_file)
    smiles_in = df['smiles'].tolist()
    
    # MUST convert SMILES to binary mols for the model to accept them (it re-converts them to SMILES internally)
    mols_in = [Chem.rdchem.Mol.ToBinary(Chem.MolFromSmiles(smiles)) for smiles in smiles_in]
    latent = model.transform(model.vectorize(mols_in))
    
    # Writing JSON data
    os.makedirs(os.path.dirname(output_latent_file_path), exist_ok=True)
    with open(output_latent_file_path, 'w') as f:
        json.dump(latent.tolist(), f)

    print('Encoding completed!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and train a model")
    
    parser.add_argument("--smiles-file", "-sf", default='./data/crossdock/crossdock_data_process/final_filtered_train.csv', help="The path to a data file.", type=str) # , required=True
    parser.add_argument("--output_latent_file_path", "-o", default='./storage/encoded_smiles.latent', help="Path to output smiles.", type=str)
    parser.add_argument("--encoder", default='chembl', help="The data set the pre-trained heteroencoder has been trained on [chembl|moses] DEFAULT:chembl", type=str)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    encode(**args)
