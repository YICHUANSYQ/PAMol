import json
import pandas as pd
import os
import torch
from tqdm import tqdm

def get_proteins_feature_crossdock():
    pocket_path = "./data/crossdock/crossdock_data_process/final_filtered_train.csv"
    encoded_proteins_path = './storage/encoded_proteins_seq_train.latent'

    df = pd.read_csv(pocket_path)
    if df.empty:
        print("The final_filtered_train.csv file is empty.")
        return
    
    feature_matrix = torch.zeros((df.shape[0], 768))
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing pockets"):
        pdb_path = os.path.join(pocket_path, row.iloc[0])

        npz_path = os.path.splitext(pdb_path)[0] + ".npz"
        feature = torch.load(npz_path)
        feature_matrix[index] = feature

    res = feature_matrix.tolist()
    
    with open(encoded_proteins_path, 'w') as f:
        json.dump(res, f)

if __name__ == "__main__":
    get_proteins_feature_crossdock()