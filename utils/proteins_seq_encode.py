from json.tool import main
from webbrowser import get
from tape import ProteinBertModel, TAPETokenizer
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F
import os
import time
from tqdm import tqdm
from Bio import SeqIO
from Bio.PDB import PDBParser, Polypeptide
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.SeqUtils import seq1

model = ProteinBertModel.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')

def proteins_encode(sequence):
    token_ids = torch.tensor([tokenizer.encode(sequence)])
    output = model(token_ids)
    sequence_output = output[0]
    sequence_output = torch.mean(sequence_output, dim=1)
    #return torch.squeeze(sequence_output)
    return sequence_output

def get_feature_new():
    base_path = "./data/crossdock/case_study"
    pocket_path = "./data/crossdock/crossdock_data_process/final_filtered_train.csv"

    df = pd.read_csv(pocket_path)
    if df.empty:
        print("The final_filtered_train.csv file is empty.")
        return

    start_time = time.time()

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing pockets"):
        pdb_path = os.path.join(base_path, row.iloc[0])
        
        feature = proteins_encode(row.iloc[2])
        res = torch.squeeze(feature)
        npz_path = os.path.splitext(pdb_path)[0] + ".npz"
        torch.save(res, npz_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"All PDB files processed in {elapsed_time:.2f} seconds.")

def get_protein_sequence(pdb_path):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_path)
    sequence = ""

    for model in structure:
        for chain in model:
            for residue in chain:
                if Polypeptide.is_aa(residue, standard=True) and residue.has_id("CA"):
                    aa = seq1(residue.get_resname())
                    sequence += aa
    return sequence

if __name__ == "__main__":
    get_feature_new()