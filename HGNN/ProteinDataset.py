import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import utils.hypergraph_utils as hgut
from hgnn_model import HGNN
from Bio.PDB import PDBParser, DSSP, Polypeptide
from Bio.SeqUtils import seq1
import time
from tqdm import tqdm
import argparse
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

def create_hyperedge(pdb_filename, cutoff, k):

    parser = PDBParser()
    structure = parser.get_structure("protein_structure", pdb_filename)

    if len(structure) == 0:
        raise ValueError(f"{pdb_filename}")
    model = structure[0]
    dssp = DSSP(model, pdb_filename, dssp='mkdssp')

    coords = []
    amino_acids_num = 0

    first_vertex_feature_vectors = []
    
    second_vertex_feature_vectors = []

    third_vertex_feature_vectors = []
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    one_hot_vectors = {aa: [0]*20 for aa in amino_acids}
    for i, aa in enumerate(amino_acids):
        one_hot_vectors[aa][i] = 1

    fourth_vertex_feature_vectors = []
    amino_acid_chemical_properties = {
        'ALA': [0, 0, 0],
        'CYS': [0, 0, 1],
        'ASP': [0, 0, 0],
        'GLU': [0, 0, 0],
        'PHE': [1, 0, 0],
        'GLY': [0, 0, 0],
        'HIS': [1, 0, 0],
        'ILE': [0, 0, 0],
        'LYS': [0, 0, 0],
        'LEU': [0, 0, 0],
        'MET': [0, 0, 1],
        'ASN': [0, 0, 0],
        'PRO': [0, 0, 0],
        'GLN': [0, 0, 0],
        'ARG': [0, 0, 0],
        'SER': [0, 1, 0],
        'THR': [0, 1, 0],
        'VAL': [0, 0, 0],
        'TRP': [1, 0, 0],
        'TYR': [1, 1, 0]
    }

    dssp_codes = 'HBEGITS-'
    vector_map = {code: np.zeros(len(dssp_codes)) for code in dssp_codes}
    for i, code in enumerate(dssp_codes):
        vector_map[code][i] = 1

    for chain in model:
        for residue in chain:
            if Polypeptide.is_aa(residue, standard=True) and residue.has_id("CA") and (residue.parent.id, residue.id) in dssp:
                ca = residue["CA"]
                coords.append(ca.get_coord())

                amino_acids_num += 1

                ss = dssp[(residue.parent.id, residue.id)][2]
                vector = vector_map.get(ss, np.zeros(len(dssp_codes)))
                first_vertex_feature_vectors.append(vector)

                rasa = dssp[(residue.parent.id, residue.id)][3]
                phi = dssp[(residue.parent.id, residue.id)][4]
                psi = dssp[(residue.parent.id, residue.id)][5]
                second_vertex_feature_vectors.append([rasa, phi, psi])

                resname = residue.get_resname()
                aa = seq1(residue.get_resname())
                third_vertex_feature_vectors.append(one_hot_vectors[aa])
                fourth_vertex_feature_vectors.append(amino_acid_chemical_properties[resname])

    coords = np.array(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))

    space_structure_hyperedges = np.array((distances >= 0) & (distances <= cutoff))
    
    spatial_structure_correlation_matrix = np.exp(-(distances**2 / np.mean(distances**2, axis=-1, keepdims=True)))
    for i in range(space_structure_hyperedges.shape[0]):
        for j in range(space_structure_hyperedges.shape[1]):
            if space_structure_hyperedges[i][j] == 0:
                spatial_structure_correlation_matrix[i][j] = 0

    sequence_structure_hyperedges = []
    for i in range(amino_acids_num):
        hyperedge = np.zeros(amino_acids_num)
        if i + k < amino_acids_num:
            hyperedge[i:i + k] = 1
        else:
            hyperedge[i:amino_acids_num] = 1
        sequence_structure_hyperedges.append(hyperedge)
    sequence_structure_hyperedges = np.array(sequence_structure_hyperedges).T
    
    V = sequence_structure_hyperedges.shape[0]
    sequence_structure_correlation_matrix = np.zeros((V, V))
    for e in range(sequence_structure_hyperedges.shape[1]):
        amino_acids_in_edge = np.where(sequence_structure_hyperedges[:, e] == 1)[0]
        for i in amino_acids_in_edge:
            for k in amino_acids_in_edge:
                sequence_structure_correlation_matrix[i, k] = 1
    
    first_hyperedges = np.hstack((space_structure_hyperedges.astype(float).T, sequence_structure_hyperedges))
    structure_matrix = np.hstack((spatial_structure_correlation_matrix.T, sequence_structure_correlation_matrix))
    
    first_vertex_feature_vectors = np.array(first_vertex_feature_vectors)
    second_vertex_feature_vectors = np.array(second_vertex_feature_vectors)
    third_vertex_feature_vectors = np.array(third_vertex_feature_vectors)
    fourth_vertex_feature_vectors = np.array(fourth_vertex_feature_vectors)
    vertex_feature_vectors = np.hstack((first_vertex_feature_vectors, second_vertex_feature_vectors, third_vertex_feature_vectors, fourth_vertex_feature_vectors))

    dis = vertex_feature_vectors[:, np.newaxis, :] - vertex_feature_vectors[np.newaxis, :, :]
    os_dis = np.sqrt(np.sum(dis**2, axis=-1))
    second_feature_hyperedges = np.array((os_dis >= 0) & (os_dis <= np.mean(os_dis))).astype(float).T

    all_hyperedges = np.hstack((first_hyperedges, second_feature_hyperedges))

    return all_hyperedges, vertex_feature_vectors, structure_matrix

def process_output_file(input_path, output_path):
    all_outputs = []

    with open(input_path, 'r') as infile:
        for line in infile:
            try:
                data = json.loads(line.strip())
                all_outputs.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}. Error: {e}")

    with open(output_path, 'w') as outfile:
        json.dump(all_outputs, outfile)
    print(f"Processed data saved to: {output_path}")

def get_crossdocked_protein_hypergraph_feature(base_path, split_by_name_file, output_latent_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(split_by_name_file)
    data = df.iloc[:, 0]
    
    # Open the output file in write mode (overwrite or create new file)
    with open(output_latent_file_path, 'w') as f:
        for pocket in tqdm(data, desc="Processing PDB Files", unit="pocket"):
            pdb_filename = os.path.join(base_path, pocket)
    
            cutoff = 5.0  # Distance threshold (Å) for spatial structure hyperedges
            k = 5  # Number of adjacent amino acids in each hyperedge (sequence structure hyperedge)
            all_hyperedges, vertex_feature_vectors, structure_matrix = create_hyperedge(pdb_filename, cutoff, k)

            fts = torch.tensor(np.hstack((vertex_feature_vectors, structure_matrix))).float().to(device)
            G = torch.tensor(hgut.generate_G_from_H(all_hyperedges)).float().to(device)

            model_ft = HGNN(in_ch=fts.shape[-1], out_size=384, n_hid=256, dropout=0.5)
            model_ft = model_ft.to(device)

            out_put = model_ft(fts, G)

            # Immediately write the output to the file
            json.dump(out_put.tolist(), f)  # .tolist() to ensure it's in a JSON serializable format
            f.write("\n")  # Add newline to separate outputs

    print(f"Latent features saved to {output_latent_file_path}")

    process_output_file(output_latent_file_path, output_latent_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and train a model")
    
    ## crossdocked2020 datasets
    parser.add_argument("--base-path", "-bp", default='./data/crossdock/case_study', help="The path to a data file.", type=str)
    parser.add_argument("--split-by-name-file", "-sbnf", default='./data/crossdock/crossdock_data_process/final_filtered_case_study.csv', help="The path to a data file.", type=str)
    parser.add_argument("--output-latent-file-path", "-o", default='./storage/encoded_proteins_hypergraph_case_study.latent', help="Path to output encode protein hypergraph.", type=str)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    start_time = time.time()
    get_crossdocked_protein_hypergraph_feature(**args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"All Protein encode hypergraph in {elapsed_time:.2f} seconds.")