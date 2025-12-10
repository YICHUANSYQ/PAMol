"""
sample 30000 --- valid 14000~16000 --- random 500 cal vina score --- head 100
"""
from statistics import mean
from joblib import Parallel, delayed

from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP, qed
from evaluation.sascorer import *
from evaluation.score_func import *
from evaluation.similarity import *
from evaluation.docking_2 import *
from utils.crossdock.clean import *

import pandas as pd
import os
import json
import torch

RDLogger.DisableLog('rdApp.*')

sys.stdout = open('evaluate.log', 'w')

def evaluate(m, n):
    smile = m['smile']
    mol = m['mol']
    protein_filename = m['protein_filename']
    ligand_filename = m['ligand_file']
    rd_vina_score = m['rd_vina_score']

    try:
        _, g_sa = compute_sa_score(mol)
        print("Generate SA score:", g_sa)

        g_qed = qed(mol)
        print("Generate QED score:", g_qed)

        g_logP = MolLogP(mol)
        print("Generate logP:", g_logP)

        g_Lipinski = obey_lipinski(mol)
        print("Generate Lipinski:", g_Lipinski)
    except:
        print('mol error')
        return None

    receptor_file = os.path.join('/home/lxw/workspace/MMF2Drug/data/crossdock/test_data', '{}.pdbqt'.format(pdb_path.rstrip('.pdb')))
    out_dir = os.path.join('/home/lxw/workspace/MMF2Drug/data/crossdock/test_data', protein_filename.split('/')[0])
    index = n % 100
    try:
        g_vina_score = calculate_qvina2_score(receptor_file, smile, out_dir, smile_flag=True, return_rdmol=False, index=index)[0]
    except:
        return None
    print("Generate vina score:", g_vina_score)

    # g_high_affinity = 0
    # if float(g_vina_score) < float(rd_vina_score):
    #     g_high_affinity = 1

    metrics = {'SA': g_sa, 'QED': g_qed, 'logP': g_logP, 'Lipinski': g_Lipinski, 'vina': g_vina_score, 'rd_vina_score': rd_vina_score}
    result = {
        'smile': smile,
        'protein_file': protein_filename,
        'ligand_file': ligand_filename,
        'mol': mol,
        'metrics': metrics}

    return result

if __name__ == '__main__':
    file_path = "./data/crossdock/crossdock_data_process/final_filtered_test.csv"
    df = pd.read_csv(file_path)

    results_mol = []
    sa_list = []
    qed_list = []
    logP_list = []
    Lipinski_list = []
    vina_score_list = []
    diversity_list = []

    results = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows", position=0):
        pdb_path = row['pdb_path']
        
        gen_path = pdb_path.rstrip('.pdb')
        gen_path = os.path.join('./storage/cross_dock/fusion/proteins', '{}.latent'.format(gen_path))

        metrics_path = os.path.join("./data/crossdock/test_data", pdb_path.replace("_pocket10.pdb", "") + '.csv')
        metrics_df = pd.read_csv(metrics_path)
        rd_vina_score = metrics_df['vina'].mean()

        gen_smiles = []
        gen = []

        cleaner = MolCleaner()
        with open(gen_path, 'r') as file:
            gen_smiles_all = json.load(file)
        for index, smiles in enumerate(gen_smiles_all):
            try:
                smi = cleaner.process(smiles)
                if smi is not None and 20 <= len(smi) <= 120:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        gen_smiles.append(smi)
                        gen.append(mol)
            except:
                continue
        print(len(gen))
        data = []
        for i in range(len(gen)):
             data.append({
                "mol": gen[i],
                "smile": gen_smiles[i],
                "ligand_file": gen_path,
                "protein_filename": pdb_path,
                "rd_vina_score": rd_vina_score
            })

        count = 0
        for n, m in tqdm(enumerate(data), position=1, leave=False):
            result = evaluate(m, n)
            results.append(result)
        
        diversity_list.append(calculate_diversity(gen))
        print(diversity_list)

    for result in tqdm(results):
        if result is not None:
            results_mol.append(result)
            metrics = result['metrics']
            g_sa, g_qed, g_logP, g_Lipinski, g_vina = metrics['SA'], metrics['QED'], metrics['logP'], metrics[
                'Lipinski'], metrics['vina']

            sa_list.append(g_sa)
            qed_list.append(g_qed)
            logP_list.append(g_logP)
            Lipinski_list.append(g_Lipinski)
            if g_vina < 0:
                vina_score_list.append(g_vina)

    print('mean sa: %f' % mean(sa_list))
    print('mean qed: %f' % mean(qed_list))
    print('mean logP: %f' % mean(logP_list))
    print('mean Lipinski: %f' % np.mean(Lipinski_list))
    print('mean vina: %f' % mean(vina_score_list))
    print('mean diversity: %f' % mean(diversity_list))

    sa_list = torch.tensor(sa_list)
    qed_list = torch.tensor(qed_list)
    logP_list = torch.tensor(logP_list)
    Lipinski_list = torch.tensor(Lipinski_list)
    vina_score_list = torch.tensor(vina_score_list)
    metrics_list = {
        'diversity': diversity_list,
        'sa': sa_list,
        'qed': qed_list,
        'logP': logP_list,
        'Lipinski': Lipinski_list,
        'vina': vina_score_list}

    save_mol_result_path = os.path.join('./data/crossdock/test_data', 'mol_results.latent')
    with open(save_mol_result_path, 'wb') as f:
        pickle.dump(results_mol, f)
        f.close()

    save_metric_result_path = os.path.join('./data/crossdock/test_data', 'metric_results.latent')
    with open(save_metric_result_path, 'wb') as f:
        pickle.dump(metrics_list, f)
        f.close()

    sys.stdout.close()
    sys.stdout = sys.__stdout__
# CUDA_VISIBLE_DEVICES=1 python evaluate.py