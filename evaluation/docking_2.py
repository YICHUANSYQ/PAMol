import os
import subprocess
import re
import torch
from pathlib import Path
import argparse
from loguru import logger

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
from tqdm import tqdm
import time


def CaculateAffinity(smi, file_protein='./1zys.pdb', file_lig_ref = './1zys_D_199.sdf', out_path = './', prefix=''):
    try:
        mol = Chem.MolFromSmiles(smi)
        m2=Chem.AddHs(mol)
        AllChem.EmbedMolecule(m2)
        m3 = Chem.RemoveHs(m2)
        file_output = os.path.join(out_path, prefix + str(time.time())+ '.pdb')
        Chem.MolToPDBFile(m3, file_output)
        
        smina_cmd_output = os.path.join(out_path, prefix + str(time.time()))
        launch_args = ["smina", "-r", file_protein, "-l", file_output, "--autobox_ligand", 
                    file_lig_ref, "--autobox_add", "10", "--seed", "1000", "--exhaustiveness", "9",">>", smina_cmd_output]
        launch_string = ' '.join(launch_args)
        logger.info(launch_string)
        p = subprocess.Popen(launch_string, shell=True, stdout=subprocess.PIPE)
        p.communicate()

        affinity = 500
        with open(smina_cmd_output, 'r') as f:
            for lines in f.readlines():
                lines = lines.split()
                if len(lines) == 4 and lines[0] == '1':
                    affinity = float(lines[1])
                
        p = subprocess.Popen('rm -rf ' + smina_cmd_output, shell=True, stdout=subprocess.PIPE)
        p.communicate()
        p = subprocess.Popen('rm -rf ' + file_output, shell=True, stdout=subprocess.PIPE)
        p.communicate()
    
    except:
        affinity = 500

    if affinity == 500:
        logger.error('affinity error')

    return affinity


def calculate_smina_score(pdb_file, sdf_file):
    # add '-o <name>_smina.sdf' if you want to see the output
    out = os.popen(f'smina.static -l {sdf_file} -r {pdb_file} '
                   f'--score_only').read()
    matches = re.findall(
        r"Affinity:[ ]+([+-]?[0-9]*[.]?[0-9]+)[ ]+\(kcal/mol\)", out)
    return [float(x) for x in matches]


def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
    os.popen(f'/home/lxw/.conda/envs/MMF2Drug/bin/obabel {sdf_file} -O {pdbqt_outfile} -p 7.4').read()
    return pdbqt_outfile


def smi_to_pdb(smi, save_path):
    try:
        mol = pybel.readstring("smi", smi)
        # strip salt 
        mol.OBMol.StripSalts(10)
        mols = mol.OBMol.Separate()

        mol = pybel.Molecule(mols[0])
        for imol in mols:
            imol = pybel.Molecule(imol)
            if len(imol.atoms) > len(mol.atoms):
                mol = imol

        # Generate 3D coordinates, force field optimization
        mol.addh()
        mol.make3D(forcefield='mmff94', steps=100)
        mol.localopt()
        mol.write(format='pdb', filename=str(save_path), overwrite=True)
        return 1
    except:
        print(f"Tranformation of {smi} failed! ")
        return 0


def pdb_to_pdbqt(ligand_pdb_file, ligand_pdbqt_file):
    """Convert PDB to PDBQT using prepare_ligand4.py"""
    try:
        command = [
            "/home/lxw/.conda/envs/adt/bin/python",
            "/home/lxw/.conda/envs/adt/bin/prepare_ligand4.py",
            "-l", ligand_pdb_file,
            "-o", ligand_pdbqt_file
        ]
        subprocess.run(command, check=True)
    except:
        print(f"Tranformation of {ligand_pdb_file} failed! ")


def get_center_from_pdbqt(file_path):
    center_coords = {"CENTER_X": None, "CENTER_Y": None, "CENTER_Z": None}
    atom_coords = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("REMARK") and any(key in line for key in center_coords):
                parts = line.split()
                for key in center_coords.keys():
                    if key in line:
                        center_coords[key] = float(parts[-1])
            
            elif line.startswith("ATOM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atom_coords.append((x, y, z))

    if all(center_coords.values()):
        return tuple(center_coords[key] for key in ["CENTER_X", "CENTER_Y", "CENTER_Z"])
    
    if atom_coords:
        x_avg = sum(coord[0] for coord in atom_coords) / len(atom_coords)
        y_avg = sum(coord[1] for coord in atom_coords) / len(atom_coords)
        z_avg = sum(coord[2] for coord in atom_coords) / len(atom_coords)
        return (x_avg, y_avg, z_avg)
    
    return None


def calculate_qvina2_score(receptor_file, sdf_file, out_dir, size=126,
                           exhaustiveness=16, smile_flag=False, return_rdmol=False, index=None):
    receptor_file = Path(receptor_file)
    receptor_name = os.path.basename(receptor_file)[:4]
    

    if receptor_file.suffix == '.pdb':
        # prepare receptor, requires Python 2.7
        receptor_pdbqt_file = Path(out_dir, receptor_file.stem + '.pdbqt')
        # os.popen(f'/home/lxw/.conda/envs/adt/bin/prepare_receptor4.py -r {receptor_file} -O {receptor_pdbqt_file}')
        command = [
            "/home/lxw/.conda/envs/adt/bin/python",
            "/home/lxw/.conda/envs/adt/bin/prepare_receptor4.py",
            "-r", receptor_file,
            "-o", receptor_pdbqt_file
        ]
        subprocess.run(command, check=True)
    else:
        receptor_pdbqt_file = receptor_file

    scores = []
    rdmols = []  # for if return rdmols
    if smile_flag == True:
        ligand_name = f'{receptor_name}_{index}'
        os.makedirs(os.path.join(out_dir, receptor_name), exist_ok=True)
        ligand_pdb_file = os.path.join(out_dir, receptor_name, ligand_name + '.pdb')
        if not os.path.exists(ligand_pdb_file):
            smi_to_pdb(sdf_file, ligand_pdb_file)

        ligand_pdbqt_file = Path(out_dir, receptor_name, ligand_name + '.pdbqt')
        out_sdf_file = Path(out_dir, receptor_name, ligand_name + '_out.sdf')
        out_pdbqt_file = Path(out_dir, receptor_name, ligand_name + '_out.pdbqt')

        # you have to assdign your own mol envrionment
        if out_pdbqt_file.exists(): # False and 
            with open(out_pdbqt_file, 'r') as file:
                content = file.read()
            
            affinities = re.findall(r'^\s*\d+\s+(-?\d+\.\d+)', content, re.MULTILINE)
            lowest_affinity = 0
            if affinities:
                lowest_affinity = min(map(float, affinities))
            scores.append(lowest_affinity)
        else:
            pdb_to_pdbqt(ligand_pdb_file, ligand_pdbqt_file)

            center = get_center_from_pdbqt(receptor_pdbqt_file)
            cx, cy, cz = 0, 0, 0
            if center:
                cx = center[0]
                cy = center[1]
                cz = center[2]
            # # 添加氢原子以获得更准确的构象
            # mol = Chem.MolFromSmiles(sdf_file)
            # mol = Chem.AddHs(mol)
            # # 生成 3D 坐标
            # if AllChem.EmbedMolecule(mol) != 0:
            #     pass
            # if AllChem.UFFOptimizeMolecule(mol) != 0:
            #     pass
            # cx, cy, cz = mol.GetConformer().GetPositions().mean(0)

            # run QuickVina 2.1 -w
            out = os.popen(
                f'/home/lxw/qvina2.1 --receptor {receptor_pdbqt_file} '
                f'--ligand {ligand_pdbqt_file} '
                f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
                f'--size_x {size} --size_y {size} --size_z {size} '
                f'--exhaustiveness {exhaustiveness}'
            ).read()

            affinities = re.findall(r'^\s*\d+\s+(-?\d+\.\d+)', out, re.MULTILINE)
            lowest_affinity = 0
            if affinities:
                lowest_affinity = min(map(float, affinities))
            scores.append(lowest_affinity)

        if return_rdmol:
            rdmol = Chem.SDMolSupplier(str(out_sdf_file))[0]
            rdmols.append(rdmol)
    else:
        pdb_flag = False
        if type(sdf_file) == str:
            sdf_file = Path(sdf_file)
            suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
            pdb_flag = True
        else:
            suppl = [sdf_file]
            ligand_name = f'{receptor_name}_{index}'
            os.makedirs(os.path.join(out_dir,receptor_name), exist_ok=True)
            ligand_file = os.path.join(out_dir,receptor_name, ligand_name + '.sdf')
            if not Path(ligand_file).exists() or Path(ligand_file).stat().st_size== 0:
                sdf_writer = Chem.SDWriter(ligand_file)
                sdf_writer.write(sdf_file)
                sdf_writer.close()
            sdf_file = ligand_file

        for i, mol in enumerate(suppl):  # sdf file may contain several ligands
            if index is not None:
                i = index
            ligand_name = f'{receptor_name}_{i}'
            ligand_name = os.path.basename(sdf_file)
            ligand_name = os.path.basename(sdf_file).split('.sdf')[0]
            ligand_name = f'{ligand_name}_{i}'
            
            # prepare ligand
            if pdb_flag:
                ligand_pdbqt_file = Path(out_dir, ligand_name + '.pdbqt')
                out_sdf_file = Path(out_dir, ligand_name + '_out.sdf')
                out_pdbqt_file = Path(out_dir, ligand_name + '_out.pdbqt')
            else:
                ligand_pdbqt_file = Path(out_dir, receptor_name, ligand_name + '.pdbqt')
                out_sdf_file = Path(out_dir, receptor_name, ligand_name + '_out.sdf')
                out_pdbqt_file = Path(out_dir, receptor_name, ligand_name + '_out.pdbqt')

            # you have to assdign your own mol envrionment
            if False and out_pdbqt_file.exists() and not out_sdf_file.exists():
                os.popen(f'/home/lxw/.conda/envs/MMF2Drug/bin/obabel {out_pdbqt_file} -O {out_sdf_file}').read()
            if False and out_sdf_file.exists() and out_sdf_file.stat().st_size != 0:
                print(out_sdf_file)
                with open(out_sdf_file, 'r') as f:
                    scores.append(
                        min([float(x.split()[2]) for x in f.readlines()
                            if x.startswith(' VINA RESULT:')])
                    )

            else:
                sdf_to_pdbqt(sdf_file, ligand_pdbqt_file, i)

                center = get_center_from_pdbqt(receptor_pdbqt_file)
                cx, cy, cz = 0, 0, 0
                if center:
                    cx = center[0]
                    cy = center[1]
                    cz = center[2]
                # cx, cy, cz = mol.GetConformer().GetPositions().mean(0)

                # run QuickVina 2.1 -w
                out = os.popen(
                    f'/home/lxw/qvina2.1 --receptor {receptor_pdbqt_file} '
                    f'--ligand {ligand_pdbqt_file} '
                    f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
                    f'--size_x {size} --size_y {size} --size_z {size} '
                    f'--exhaustiveness {exhaustiveness}'
                ).read()

                try:
                    out_split = out.splitlines()
                    # print(out_split)
                    best_idx = out_split.index('-----+------------+----------+----------') + 1
                    best_line = out_split[best_idx].split()
                    assert best_line[0] == '1'
                    scores.append(float(best_line[1]))
                except:
                    scores.append(float(0))
                    pass

                
                if out_pdbqt_file.exists():
                    os.popen(f'/home/lxw/.conda/envs/MMF2Drug/bin/obabel {out_pdbqt_file} -O {out_sdf_file}').read()

            if return_rdmol:
                rdmol = Chem.SDMolSupplier(str(out_sdf_file))[0]
                rdmols.append(rdmol)

    if return_rdmol:
        return scores, rdmols
    else:
        return scores


if __name__ == '__main__':
    start_t = time.time()
    parser = argparse.ArgumentParser('QuickVina evaluation')
    parser.add_argument('--pdbqt_dir', type=Path,
                        help='Receptor files in pdbqt format')
    parser.add_argument('--sdf_dir', type=Path, default=None,
                        help='Ligand files in sdf format')
    parser.add_argument('--sdf_files', type=Path, nargs='+', default=None)
    parser.add_argument('--out_dir', type=Path)
    parser.add_argument('--write_csv', action='store_true')
    parser.add_argument('--write_dict', action='store_true')
    parser.add_argument('--dataset', type=str, default='crossdocked')
    args = parser.parse_args()
    assert (args.sdf_dir is not None) ^ (args.sdf_files is not None)

    results = {'receptor': [], 'ligand': [], 'scores': []}
    results_dict = {}
    
    sdf_files = list(args.sdf_dir.glob('*/rank1.sdf')) \
        if args.sdf_dir is not None else args.sdf_files
    pbar = tqdm(sdf_files)
    for sdf_file in pbar:
        pbar.set_description(f'Processing {sdf_file.name}')

        if args.dataset == 'moad':
            """
            Ligand file names should be of the following form:
            <receptor-name>_<pocket-id>_<some-suffix>.sdf
            where <receptor-name> and <pocket-id> cannot contain any 
            underscores, e.g.: 1abc-bio1_pocket0_gen.sdf
            """
            ligand_name = sdf_file.stem
            receptor_name, pocket_id, *suffix = ligand_name.split('_')
            suffix = '_'.join(suffix)
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')
        elif args.dataset == 'crossdocked':
            ligand_name = sdf_file.stem
            # receptor_name = ligand_name[:-4]
            start = ligand_name.find('_') + 1 # 找到第一个_的位置并加一
            end = ligand_name.find('_gen', start) # 从start位置开始找到第一个_gen的位置
            receptor_name = ligand_name[start:end]
            receptor_name = '8h6pcut6_pocket'
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')
            receptor_name = str(sdf_file).split('/')[-2]

        try:
            sdf_file = str(sdf_file)
            scores, rdmols = calculate_qvina2_score(
                receptor_file, sdf_file, args.out_dir, return_rdmol=True, index=receptor_name)
        except (ValueError, AttributeError) as e:
            print(e)
            continue
        results['receptor'].append(str(receptor_file))
        results['ligand'].append(str(sdf_file))
        results['scores'].append(scores)

        if args.write_dict:
            results_dict[receptor_name] = [scores, rdmols]

    if args.write_csv:
        df = pd.DataFrame.from_dict(results)
        df.to_csv(Path(args.out_dir, 'qvina2_scores.csv'))

    if args.write_dict:
        torch.save(results_dict, Path(args.out_dir, 'qvina2_scores.pt'))
    
    end_t = time.time()
    print('Time:',end_t-start_t)