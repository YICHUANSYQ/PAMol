# PAMol

## Data

We trained/tested PAMol using the same data sets as [Pocket2Mol](https://github.com/pengxingang/Pocket2Mol) model.

1. Download the dataset archive `crossdocked_pocket10.tar.gz` and the split file `split_by_name.pt` from [this link](https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM).
2. Extract the TAR archive using the command: `tar -xzvf crossdocked_pocket10.tar.gz`.
3. Without altering the original division of the dataset, filter out data that cannot be constructed into a hypergraph. If the machine performance is good, you can try replacing the protein pocket hypergraph with a protein hypergraph.

## Training

1. Obtain the latent vector features of molecules via `encode.py`

2. Obtain protein sequence features via `proteins_seq_encode.py` and `prepare_data_pair.py`

3. Obtain protein pocket hypergraph features via `ProteinDataset.py`

4. Obtain fused protein latent vector features via `crossfusion.py`

5. Modify the paths of the aforementioned features in `run.py`

6. Run `run.py`

## Sampling

Execute `runner.test()` in `run.py`

<!-- # The README.md will be further improved after subsequent organization. -->
