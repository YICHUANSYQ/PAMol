import argparse
from encode import encode
import os
from runners.CreateModelRunner import CreateModelRunner
from runners.TrainModelRunner import TrainModelRunner
import time
import random
import numpy as np


class RunRunner:

    def __init__(self, smiles_file, pdb_file, storage_path, latent_file, proteins_file, proteins_test_file, encoder, 
                 n_epochs, batch_size, lr, b1, b2, lambda_gp, n_critic, save_interval, starting_epoch):
        # init params
        self.storage_path = storage_path
        self.smiles_file = smiles_file
        self.pdb_file = pdb_file
        self.output_latent = os.path.join(self.storage_path, latent_file)
        self.proteins_latent = os.path.join(self.storage_path, proteins_file)
        self.proteins_test_latent = os.path.join(self.storage_path, proteins_test_file)
        self.encoder = encoder
        self.decoder = encoder
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.save_interval = save_interval
        self.starting_epoch = starting_epoch

    def test(self):
        T = TrainModelRunner(input_data_path=self.output_latent, proteins_data_path=self.proteins_latent, 
                             proteins_test_data_path=self.proteins_test_latent, pdb_file=self.pdb_file, 
                             output_model_folder=self.storage_path, decoder=self.decoder, 
                             n_epochs=self.n_epochs, batch_size=self.batch_size, 
                             lr=self.lr, b1=self.b1, b2=self.b2, lambda_gp=self.lambda_gp, 
                             n_critic=self.n_critic, save_interval=self.save_interval, starting_epoch=self.starting_epoch)
        T.test()

    def run(self):
        print("Model MMF2Drug running, encoding training set")
        # encode(smiles_file=self.smiles_file, output_latent_file_path=self.output_latent, encoder=self.encoder)
        print("Encoding finished finished. Creating model files")
        C = CreateModelRunner(input_data_path=self.output_latent, output_model_folder=self.storage_path)
        C.run()
        print("Model Created. Training model")
        T = TrainModelRunner(input_data_path=self.output_latent, proteins_data_path=self.proteins_latent, 
                             proteins_test_data_path=self.proteins_test_latent, pdb_file=self.pdb_file, 
                             output_model_folder=self.storage_path, decoder=self.decoder, 
                             n_epochs=self.n_epochs, batch_size=self.batch_size, 
                             lr=self.lr, b1=self.b1, b2=self.b2, lambda_gp=self.lambda_gp, 
                             n_critic=self.n_critic, save_interval=self.save_interval, starting_epoch=self.starting_epoch)
        T.run()
        print("Model finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and train a model")

    parser.add_argument("--smiles_file", "-sf", help="The path to a data file", type=str, default="./data/crossdock/crossdock_data_process/final_filtered_train.csv")
    parser.add_argument("--pdb_file", "-pdbf", help="The path to a data file", type=str, default="./data/crossdock/crossdock_data_process/final_filtered_test.csv")
    parser.add_argument("--storage_path", "-st", help="The path to all outputs", type=str, default="./storage/cross_dock")
    parser.add_argument("--latent_file", "-lf", help="Name of latent vector file", type=str, default="encoded_smiles.latent")
    parser.add_argument("--proteins_file", "-pf", help="Name of proteins latent vector file", type=str, default="encoded_proteins_cross_fusion_train.latent")
    parser.add_argument("--proteins_test_file", "-ptf", help="Name of proteins latent vector file", type=str, default="encoded_proteins_cross_fusion_test.latent")
    parser.add_argument("--encoder", help="The data set the pre-trained heteroencoder has been trained on [chembl|moses] DEFAULT:chembl", type=str, default="chembl")
    
    parser.add_argument("--n_epochs", help="Number of epochs to run, default 2000", type=int, default=2000)
    parser.add_argument("--batch_size", help="Batch size for training, default 4096", type=int, default=64)
    parser.add_argument("--lr", help="Initial learning rate, default 0.0001", type=float, default=0.0002)
    parser.add_argument("--b1", help="default 0.5", type=float, default=0.5)
    parser.add_argument("--b2", help="default 0.999", type=float, default=0.999)
    parser.add_argument("--lambda_gp", help="Gradient penalty lambda_gp hyperparameter, default 10", type=float, default=10)

    parser.add_argument("--n_critic", help="Train the generator every n_critic steps, default 5", type=int, default=5)
    parser.add_argument("--save_interval", help="Save interval during model training, default 500", type=int, default=1000)
    parser.add_argument("--starting_epoch", help="Start training epoch, default 1", type=int, default=1)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    runner = RunRunner(**args)
    runner.run()
    # runner.test()