import pickle

from datasets.TargetMolsDataset import TargetMolsDataset
from datasets.LatentMolsDataset import LatentMolsDataset
from models.Discriminator import Discriminator, C_Discriminator
from models.Generator import Generator, C_Generator
from src.Sampler import Sampler, C_Sampler
from decode import decode_crossdock
import os
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import json
import time
import sys
from tqdm import tqdm
import pandas as pd
import logging
import random
import csv

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(
    filename='function_execution.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        logging.info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class TrainModelRunner:

    def __init__(self, input_data_path, proteins_data_path, proteins_test_data_path, pdb_file, 
                 output_model_folder, decoder,
                 n_epochs, batch_size, lr, b1, b2, lambda_gp, n_critic, save_interval, 
                 starting_epoch, message=""):
        self.message = message

        # init params
        self.input_data_path = input_data_path
        self.proteins_data_path = proteins_data_path
        self.proteins_test_data_path = proteins_test_data_path
        self.pdb_file =  pdb_file
        self.output_model_folder = output_model_folder
        self.decoder = decoder
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.lambda_gp = lambda_gp
        
        self.n_critic = n_critic
        self.save_interval = save_interval
        self.starting_epoch = starting_epoch

        # initialize molecular dataloader
        json_smiles = open(self.input_data_path, "r")
        latent_space_mols = np.array(json.load(json_smiles))
        latent_space_mols = latent_space_mols.reshape(latent_space_mols.shape[0], 512)
        
        # initialize protein dataloader
        json_proteins = open(self.proteins_data_path, "r")
        proteins_feature = np.array(json.load(json_proteins))
        proteins_feature = proteins_feature.reshape(proteins_feature.shape[0], 768)
        
        self.dataloader = TargetMolsDataset(latent_space_mols=latent_space_mols, proteins_feature=proteins_feature, batch_size=self.batch_size)
        
        # load discriminator
        discriminator_name = 'discriminator.pth' if self.starting_epoch == 1 else str(self.starting_epoch - 1) + '_discriminator.pth'
        discriminator_path = os.path.join(output_model_folder, discriminator_name)
        self.D = Discriminator.load(discriminator_path)
        # load condition discriminator
        discriminator_name = 'c_discriminator.pth' if self.starting_epoch == 1 else str(self.starting_epoch - 1) + 'c_discriminator.pth'
        discriminator_path = os.path.join(output_model_folder, discriminator_name)
        self.c_D = C_Discriminator.load(discriminator_path)

        # load generator
        generator_name = 'generator.pth' if self.starting_epoch == 1 else str(self.starting_epoch - 1) + '_generator.pth'
        generator_path = os.path.join(output_model_folder, generator_name)
        self.G = C_Generator.load(generator_path)

        # initialize condition sampler
        self.Sampler = C_Sampler(self.G)

        # initialize optimizer
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_c_D = torch.optim.Adam(self.c_D.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.G.cuda()
            self.D.cuda()
            self.c_D.cuda()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def run(self):
        print("Training of GAN started.")
        print("Message: %s" % self.message)
        sys.stdout.flush()

        disc_loss_log = []
        g_loss_log = []
        for epoch in range(self.starting_epoch, self.n_epochs + 1):
            for i, data in enumerate(tqdm(self.dataloader.loader, desc=f'Epoch {epoch}/{self.n_epochs}', unit='Batch')):
                real_mols, proteins = data['mols'], data['proteins']
                proteins = proteins.cuda()
                # Configure input
                real_mols = real_mols.type(self.Tensor)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()

                # Generate a batch of mols from noise
                fake_mols = self.Sampler.sample(real_mols.shape[0], proteins)

                # Real mols
                real_validity = (self.D(real_mols) + self.c_D(real_mols, proteins)) / 2
                # Fake mols
                fake_validity = (self.D(fake_mols) + self.c_D(fake_mols, proteins)) / 2

                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(real_mols.data, fake_mols.data, proteins)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
                
                d_loss.backward()
                self.optimizer_D.step()
                self.optimizer_c_D.zero_grad()
                self.optimizer_G.zero_grad()
                
                
                # Train the generator every n_critic steps
                if i % self.n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------
                    
                    # Generate a batch of mols
                    fake_mols = self.Sampler.sample(real_mols.shape[0], proteins)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = (self.D(fake_mols) + self.c_D(fake_mols, proteins)) / 2
                    
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    self.optimizer_G.step()

                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, self.n_epochs, i + 1, len(self.dataloader.loader), d_loss.item(), g_loss.item()))
                    sys.stdout.flush()
                    
                    disc_loss_log.append([time.time(), epoch, i + 1, d_loss.item()])
                    g_loss_log.append([time.time(), epoch, i + 1, g_loss.item()])

            if epoch % self.save_interval == 0:
                generator_save_path = os.path.join(self.output_model_folder, str(epoch) + '_generator.pth')
                discriminator_save_path = os.path.join(self.output_model_folder, str(epoch) + '_discriminator.pth')
                c_discriminator_save_path = os.path.join(self.output_model_folder, str(epoch) + 'c_discriminator.pth')
                self.G.save(generator_save_path)
                self.D.save(discriminator_save_path)
                self.c_D.save(c_discriminator_save_path)

        # log the losses
        with open(os.path.join(self.output_model_folder, 'disc_loss.json'), 'w') as json_file:
            json.dump(disc_loss_log, json_file)
        with open(os.path.join(self.output_model_folder, 'gen_loss.json'), 'w') as json_file:
            json.dump(g_loss_log, json_file)

    def compute_gradient_penalty(self, real_samples, fake_samples, condition):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1)))

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = (self.D(interpolates) + self.c_D(interpolates, condition)) / 2
        fake = self.Tensor(real_samples.shape[0], 1).fill_(1.0)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
    
    @timer
    def test(self):
        n_pt = self.n_epochs
        generator_path = os.path.join(self.output_model_folder, "{}_generator.pth".format(n_pt))
        self.G = C_Generator.load(generator_path)
        self.G.cuda()
        self.Sampler = C_Sampler(self.G)
        self.G.eval()
        self.condition_sample()

    def condition_sample(self):
        json_proteins = open(self.proteins_test_data_path, "r")
        proteins_feature = np.array(json.load(json_proteins))
        proteins_feature = proteins_feature.reshape(proteins_feature.shape[0], 768).tolist()

        if not isinstance(proteins_feature, list):
            raise ValueError("The loaded data is not a list")

        pdb_paths = pd.read_csv(self.pdb_file)
        pdb_paths = pdb_paths['pdb_path'].str.rstrip('.pdb')

        for index, (feature, gen_file_name) in enumerate(zip(proteins_feature, pdb_paths)):
            feature_array = np.array(feature)

            if feature_array.ndim == 1:
                feature_array = feature_array[np.newaxis, :]
            
            feature_tensor = torch.tensor(feature_array, dtype=torch.float32)

            if torch.cuda.is_available():
                feature_tensor = feature_tensor.cuda()
            
            valid_smiles = []

            while(len(valid_smiles) < 100):
                proteins_batch = feature_tensor.repeat(25600, 1)
                proteins_batch = proteins_batch.cuda()

                gen_mols = self.Sampler.sample(proteins_batch.shape[0], proteins_batch)
                gen_mols = gen_mols.detach().cpu().numpy().tolist()

                res = decode_crossdock(gen_mols, model=self.decoder)
                valid_smiles.extend(res)
            
            final_smiles_path = os.path.join(self.output_model_folder, 'proteins', '{}.latent'.format(gen_file_name))
            os.makedirs(os.path.dirname(final_smiles_path), exist_ok=True)

            with open(final_smiles_path, 'w') as smiles_file:
                json.dump(valid_smiles, smiles_file)
    