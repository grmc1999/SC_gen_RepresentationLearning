
import os
import requests
import muon as mu

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset

from src.models import SimpleAE, SimCLRGenomics
from src.utils import augment_genomics, nt_xent_loss

def download_data(url, filename):
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        else:
            print(f"File {filename} already exists.")
    
    

if __name__ == '__main__':
    url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_10k/pbmc_unsorted_10k_filtered_feature_bc_matrix.h5"
    filename = "pbmc_10k_multiome.h5"

    download_data(url, filename)

    mdata = mu.read_10x_h5(filename)
    mdata.var_names_make_unique()
    rna = mdata.mod['rna']

    X = rna.X.toarray() # Convert sparse matrix to dense for the NN

    X_tensor = torch.FloatTensor(X)
    dataset = DataLoader(TensorDataset(X_tensor), batch_size=64, shuffle=True)

    # Simple Autoencoder
    model = SimpleAE(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(10)):
        for batch in dataset:
            inputs = batch[0]
            recon, latent = model(inputs)
            loss = criterion(recon, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


    x_data = torch.FloatTensor(rna.X.toarray())
    dataset = DataLoader(TensorDataset(x_data), batch_size=128, shuffle=True)

    # SimCLR
    model = SimCLRGenomics(input_dim=36601)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        for (x_batch,) in tqdm  (dataset):
            x_batch = x_batch

            # Create two augmented versions of the same batch
            view1 = augment_genomics(x_batch)
            view2 = augment_genomics(x_batch)

            # Pass through the model
            _, z1 = model(view1)
            _, z2 = model(view2)

            # Calculate loss and update
            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")