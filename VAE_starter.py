
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import imageio
import matplotlib.image as mpimg
from scipy import ndimage
from sklearn.manifold import TSNE


def scatter_plot(latent_representations, labels):
    '''
    the scatter plot for visualizing the latent representations with the ground truth class label
    ----------
    latent_presentations: (N, dimension_latent_representation)
    labels: (N, )  the labels of the ground truth classes
    '''
    # borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    
    # Note that if the dimension_latent_representation > 2 you need to apply TSNE transformation
    # to map the latent representations from higher dimensionality to 2D
    # You can use #from sklearn.manifold import TSNE#
    
    def discrete_cmap(n, base_cmap=None):
        """Create an N-bin discrete colormap from the specified input map"""
        base = plt.cm.get_cmap(base_cmap)
        return base.from_list(base.name + str(n), base(np.linspace(0, 1, n)), n)

    if latent_representations.shape[1] > 2:
        latent_representations = TSNE(n_components=2).fit_transform(latent_representations)

    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], cmap=discrete_cmap(10, 'jet'), c=labels, edgecolors='black')
    plt.colorbar()
    plt.grid()
    plt.show()

def Plot_Kernel(_model):
    '''
    the plot for visualizing the learned weights of the autoencoder's encoder .
    ----------
    _model: Autoencoder
    '''
    # needs your implementation
    w = _model.encoder.nn[0].weight
    w = w.view(w.size(0), 28, 28)
    w = w.cpu().detach().numpy()
    # for layer_index, layer_weight in enumerate(y):
    for i in range(0, 6):
        plt.figure(figsize=(18, 6))
        for j in range(0, 5):
            plt.subplot(1, 5, j + 1)
            plt.imshow(w[i * 5 + j])
            plt.axis('off')
    plt.show()


def display_images_in_a_row(images,file_path='./tmp.png', display=True):
  '''
  images: (N,28,28): N images of 28*28 as a numpy array
  file_path: file path name for where to store the figure
  display: display the image or not
  '''
  save_image(images.view(-1, 1, 28, 28),
            '{}'.format(file_path))
  if display is True:
    plt.imshow(mpimg.imread('{}'.format(file_path)))


# Defining Model
class VAE_Trainer(object):
    '''
    The trainer for
    '''
    def __init__(self, autoencoder_model, learning_rate=1e-3, path_prefix = ""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_dataset(path_prefix)
        self.model = autoencoder_model
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)


    def init_dataset(self, path_prefix = ""):
        # load and preprocess dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainTransform  = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.FashionMNIST(root='{}./data'.format(path_prefix),  train=True,download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=4)
        valset = torchvision.datasets.FashionMNIST(root='{}./data'.format(path_prefix), train=False, download=True, transform=transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, num_workers=4)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.valset = valset
        self.trainset = trainset

    def loss_function(self, recon_x, x, mu, logvar):
        # Note that this function should be modified for the VAE part.
        # KLD term should be added to the final Loss.
        
        BCE = F.mse_loss(recon_x, x)
        KLD = 0.5 * torch.mean(logvar.exp() - logvar - 1 + mu.pow(2))
        # print("Rec shape:", recon_x.shape)
        # print("BCE:",BCE, "\nKLD", KLD)
        Loss = BCE +  KLD
        return Loss

    def get_train_set(self):
        images = torch.vstack([ x for x,_ in self.train_loader]) # get the entire train set
        return images
        
    def get_val_set(self):
        images = torch.vstack([ x for x,_ in self.val_loader]) # get the entire val set
        return images
    
    def train(self, epoch):
        # Note that you need to modify both trainer and loss_function for the VAE model
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in tqdm(enumerate(self.train_loader), total=len(self.train_loader) ) :
            data = data.to(self.device)
            self.optimizer.zero_grad()
            data = data.view(data.size(0), -1)
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        train_loss /= len(self.train_loader.dataset)/32 # 32 is the batch size
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss ))

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (data, _) in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                data = data.view(data.size(0), -1)
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                val_loss += self.loss_function(recon_batch, data, mu, logvar).item()

        val_loss /= len(self.val_loader.dataset)/32 # 32 is the batch size
        print('====> Val set loss (reconstruction error) : {:.4f}'.format(val_loss))
