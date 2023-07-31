import os
import argparse
import multiprocessing as mp
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
from torchvision import transforms, datasets

from modules import Model
from util import tensor2im, save_image


parser = argparse.ArgumentParser(description='CVQ-VAE')
# General
parser.add_argument('--data_folder', type=str, help='name of the data folder')
parser.add_argument('--dataset', type=str, help='name of the dataset (mnist, fashion-mnist, cifar10)')
parser.add_argument('--batch_size', type=int, default=16, help='batch size (default: 16)')
# Latent space
parser.add_argument('--hidden_size', type=int, default=128, help='size of the latent vectors (default: 128)')
parser.add_argument('--num_residual_hidden', type=int, default=32, help='size of the redisual layers (default: 32)')
parser.add_argument('--num_residual_layers', type=int, default=2, help='number of residual layers (default: 2)')
# Quantiser parameters
parser.add_argument('--embedding_dim', type=int, default=64, help='dimention of codebook (default: 64)')
parser.add_argument('--num_embedding', type=int, default=512, help='number of codebook (default: 512)')
parser.add_argument('--distance', type=str, default='cos', help='distance for codevectors and features')
# Miscellaneous
parser.add_argument('--output_folder', type=str, default='/scratch/shared/beegfs/cxzheng/normcode/final_vqvae/', help='name of the output folder (default: vqvae)')
parser.add_argument('--model_name', type=str, default='fashionmnist_probrandom_contramin1/best.pt', help='name of the output folder (default: vqvae)')
parser.add_argument('--num_workers', type=int, default=mp.cpu_count() - 1, help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
parser.add_argument('--device', type=str, default='cuda', help='set the device (cpu or cuda, default: cpu)')
args = parser.parse_args()


# load dataset
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
    ])
if args.dataset == 'mnist':
    # Define the train & test datasets
    test_dataset = datasets.MNIST(args.data_folder, train=False, download=True, transform=transform)
    num_channels = 1
elif args.dataset == 'fashion-mnist':
    # Define the train & test datasets
    test_dataset = datasets.FashionMNIST(args.data_folder, train=False, download=True, transform=transform)
    num_channels = 1
elif args.dataset == 'cifar10':
    # Define the train & test datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(args.data_folder, train=False, download=True, transform=transform)
    num_channels = 3
elif args.dataset == 'celeba':
    # Define the train & test datasets
    transform = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CelebA(args.data_folder, split='valid', download=True, transform=transform)
    num_channels = 3

# Define the dataloaders
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Define the model
model = Model(num_channels, args.hidden_size, args.num_residual_layers, args.num_residual_hidden,
                  args.num_embedding, args.embedding_dim, distance=args.distance)

# load model
ckpt = torch.load(os.path.join(os.path.join(os.path.join(args.output_folder, 'models'), args.model_name)))
model.load_state_dict(ckpt)
model = model.to(args.device)
model.eval()

# store results
results_path = os.path.join(os.path.join(args.output_folder, 'results'), args.model_name)
original_path = os.path.join(results_path, 'original')
rec_path = os.path.join(results_path, 'rec')
if not os.path.exists(results_path):
    os.makedirs(original_path)
    os.makedirs(rec_path)

# test model
encodings = []
indexes = []
labels = []
all_images = []
imageid = 0
for images, label in test_loader:
    images = images.to(args.device)
    x_recons, loss, perplexity, encoding = model(images)
    # save indexes
    index = encoding.argmax(dim=1).view(images.size(0), -1)
    indexes.append(index)
    all_images.append(images.view(images.size(0), -1))
    # save labels
    labels.append(label)
    # save encodings
    encodings.append(encoding)
    # save image
    for x_recon, image in zip(x_recons, images):
        x_recon = tensor2im(x_recon)
        image = tensor2im(image)
        name = str(imageid).zfill(8) + '.jpg'
        save_image(image, os.path.join(original_path, name))
        save_image(x_recon, os.path.join(rec_path, name))
        imageid += 1

# calculate the perplexity in whole test images
encodings = torch.cat(encodings, dim=0)
# save the codebook count
count = torch.sum(encodings, dim=0).cpu().numpy()
usage = 1 - len(count[count==0])/len(count)
plt.figure(1)
plt.plot(count)
plt.ylabel("Count Number")
plt.xlabel("Vocabulary Index")
plt.savefig(os.path.join(results_path, 'validation.png'))
print("usage of the codebook vector: {}".format(usage))
# save the sorted codebook count
sort_count = -np.sort(-count)
plt.plot(sort_count)
plt.ylabel("Count Number")
plt.xlabel("Vocabulary Index")
plt.savefig(os.path.join(results_path,'csort_validation.png'))
avg_probs = torch.mean(encodings, dim=0)
perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
print("the perplexity of the codebook: {}".format(perplexity))

# visualize codebook
plt.figure(2)
code_book = model._vq_vae.embedding.weight.data.cpu()
tsne = TSNE(n_components=2, perplexity=5, n_iter=5000, verbose=True)
projections = tsne.fit_transform(code_book)
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False
plt.scatter(*projections[count==0].T,label=str(0),color=plt.cm.Dark2(0),alpha=0.425,zorder=2,)
plt.scatter(*projections[count>0].T,label=str(1),color=plt.cm.Dark2(1),alpha=0.425,zorder=2,)
plt.savefig(os.path.join(results_path,'code_distribution.png'), dpi=300)