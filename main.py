import os, random
import argparse
import multiprocessing as mp
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import  make_grid
from tensorboardX import SummaryWriter

from modules import Model


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(data_loader, model, optimizer, args, writer, data_variance=1):
    """trianing the model"""
    for images, _ in data_loader:
        images = images.to(args.device)
        optimizer.zero_grad()
        x, loss_vq, perplexity, _ = model(images)

        # loss function
        loss_recons = F.mse_loss(x, images) / data_variance
        loss = loss_recons + loss_vq
        loss.backward()

        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)
        writer.add_scalar('loss/train/perplexity', perplexity.item(), args.steps)

        optimizer.step()

        args.steps +=1


def test(data_loader, model, args, writer):
    """evaluation model"""
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _ in data_loader:
            images = images.to(args.device)
            x, loss, _, _ = model(images)
            loss_recons += F.mse_loss(x, images)
            loss_vq += loss
        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)
    
    # Logs
    writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
    writer.add_scalar('loss/test/quantization', loss_vq.item(), args.steps)

    return loss_recons.item(), loss_vq.item()


def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x, _, _, _ = model(images)
    return x


def main(args):
    writer = SummaryWriter(os.path.join(os.path.join(args.output_folder, 'logs'), args.exp_name))
    save_filename = os.path.join(os.path.join(args.output_folder, 'models'), args.exp_name)
    seed_everything(args.seed)

    # load dataset
    data_variance=1
    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10', 'celeba']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        if args.dataset == 'mnist':
            # Define the train & test datasets
            train_dataset = datasets.MNIST(args.data_folder, train=True,
                download=True, transform=transform)
            test_dataset = datasets.MNIST(args.data_folder, train=False,
                download=True, transform=transform)
            data_variance=np.var(train_dataset.data.numpy() / 255.0)
            num_channels = 1
        elif args.dataset == 'fashion-mnist':
            # Define the train & test datasets
            train_dataset = datasets.FashionMNIST(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(args.data_folder,
                train=False, download=True, transform=transform)
            data_variance=np.var(train_dataset.data.numpy() / 255.0)
            num_channels = 1
        elif args.dataset == 'cifar10':
            # Define the train & test datasets
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
            train_dataset = datasets.CIFAR10(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(args.data_folder,
                train=False, download=True, transform=transform)
            data_variance=np.var(train_dataset.data / 255.0)
            num_channels = 3
        elif args.dataset == 'celeba':
            # Define the train & test datasets
            transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
            train_dataset = datasets.CelebA(args.data_folder,
                split='train', download=True, transform=transform)
            test_dataset = datasets.CelebA(args.data_folder,
                split='valid', download=True, transform=transform)
            train_list = []
            for i in range(len(train_dataset)):
                train_list.append(train_dataset[i][0])
            num_channels = 3
        valid_dataset = test_dataset

    # Define the dataloaders
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, 
        worker_init_fn=seed_worker, generator=g)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=32, shuffle=False,
        worker_init_fn=seed_worker, generator=g)
    
    # Define the model
    model = Model(num_channels, args.hidden_size, args.num_residual_layers, args.num_residual_hidden,
                  args.num_embedding, args.embedding_dim, args.commitment_cost, args.distance,
                  args.anchor, args.first_batch, args.contras_loss).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Update the model
    best_loss = -1.
    for epoch in range(args.num_epochs):
        # training and testing the model
        train(train_loader, model, optimizer, args, writer, data_variance)
        loss_rec, loss_vq = test(valid_loader, model, args, writer)

        # visualization
        images, _ = next(iter(test_loader))
        rec_images = generate_samples(images, model, args)
        input_grid = make_grid(images, nrow=8, range=(-1, 1), normalize=True)
        rec_grid = make_grid(rec_images, nrow=8, range=(-1,1), normalize=True)
        writer.add_image('original', input_grid, epoch + 1)
        writer.add_image('reconstruction', rec_grid, epoch + 1)

        # save model
        if (epoch == 0) or (loss_rec < best_loss):
            best_loss = loss_rec
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(model.state_dict(), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CVQ-VAE')
    # General
    parser.add_argument('--data_folder', type=str, help='name of the data folder')
    parser.add_argument('--dataset', type=str, help='name of the dataset (mnist, fashion-mnist, cifar10)')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size (default: 1024)')
    # Latent space
    parser.add_argument('--hidden_size', type=int, default=128, help='size of the latent vectors (default: 128)')
    parser.add_argument('--num_residual_hidden', type=int, default=32, help='size of the redisual layers (default: 32)')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='number of residual layers (default: 2)')
    # Quantiser parameters
    parser.add_argument('--embedding_dim', type=int, default=64, help='dimention of codebook (default: 64)')
    parser.add_argument('--num_embedding', type=int, default=512, help='number of codebook (default: 512)')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='hyperparameter for the commitment loss')
    parser.add_argument('--distance', type=str, default='cos', help='distance for codevectors and features')
    parser.add_argument('--anchor', type=str, default='closest', help='anchor sampling methods (random, closest, probrandom)')
    parser.add_argument('--first_batch', action='store_true', help='offline version with only one time reinitialisation')
    parser.add_argument('--contras_loss', action='store_true', help='using contrastive loss')
    # Optimization
    parser.add_argument('--seed', type=int, default=42, help="seed for everything")
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate for Adam optimizer (default: 2e-4)')
    # Miscellaneous
    parser.add_argument('--output_folder', type=str, default='./', help='name of the output folder (default: vqvae)')
    parser.add_argument('--exp_name', type=str, default='vqvae', help='name of the output folder (default: vqvae)')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count() - 1, help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu', help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.output_folder, 'logs')):
        os.makedirs(os.path.join(args.output_folder, 'logs'))
    if not os.path.exists(os.path.join(args.output_folder, 'models')):
        os.makedirs(os.path.join(args.output_folder, 'models'))
    # Device
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.exp_name += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists(os.path.join(os.path.join(args.output_folder, 'models'), args.exp_name)):
        os.makedirs(os.path.join(os.path.join(args.output_folder, 'models'), args.exp_name))
    args.steps = 0

    main(args)