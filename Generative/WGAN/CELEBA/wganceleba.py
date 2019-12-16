import argparse
import os
import numpy as np
import math
import time
import random

import matplotlib.pyplot as plt

import os
from subprocess import call

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.utils as vutils
import torchvision.datasets as dset

# import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()

# experimentName is the current file name without extension
experimentName = os.path.splitext(os.path.basename(__file__))[0]

parser.add_argument("--DATASETPATH", type=str,
                    default=os.path.expanduser('~/data/celeba'),
                    help="Dataset file")

parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="l2 regularization")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument('--n_iter_D', type=int, default=5, help='number of D iters per each G iter')

# Check the details
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

parser.add_argument("--cuda", type=bool, default=True,
                    help="CUDA activation")
parser.add_argument("--multiplegpu", type=bool, default=True,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs in case of multiple GPU")

parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")
parser.add_argument("--image_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images. For color images this is 3")
parser.add_argument("--nz", type=int, default=64, help="Size of z latent vector (i.e. size of generator input)")
parser.add_argument("--ngf", type=int, default=3, help="Size of feature maps in generator")
parser.add_argument("--ndf", type=int, default=3, help="Size of feature maps in discriminator")

parser.add_argument("--display_interval", type=int, default=10, help="interval between samples")
parser.add_argument("--sample_interval", type=int, default=2000, help="interval between generating fake samples")
parser.add_argument("--epoch_time_show", type=bool, default=True, help="interval betwen image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=5, help="number of epops per model save")

parser.add_argument("--training", type=bool, default=True, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Training status")
parser.add_argument("--finetuning", type=bool, default=False, help="Training status")
parser.add_argument("--generate", type=bool, default=False, help="Generating Sythetic Data")
parser.add_argument("--evaluate", type=bool, default=False, help="Evaluation status")
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/model/'+experimentName),
                    help="Training status")
opt = parser.parse_args()
print(opt)

# Create experiments DIR
if not os.path.exists(opt.expPATH):
    os.system('mkdir {0}'.format(opt.expPATH))

# Random seed for pytorch
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
cudnn.benchmark = True

# Check cuda
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device BUT it is not in use...")

# Activate CUDA
device = torch.device("cuda:0" if opt.cuda else "cpu")


####### Params ##########

# # Number of workers for dataloader
# workers = 2
#
# # Batch size during training
# batch_size = 128
#
# # Spatial size of training images. All images will be resized to this
# #   size using a transformer.
# image_size = 64
#
# # Number of channels in the training images. For color images this is 3
# nc = 3
#
# # Size of z latent vector (i.e. size of generator input)
# nz = 100
#
# # Size of feature maps in generator
# ngf = 64
#
# # Size of feature maps in discriminator
# ndf = 64
#
# # Number of training epochs
# num_epochs = 5
#
# # Learning rate for optimizers
# # lr = 0.0002
#
# # Beta1 hyperparam for Adam optimizers
# # beta1 = 0.5
#
# # Number of GPUs available. Use 0 for CPU mode.
# ngpu = 1

##########################
### Dataset Processing ###
##########################

# data = np.load(os.path.expanduser(opt.DATASETPATH), allow_pickle=True)

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=opt.DATASETPATH,
                           transform=transforms.Compose([
                               transforms.Resize(opt.image_size),
                               transforms.CenterCrop(opt.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=opt.n_cpu)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.num_gpu > 0) else "cpu")


# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

# # Generate random samples for test
# random_samples = next(iter(dataloader_test))
# feature_size = random_samples.size()[1]

####################
### Architecture ###
####################

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(opt.nz, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)



#################
### Functions ###
#################

def discriminator_accuracy(predicted, y_true):
    """
    The discriminator accuracy on samples
    :param predicted: The predicted labels
    :param y_true: The gorund truth labels
    :return: Accuracy
    """
    total = y_true.size(0)
    correct = (torch.abs(predicted - y_true) <= 0.5).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


#############
### Model ###
#############

# Initialize generator and discriminator
generatorModel = Generator(opt.num_gpu).to(device)
discriminatorModel = Discriminator(opt.num_gpu).to(device)

# Define cuda Tensors
Tensor = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1


if torch.cuda.device_count() > 1 and opt.multiplegpu:

  gpu_idx = list(range(opt.num_gpu))
  generatorModel = nn.DataParallel(generatorModel, device_ids=[gpu_idx[-1]])
  discriminatorModel = nn.DataParallel(discriminatorModel, device_ids=[gpu_idx[-1]])


if torch.cuda.is_available():
    """
    model.cuda() will change the model inplace while input.cuda() 
    will not change input inplace and you need to do input = input.cuda()
    ref: https://discuss.pytorch.org/t/when-the-parameters-are-set-on-cuda-the-backpropagation-doesnt-work/35318
    """
    # generatorModel.cuda()
    # discriminatorModel.cuda()
    one, mone = one.cuda(), mone.cuda()
    Tensor = torch.cuda.FloatTensor

# Weight initialization
generatorModel.apply(weights_init)
discriminatorModel.apply(weights_init)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Optimizers
optimizer_G = torch.optim.Adam(generatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)

################
### TRAINING ###
################
if opt.training:

    epoch_start = 0
    if opt.resume:
        #####################################
        #### Load model and optimizer #######
        #####################################

        # Loading the checkpoint
        checkpoint = torch.load(os.path.join(opt.expPATH, "model_epoch_90.pth"))

        # Load models
        generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
        discriminatorModel.load_state_dict(checkpoint['Discriminator_state_dict'])

        # Load optimizers
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

        # Load epoch number
        epoch_start = checkpoint['epoch']

        generatorModel.eval()
        discriminatorModel.eval()

    gen_iterations = 0
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    for epoch in range(epoch_start, opt.n_epochs):
        epoch_start = time.time()
        for i, data in enumerate(dataloader):
            iters += 1

            samples = data[0]
            labels = data[1]

            # Adversarial ground truths
            valid = Variable(Tensor(samples.shape[0]).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(samples.shape[0]).fill_(0.0), requires_grad=False)

            # Configure input
            real_samples = Variable(samples.type(Tensor))

            # Sample noise as generator input
            z = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for p in discriminatorModel.parameters():  # reset requires_grad
                p.requires_grad = True

            # train the discriminator n_iter_D times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                n_iter_D = 100
            else:
                n_iter_D = opt.n_iter_D
            j = 0
            while j < n_iter_D:
                j += 1

                # clamp parameters to a cube
                for p in discriminatorModel.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                # reset gradients of discriminator
                optimizer_D.zero_grad()

                out_real = discriminatorModel(real_samples)
                # accuracy_real = discriminator_accuracy(torch.sigmoid(out_real), valid)
                errD_real = torch.mean(out_real.view(-1)).view(1)

                # Measure discriminator's ability to classify real from generated samples
                # The detach() method constructs a new view on a tensor which is declared
                # not to need gradients, i.e., it is to be excluded from further tracking of
                # operations, and therefore the subgraph involving this view is not recorded.
                # Refer to http://www.bnikolic.co.uk/blog/pytorch-detach.html.

                # Generate a batch of images
                fake_samples = generatorModel(z)

                out_fake = discriminatorModel(fake_samples.detach()).view(-1)
                # accuracy_fake = discriminator_accuracy(torch.sigmoid(out_fake), fake)
                errD_fake = torch.mean(out_fake).view(1)
                errD = -(errD_real - errD_fake)
                errD.backward()

                # Optimizer step
                optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            # We’re supposed to clear the gradients each iteration before calling loss.backward() and optimizer.step().
            #
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch
            # accumulates the gradients on subsequent backward passes. This is convenient while training RNNs. So,
            # the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.
            #
            # Because of this, when you start your training loop, ideally you should zero out the gradients so
            # that you do the parameter update correctly. Else the gradient would point in some other direction
            # than the intended direction towards the minimum (or maximum, in case of maximization objectives).

            # Since the backward() function accumulates gradients, and you don’t want to mix up gradients between
            # minibatches, you have to zero them out at the start of a new minibatch. This is exactly like how a general
            # (additive) accumulator variable is initialized to 0 in code.

            for p in discriminatorModel.parameters():  # reset requires_grad
                p.requires_grad = False

            # Zero grads
            optimizer_G.zero_grad()

            # Loss measures generator's ability to fool the discriminator
            errG = -torch.mean(discriminatorModel(fake_samples).view(-1)).view(1)
            errG.backward()

            # read more at https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/4
            optimizer_G.step()
            gen_iterations += 1

            if iters % opt.display_interval == 0:
                    print('TRAIN: [Epoch %d/%d] [Batch %d/%d] Loss_D: %.6f Loss_G: %.6f'
                          % (epoch + 1, opt.n_epochs, i, len(dataloader),
                             errD.item(), errG.item()), flush=True)

                        # Save Losses for plotting later
                    G_losses.append(errG.item())
                    D_losses.append(errD.item())

                    # Check how the generator is doing by saving G's output on fixed_noise
                    if (iters % opt.sample_interval == 0) or ((epoch == opt.n_epochs - 1) and (i == len(dataloader) - 1)):
                        with torch.no_grad():
                            fake = generatorModel(fixed_noise).detach().cpu()
                            gridimg = np.transpose(
                                    vutils.make_grid(fake.to(device)[:64], padding=5, normalize=True).cpu(),
                                    (1, 2, 0)).data.numpy()
                            plt.imsave(os.path.join(opt.expPATH, "img_%d.png" % (iters)), gridimg)
                        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))


        # End of epoch
        epoch_end = time.time()
        if opt.epoch_time_show:
            print("It has been {0} seconds for this epoch".format(epoch_end - epoch_start), flush=True)

        if (epoch + 1) % opt.epoch_save_model_freq == 0:
            # Refer to https://pytorch.org/tutorials/beginner/saving_loading_models.html
            torch.save({
                'epoch': epoch + 1,
                'Generator_state_dict': generatorModel.state_dict(),
                'Discriminator_state_dict': discriminatorModel.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, os.path.join(opt.expPATH, "model_epoch_%d.pth" % (epoch + 1)))

            # keep only the most recent 10 saved models
            # ls -d -1tr /home/sina/experiments/pytorch/model/* | head -n -10 | xargs -d '\n' rm -f
            # call("ls -d -1tr " + opt.expPATH + "/*" + " | head -n -10 | xargs -d '\n' rm -f", shell=True)

    # np.save('G_losses', np.array(G_losses), allow_pickle=True)
    # np.save('D_losses', np.array(D_losses), allow_pickle=True)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Figures
    fig = plt.figure(figsize=(8, 8))
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    plt.show()


    #### Image Comparison ####
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()

if opt.finetuning:

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(opt.PATH, "model_epoch_100.pth"))

    # Setup model
    generatorModel = Generator()
    discriminatorModel = Discriminator()

    if opt.cuda:
        generatorModel.cuda()
        discriminatorModel.cuda()

    # Setup optimizers
    optimizer_G = torch.optim.Adam(generatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Load models
    generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
    discriminatorModel.load_state_dict(checkpoint['Discriminator_state_dict'])

    # Load optimizers
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    # Load losses
    g_loss = checkpoint['g_loss']
    d_loss = checkpoint['d_loss']

    # Load epoch number
    epoch = checkpoint['epoch']

    generatorModel.eval()
    discriminatorModel.eval()

if opt.generate:

    # Check cuda
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device BUT it is not in use...")

    # Activate CUDA
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    #####################################
    #### Load model and optimizer #######
    #####################################

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(opt.expPATH, "model_epoch_50.pth"))

    # Load models
    generatorModel.load_state_dict(checkpoint['Generator_state_dict'])

    # insert weights [required]
    generatorModel.eval()

    #######################################################
    #### Load real data and generate synthetic data #######
    #######################################################

    #### Image Comparison ####
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))


    # Generate a batch of fake images
    z = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)
    fake = generatorModel(z)
    grid = vutils.make_grid(fake, padding=2, normalize=True)

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()

    # # Load real data
    # real_samples = dataset.return_data()
    # num_fake_samples = 10000
    #
    # # Generate a batch of samples
    # gen_samples = np.zeros_like(real_samples, dtype=type(real_samples))
    # n_batches = int(num_fake_samples / opt.batch_size)
    # for i in range(n_batches):
    #     # Sample noise as generator input
    #     z = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)
    #     fake_imgs = generatorModel(z)
    #     gen_samples[i * opt.batch_size:(i + 1) * opt.batch_size, :] = fake_imgs.cpu().data.numpy()
    #     # Check to see if there is any nan
    #     assert (gen_samples[i, :] != gen_samples[i, :]).any() == False
    #
    # gen_samples = np.delete(gen_samples, np.s_[(i + 1) * opt.batch_size:], 0)
    #
    # # Trasnform Object array to float
    # gen_samples = gen_samples.astype(np.float32)


if opt.evaluate:
    # Load synthetic data
    gen_samples = np.load(os.path.join(opt.expPATH, "synthetic.npy"), allow_pickle=False)

    # Load real data
    real_samples = dataset_train_object.return_data()[0:gen_samples.shape[0], :]

    # Dimenstion wise probability
    prob_real = np.mean(real_samples, axis=0)
    prob_syn = np.mean(gen_samples, axis=0)

    p1 = plt.scatter(prob_real, prob_syn, c="b", alpha=0.5, label="WGAN")
    x_max = max(np.max(prob_real), np.max(prob_syn))
    x = np.linspace(0, x_max + 0.1, 1000)
    p2 = plt.plot(x, x, linestyle='-', color='k', label="Ideal")  # solid
    plt.tick_params(labelsize=12)
    plt.legend(loc=2, prop={'size': 15})
    # plt.title('Scatter plot p')
    # plt.xlabel('x')
    # plt.ylabel('y')
    plt.show()