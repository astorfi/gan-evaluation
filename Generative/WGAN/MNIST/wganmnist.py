import argparse
import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import os
import random
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from subprocess import call
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import make_grid
from torchvision.utils import save_image

"""
The code parameters

Notes
-----
Here, we define necessary parameters that we use in the code and store them in arg parser.
"""

# Parser for the parameters
parser = argparse.ArgumentParser()

# experimentName is the current file name without extension
experimentName = os.path.splitext(os.path.basename(__file__))[0]

# Required paths
parser.add_argument("--DATASETPATH", type=str,
                    default=os.path.expanduser('~/data'),
                    help="Dataset file")
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/model/' + experimentName),
                    help="Training status")

parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="l2 regularization")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument('--n_iter_D', type=int, default=5, help='number of D iters per each G iter')

# WGAN parameters
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

# Cuda
parser.add_argument("--cuda", type=bool, default=True,
                    help="CUDA activation")
parser.add_argument("--multiplegpu", type=bool, default=True,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs in case of multiple GPU")

# Model parameters
parser.add_argument("--latent_dim", type=int, default=32, help="dimensionality of the latent space")
parser.add_argument("--image_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=1,
                    help="Number of channels in the training images. For color images this is 3")
parser.add_argument("--nz", type=int, default=64, help="Size of z latent vector (i.e. size of generator input)")
parser.add_argument("--ngf", type=int, default=32, help="Size of feature maps in generator")
parser.add_argument("--ndf", type=int, default=32, help="Size of feature maps in discriminator")

# Training
parser.add_argument("--display_interval", type=int, default=10, help="interval between samples")
parser.add_argument("--sample_interval", type=int, default=2000, help="interval between generating fake samples")
parser.add_argument("--epoch_time_show", type=bool, default=True, help="interval betwen image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=5, help="number of epochs per model save")

# Phase
parser.add_argument("--training", type=bool, default=False, help="Training status")
parser.add_argument("--generate", type=bool, default=True, help="Generating Sythetic Data")
parser.add_argument("--resume", type=bool, default=False, help="Training status")
parser.add_argument("--finetuning", type=bool, default=False, help="Training status")
parser.add_argument("--evaluate", type=bool, default=False, help="Evaluation status")

opt = parser.parse_args()
print(opt)

"""
Initialization

Notes
-----
We check some precursor elements such as path and CUDA existence.
"""

# Create experiments DIR
if not os.path.exists(opt.expPATH):
    os.system('mkdir {0}'.format(opt.expPATH))

# Random seed for pytorch
opt.manualSeed = random.randint(1, 10000)  # fix seed
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

"""
Dataset preparation

Notes
-----
Dataset creation and preprocessing will be done here.
"""

# Create the dataset
# torchvision.transforms.Resize: Operation on PIL Image
# transforms.CenterCrop: Operation on PIL Image
# transforms.ToTensor: Convert a PIL Image or numpy.ndarray to tensor
# transforms.Normalize: Normalize a tensor image of size (C, H, W)
# For transforms.Normalize we use (0.5,), (0.5,) as we have only one channel for MNIST (new range: [-1,1])
# Read more: https://pytorch.org/docs/stable/torchvision/transforms.html
# ATTENTION: transforms.ToTensor() is required before transforms.Normalize as transforms.Normalize only operates on
# Tensors and not PIL images
datasetTrain = torchvision.datasets.MNIST(root=opt.DATASETPATH, train=True, transform=transforms.Compose([
    transforms.Resize(opt.image_size),
    transforms.CenterCrop(opt.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
]), target_transform=None, download=True)

# Create the dataloader
dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, batch_size=opt.batch_size,
                                              shuffle=True, num_workers=opt.n_cpu)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.num_gpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloaderTrain))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()

# # Uncomment to investigate data
# # Generate random samples for test
# random_samples = next(iter(dataloader_test))
# feature_size = random_samples.size()[1]

####################
### Architecture ###
####################

class Generator(nn.Module):

    """
    The generator of the GAN.

    ...

    Attributes
    ----------
    ngpu : int
        number of gpus
    main : function
        the sequential function consists of different layers

    Methods
    -------
    forward(input)
        The forward pass of the network
    """
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(opt.nz, opt.ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (ngf) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):

    """
    The discriminator of the GAN.

    ...

    Attributes
    ----------
    ngpu : int
        number of gpus
    conv1-conv5 : module
        the convolutional layers

    Methods
    -------
    forward(input)
        The forward pass of the network
    """

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.conv1 = nn.ModuleDict({
            'conv': nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            'activation': nn.LeakyReLU(0.2, inplace=True)
        })
        self.conv2 = nn.ModuleDict({
            'conv': nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            'bn': nn.BatchNorm2d(opt.ndf * 2),
            'activation': nn.LeakyReLU(0.2, inplace=True)
        })
        self.conv3 = nn.ModuleDict({
            'conv': nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            'bn': nn.BatchNorm2d(opt.ndf * 4),
            'activation': nn.LeakyReLU(0.2, inplace=True)
        })
        self.conv4 = nn.ModuleDict({
            'conv': nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            'bn': nn.BatchNorm2d(opt.ndf * 8),
            'activation': nn.LeakyReLU(0.2, inplace=True)
        })

        self.conv5 = nn.ModuleDict({
            'conv': nn.Conv2d(opt.ndf * 8, 1, 2, 1, 0, bias=False),
        })

    def forward(self, input):
        # Layer 1
        out = self.conv1['conv'](input)
        out = self.conv1['activation'](out)

        # Layer 2
        out = self.conv2['conv'](out)
        out = self.conv2['bn'](out)
        out = self.conv2['activation'](out)

        # Layer 3
        out = self.conv3['conv'](out)
        out = self.conv3['bn'](out)
        out = self.conv3['activation'](out)

        # Layer 4
        out = self.conv4['conv'](out)
        out = self.conv4['bn'](out)
        out = self.conv4['activation'](out)

        # Layer 5
        out = self.conv5['conv'](out)

        return out


#################
### Functions ###
#################

def discriminator_accuracy(predicted, y_true):
    """
    The discriminator accuracy on samples.

    Parameters:

    - `predicted`: The predicted labels.
    - `y_true`: The gorund truth labels.


    Return the values:

    - Accuracy: possibly modified from the parameter `context`;

    Not necessary for training.
    """

    total = y_true.size(0)
    correct = (torch.abs(predicted - y_true) <= 0.5).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy


def weights_init(m):
    """
    Weight initialization.

    Parameters:

    - `m`: The weights.

    Return the values by changing them in-place:

    - Weights: initialized weights`;
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


"""
Model creation

How To Do This
======================
(See the individual classes, methods, and attributes for details.)

1. Initialize generator and discriminator::

   Generator(opt.num_gpu).to(device)
   Discriminator(opt.num_gpu).to(device)

2. Create parallel processing with ``nn.DataParallel``.

3. Create optimizers::

    optimizer_G = torch.optim.Adam(generatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)

"""

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
optimizer_G = torch.optim.Adam(generatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)
optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)

"""
Training

1. Initialize generator and discriminator::

   Generator(opt.num_gpu).to(device)
   Discriminator(opt.num_gpu).to(device)

2. Create parallel processing with ``nn.DataParallel``.

3. Create optimizers::

    optimizer_G = torch.optim.Adam(generatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)

"""
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
        for i, data in enumerate(dataloaderTrain):
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

            # # train the discriminator n_iter_D times
            # if gen_iterations < 25 or gen_iterations % 500 == 0:
            #     n_iter_D = 100
            # else:
            #     n_iter_D = opt.n_iter_D
            j = 0
            while j < opt.n_iter_D:
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

            # read more at https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in
            # -pytorch/4903/4
            optimizer_G.step()
            gen_iterations += 1

            if iters % opt.display_interval == 0:
                print('TRAIN: [Epoch %d/%d] [Batch %d/%d] Loss_D: %.6f Loss_G: %.6f'
                      % (epoch + 1, opt.n_epochs, i, len(dataloaderTrain),
                         errD.item(), errG.item()), flush=True)

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % opt.sample_interval == 0) or (
                        (epoch == opt.n_epochs - 1) and (i == len(dataloaderTrain) - 1)):
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
            }, os.path.join(opt.expPATH, "model_generative_epoch_%d.pth" % (epoch + 1)))

            # keep only the most recent 10 saved models
            # ls -d -1tr /home/sina/experiments/pytorch/model/* | head -n -10 | xargs -d '\n' rm -f
            # call("ls -d -1tr " + opt.expPATH + "/*" + " | head -n -10 | xargs -d '\n' rm -f", shell=True)

if opt.generate:
    """
    The generation part.
    
    1. Load model
    2. Create fake samples
    3. Save images:
    
        - Output images' range should be in [0,1] 
        - Output images' size should be as [NWHC]
    
    """

    # Check cuda
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device BUT it is not in use...")

    # Activate CUDA
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    #####################################
    #### Load model and optimizer #######
    #####################################

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(opt.expPATH, "model_generative_epoch_20.pth"))

    # Load models
    generatorModel.load_state_dict(checkpoint['Generator_state_dict'])

    # insert weights [required]
    generatorModel.eval()

    #######################################################
    #### Load real data and generate synthetic data #######
    #######################################################

    #### Image Comparison ####
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloaderTrain))

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
    numpygrid = grid.cpu().detach().numpy()
    plt.imshow(np.transpose(numpygrid, (1, 2, 0)))
    plt.show()

    # Creare fake sample
    num_fake_samples = 10000
    #
    # Generate a batch of samples
    gen_samples = np.zeros((num_fake_samples, opt.image_size, opt.image_size, opt.nc), dtype=type(numpygrid))
    n_batches = int(num_fake_samples / opt.batch_size)
    for i in range(n_batches):
        # Sample noise as generator input
        z = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)
        fake_imgs = generatorModel(z)

        # Transform to numpy
        fake_imgs = fake_imgs.cpu().data.numpy()

        # back into the range of [0,1]
        fake_imgs = fake_imgs * 0.5 + 0.5

        # Channel order is different in Matplotlib and PyTorch
        # PyTorch : [NCWH]
        # Numpy: [NWHC]
        fake_imgs = np.transpose(fake_imgs, (0, 2, 3, 1))

        gen_samples[i * opt.batch_size:(i + 1) * opt.batch_size, :] = fake_imgs
        if (i + 1) % 10 == 0:
            print('processed {}-th batch'.format(i))
        # Check to see if there is any nan
        assert (gen_samples[i, :] != gen_samples[i, :]).any() == False

    gen_samples = np.delete(gen_samples, np.s_[(i + 1) * opt.batch_size:], 0)
    print('type(gen_samples)', type(gen_samples), gen_samples.shape)
    gen_samples = gen_samples.astype(np.float32)
    np.save(os.path.join(opt.expPATH, "fakeimages.npy"), gen_samples, allow_pickle=False)
    print('type(gen_samples)', type(gen_samples), gen_samples.shape)
