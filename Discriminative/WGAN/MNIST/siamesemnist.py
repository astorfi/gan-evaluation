import argparse
import os
import numpy as np
import math
import time
import random
import os
from subprocess import call
import matplotlib.pyplot as plt
from sklearn import metrics
import json

import PIL.ImageOps

# Pytorch library
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch

# experimentName is the current file name without extension
experimentName = os.path.splitext(os.path.basename(__file__))[0]

parser = argparse.ArgumentParser()
parser.add_argument("--DATASETPATH", type=str,
                    default=os.path.expanduser('~/data'),
                    help="Dataset file")

parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--num_pairs", type=int, default=100000, help="number of pairs")
parser.add_argument("--image_size", type=int, default=32, help="img size")

parser.add_argument("--training", type=bool, default=False, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Resume training or not")
parser.add_argument("--generate", type=bool, default=False, help="Generating features")
parser.add_argument("--evaluate", type=bool, default=True, help="Evaluation")
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/model/' + experimentName),
                    help="Training status")

parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="l2 regularization")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

parser.add_argument("--nc", type=int, default=1,
                    help="Number of channels in the training images. For color images this is 3")
parser.add_argument("--nz", type=int, default=64, help="Size of z latent vector (i.e. size of generator input)")
parser.add_argument("--ndf", type=int, default=64, help="Size of feature maps in discriminator")

parser.add_argument("--sample_interval", type=int, default=10, help="interval between samples")
parser.add_argument("--epoch_time_show", type=bool, default=False, help="interval betwen image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=1, help="number of epochs per model save")
parser.add_argument("--display_sample", type=bool, default=True, help="Display sample images")

parser.add_argument("--cuda", type=bool, default=True,
                    help="CUDA activation")
parser.add_argument("--multiplegpu", type=bool, default=True,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs in case of multiple GPU")

opt = parser.parse_args()
print(opt)

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

##########################
### Dataset Processing ###
##########################


if opt.training:

    MNISTTrain = torchvision.datasets.MNIST(root=opt.DATASETPATH, train=True, transform=None, target_transform=None,
                                            download=True)
    MNISTTest = torchvision.datasets.MNIST(root=opt.DATASETPATH, train=False, transform=None, target_transform=None,
                                           download=True)

    print('Train data shape:', MNISTTrain.data.shape)
    print('Train labels shape:', MNISTTrain.targets.shape)

    print('Test data shape:', MNISTTest.data.shape)
    print('Test labels shape:', MNISTTest.targets.shape)


    class SiameseDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, invert_status=True, transform=None):
            self.data = dataset.data
            self.targets = dataset.targets
            self.dataSize = self.data.shape[0]
            self.transform = transform
            self.invert_status = invert_status
            self.transfortoPIL = torchvision.transforms.ToPILImage()

        def __getitem__(self, index):

            rand_idx = np.random.randint(self.dataSize, size=1)
            # self.data.shape = (N,W,H), self.data[rand_idx].shape = (1,W,H)
            img0_tuple = (self.data[rand_idx], self.targets[rand_idx])
            same_class_status = random.randint(0, 1)
            if same_class_status:
                while True:
                    rand_idx = np.random.randint(self.dataSize, size=1)
                    # keep looping till the same class image is found
                    img1_tuple = (self.data[rand_idx], self.targets[rand_idx])
                    if img0_tuple[1] == img1_tuple[1]:
                        break
            else:
                while True:
                    rand_idx = np.random.randint(self.dataSize, size=1)
                    # keep looping till a different class image is found
                    img1_tuple = (self.data[rand_idx], self.targets[rand_idx])
                    if img0_tuple[1] != img1_tuple[1]:
                        break

            # samples (of type tensors)
            img0 = img0_tuple[0]
            img1 = img1_tuple[0]

            if self.transform is not None:
                img0 = self.transform(img0)
                img1 = self.transform(img1)

            return img0, img1, torch.from_numpy(np.array([int(img0_tuple[1] != img1_tuple[1])], dtype=np.float32))

        def __len__(self):
            return self.dataSize


    # The order of transfor matters
    # transforms.ToTensor() transform numpy to tensor
    # transforms.ToPILImage() transform tensor to PIL
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(opt.image_size),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Train data loader
    dataset_train_object = SiameseDataset(MNISTTrain, invert_status=True, transform=transform)
    samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
    dataloaderTrain = DataLoader(dataset_train_object, batch_size=opt.batch_size,
                                 shuffle=False, num_workers=2, drop_last=True, sampler=samplerRandom)

    # Test data loader
    dataset_test_object = SiameseDataset(MNISTTest, invert_status=True, transform=transform)
    samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
    dataloaderTest = DataLoader(dataset_test_object, batch_size=opt.batch_size,
                                shuffle=False, num_workers=2, drop_last=True, sampler=samplerRandom)

    if opt.display_sample:
        # Show some training pairs
        real_batch = next(iter(dataloaderTrain))
        concatenated = torch.cat((real_batch[0].to(device)[:8], real_batch[1].to(device)[:8]), 0)

        plt.figure(figsize=(2, 8))
        plt.axis("off")
        plt.title("Training Pairs")
        plt.imshow(np.transpose(torchvision.utils.make_grid(concatenated).cpu(), (1, 2, 0)))
        plt.show()


####################
### Architecture ###
####################

class Model(nn.Module):
    def __init__(self, ngpu):
        super(Model, self).__init__()
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
        # self.conv4 = nn.ModuleDict({
        #     'conv': nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
        #     'bn': nn.BatchNorm2d(opt.ndf * 8),
        #     'activation': nn.LeakyReLU(0.2, inplace=True)
        # })
        #
        # self.conv5 = nn.ModuleDict({
        #     'conv': nn.Conv2d(opt.ndf * 8, opt.ndf * 16, 2, 1, 0, bias=False),
        #     'bn': nn.BatchNorm2d(opt.ndf * 16),
        # })

        self.fc1 = nn.ModuleDict({
            'dense': nn.Linear(opt.ndf * 4 * 4 * 4, 1024),
            'bn': nn.BatchNorm1d(1024),
            'activation': nn.LeakyReLU(0.2, inplace=True)
        })

        self.fc2 = nn.ModuleDict({
            'dense': nn.Linear(1024, 1024),
            # 'bn': nn.BatchNorm1d(128),
        })

    def forward_pass(self, input):
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

        # Flatten operation to connect cnn to fc
        out_flatten = out.view(out.size(0), -1)

        # Layer 4
        out = self.fc1['dense'](out_flatten)
        out = torch.sigmoid(out)

        return out

    def forward(self, input1, input2):
        output1 = self.forward_pass(input1)
        output2 = self.forward_pass(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Genuine and Impostor Pairs have "0" and "1" labels respectively.
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        try:
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        except:
            print(output1.shape)
            print(output2.shape)
            sys.exit()
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def weights_init(m):
    """
    Custom weight initialization.
    :param m: Input argument to extract layer type
    :return: Initialized architecture
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


#############
### Model ###
#############

# Initialize model
Model = Model(opt.num_gpu).to(device)

# Cost function
criterion = ContrastiveLoss()

####### CUDA ######

# Define cuda Tensors
Tensor = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1

if torch.cuda.device_count() > 1 and opt.multiplegpu:
    gpu_idx = list(range(opt.num_gpu))
    Model = nn.DataParallel(Model, device_ids=[gpu_idx[-1]])
    criterion = nn.DataParallel(criterion, device_ids=[gpu_idx[-1]])

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
Model.apply(weights_init)

######

# Optimizers
optimizer = torch.optim.Adam(Model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

################
### TRAINING ###
################
if opt.training:

    if opt.resume:
        #####################################
        #### Load model and optimizer #######
        #####################################

        # Loading the checkpoint
        checkpoint = torch.load(os.path.join(opt.PATH, "model_epoch.pth"))

        # Load models
        Model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizers
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load losses
        loss = checkpoint['loss']

        # Load epoch number
        epoch = checkpoint['epoch']

        Model.eval()

    iter_count = 0
    for epoch in range(opt.n_epochs):
        epoch_start = time.time()
        for i, data in enumerate(dataloaderTrain):
            iter_count += 1

            # load data
            img0 = data[0]
            img1 = data[1]
            labels = data[2]

            # Configure input
            img0 = Variable(img0.type(Tensor))
            img1 = Variable(img1.type(Tensor))
            labels = Variable(labels.type(Tensor))

            # Zero grads
            optimizer.zero_grad()

            # Generate a batch of images
            out1, out2 = Model(img0, img1)
            loss = criterion(out1, out2, labels)
            loss.backward()

            # read more at https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/4
            optimizer.step()

            if iter_count % opt.sample_interval == 0:
                print('TRAIN: [Epoch %d/%d] [Batch %d/%d] Loss: %.3f'
                      % (epoch + 1, opt.n_epochs, i, len(dataloaderTrain),
                         loss.item()), flush=True)

        with torch.no_grad():

            # Variables
            samples_test = next(iter(dataloaderTest))

            # Configure input
            img_test_0 = Variable(samples_test[0].type(Tensor))
            img_test_1 = Variable(samples_test[1].type(Tensor))
            labels_test = Variable(samples_test[2].type(Tensor))

            # Evaluate with model
            pairs_test_left, pairs_test_right = img_test_0, img_test_1
            out1_test, out2_test = Model(pairs_test_left, pairs_test_right)
            euclidean_distance = F.pairwise_distance(out1_test, out2_test, keepdim=False)

            y_true = labels_test.cpu().data.numpy()
            y_scores = euclidean_distance.cpu().data.numpy()
            # pos_label=1 or pos_label=0?
            # The y_scores is the distance. This means a higher y_scores indicates a larger distance.
            # This means pos_label=1 as for impostor pairs (higher distance) we set the label to 1 by default.
            # If we define y_scores = -euclidean_distance, then pos_label=0
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)

            # Caculate area under the curve (ROC)
            roc_auc = metrics.roc_auc_score(y_true, y_scores)
            roc_auc = metrics.auc(fpr, tpr)

            # Refer to https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision
            ap = metrics.average_precision_score(y_true, y_scores)

            # AUC and AP reporting
            print(
                "TEST: [Epoch %d/%d] [AUC: %.2f] [AP: %.2f]"
                % (epoch + 1, opt.n_epochs,
                   roc_auc.item(), ap)
                , flush=True)

        # End of epoch
        epoch_end = time.time()
        if opt.epoch_time_show:
            print("It has been {0} seconds for this epoch".format(epoch_end - epoch_start), flush=True)

        if (epoch + 1) % opt.epoch_save_model_freq == 0:
            # Refer to https://pytorch.org/tutorials/beginner/saving_loading_models.html
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': Model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(opt.expPATH, "model_siamese_epoch_%d.pth" % (epoch + 1)))

            # keep only the most recent 10 saved models
            # ls -d -1tr /home/sina/experiments/pytorch/model/* | head -n -10 | xargs -d '\n' rm -f
            # call("ls -d -1tr " + opt.expPATH + "/*" + " | head -n -10 | xargs -d '\n' rm -f", shell=True)

    # When training is finished, the ROC outputs should be saved
    np.save(os.path.join(opt.expPATH, "fpr_" + str(opt.manualSeed) + ".npy"), fpr, allow_pickle=False)
    np.save(os.path.join(opt.expPATH, "tpr_" + str(opt.manualSeed) + ".npy"), tpr, allow_pickle=False)

if opt.generate:

    ########## REAL DATA ##############

    # The order of transfor matters
    # transforms.ToTensor() transform numpy to tensor
    # transforms.ToPILImage() transform tensor to PIL
    transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    MNISTTest = torchvision.datasets.MNIST(root=opt.DATASETPATH, train=False, transform=transform,
                                           target_transform=None,
                                           download=True)

    print('Test data shape:', MNISTTest.data.shape)
    print('Test labels shape:', MNISTTest.targets.shape)

    samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=MNISTTest, replacement=True)
    dataloaderReal = DataLoader(MNISTTest, batch_size=opt.batch_size,
                                shuffle=False, num_workers=2, drop_last=True, sampler=samplerRandom)

    #####################################################################################################
    ####################################################################################################

    ########## FAKE DATA ##############
    # Generate features based on the discriminative model
    transformFake = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(opt.image_size),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, invert_status=True, transform=None):
            self.data = dataset
            self.dataSize = self.data.shape[0]
            self.transform = transform
            self.invert_status = invert_status
            self.transfortoPIL = torchvision.transforms.ToPILImage()

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            # Get image by index
            img = self.data[idx]

            if self.transform is not None:
                img = self.transform(img)

            return img

        def __len__(self):
            return self.dataSize


    # load fake data
    fakeData = np.load(
        os.path.join(os.path.expanduser('~/experiments/pytorch/model/' + 'wganmnist'), "fakeimages.npy"),
        allow_pickle=False)

    # Train data loader
    dataset_fake_object = FakeDataset(fakeData, invert_status=True, transform=transformFake)
    samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_fake_object, replacement=True)
    dataloaderFake = DataLoader(dataset_fake_object, batch_size=opt.batch_size,
                                shuffle=False, num_workers=2, drop_last=True, sampler=samplerRandom)

    ####################################################################################################
    ####################################################################################################

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(opt.expPATH, "model_siamese_epoch_3.pth"))

    # Load models
    Model.load_state_dict(checkpoint['model_state_dict'])

    # insert weights [required]
    Model.eval()

    # Process the real data
    print('processing real data...')
    real_img_features_status = False
    for i, data in enumerate(dataloaderReal):

        # load data
        realImg = data[0]
        label = data[1]

        # Configure input
        realImg = Variable(realImg.type(Tensor))
        label = Variable(label.type(Tensor))

        if (i + 1) % 100 == 0:
            print('Processing {}-th real sample'.format(i + 1))

        # Generate a batch of images
        out, _ = Model(realImg, realImg)
        numpy_out = out.cpu().data.numpy()
        numpy_realImg = realImg.cpu().data.numpy()
        if not real_img_features_status:
            real_img_features = numpy_out
            real_img_matrix = numpy_realImg
            real_img_labels = label.cpu().data.numpy()
            real_img_features_status = True
        else:
            real_img_features = np.append(real_img_features, numpy_out, axis=0)
            real_img_matrix = np.append(real_img_matrix, numpy_realImg, axis=0)
            real_img_labels = np.append(real_img_labels, label.cpu().data.numpy(), axis=0)

    real_img_features = real_img_features.astype(np.float32)
    # Channel order is different in Matplotlib and PyTorch
    # PyTorch : [NCWH]
    # Numpy: [NWHC]
    real_img_matrix = np.transpose(real_img_matrix.astype(np.float32), (0, 2, 3, 1))

    # back into the range of [0,1]
    real_img_matrix = real_img_matrix * 0.5 + 0.5

    # for numpy saving
    real_img_matrix = real_img_matrix.astype(np.float32)

    if opt.display_sample:
        N = 3
        sel_real_imgs = real_img_matrix[0:N * N, :, :, :]

        # matplotlib.pyplot.imshow() needs a 2D array, or a 3D array with the third dimension being of shape 3 or 4!
        # Do repeat to copy along the thrid axis for having an RGB.
        sel_real_imgs = np.repeat(sel_real_imgs, 3, axis=3)
        f, axarr = plt.subplots(N, N)
        for i in range(N):
            for j in range(N):
                axarr[i, j].imshow(sel_real_imgs[i + j])
        plt.show()

    # process fake images
    print('processing fake data...')
    fake_img_features_status = False
    for j, fakedata in enumerate(dataloaderFake):

        fakeImg = Variable(fakedata.type(Tensor))
        if (j + 1) % 100 == 0:
            print('Processing {}-th fake sample'.format(j + 1))
        out, _ = Model(fakeImg, fakeImg)
        numpy_out = out.cpu().data.numpy()
        numpy_fakeImg = fakeImg.cpu().data.numpy()
        if not fake_img_features_status:
            fake_img_features = numpy_out
            fake_img_matrix = numpy_fakeImg
            fake_img_features_status = True
        else:
            fake_img_features = np.append(fake_img_features, numpy_out, axis=0)
            fake_img_matrix = np.append(fake_img_matrix, numpy_fakeImg, axis=0)

    fake_img_features = fake_img_features.astype(np.float32)

    # back into the range of [0,1]
    fake_img_matrix = fake_img_matrix * 0.5 + 0.5

    # Channel order is different in Matplotlib and PyTorch
    # PyTorch : [NCWH]
    # Numpy: [NWHC]
    fake_img_matrix = np.transpose(fake_img_matrix, (0, 2, 3, 1))
    fake_img_matrix = fake_img_matrix.astype(np.float32)

    if opt.display_sample:
        N = 3
        sel_fake_imgs = fake_img_matrix[0:N * N, :, :, :]

        # matplotlib.pyplot.imshow() needs a 2D array, or a 3D array with the third dimension being of shape 3 or 4!
        # Do repeat to copy along the thrid axis for having an RGB.
        sel_fake_imgs = np.repeat(sel_fake_imgs, 3, axis=3)
        f, axarr = plt.subplots(N, N)
        for i in range(N):
            for j in range(N):
                axarr[i, j].imshow(sel_fake_imgs[i + j])
        plt.show()

    np.save(os.path.join(opt.expPATH, "real_img_features.npy"), real_img_features, allow_pickle=False)
    np.save(os.path.join(opt.expPATH, "real_img_matrix.npy"), real_img_matrix, allow_pickle=False)
    np.save(os.path.join(opt.expPATH, "real_img_labels.npy"), real_img_labels, allow_pickle=False)
    np.save(os.path.join(opt.expPATH, "fake_img_features.npy"), fake_img_features, allow_pickle=False)
    np.save(os.path.join(opt.expPATH, "fake_img_matrix.npy"), fake_img_matrix, allow_pickle=False)

    ####### Generate and save distances ####
    # Status
    one_random_fake = False
    all_fake = True

    if one_random_fake:
        # random sample
        rand_idx = int(np.random.randint(fake_img_features.shape[0], size=1))
        sample_fake_feature = fake_img_features[rand_idx]
        sample_fake = fake_img_matrix[rand_idx]

        # Compare to all real ones
        min = False
        index_min = False
        for i in range(real_img_features.shape[0]):
            euclidean_distance = np.linalg.norm(sample_fake_feature - real_img_features[i], ord=2)
            if (i + 1) % 100 == 0:
                print('Processed {}-th image'.format(i))
                print('The euclidean distance is {}:'.format(euclidean_distance))
            if not min:
                min = euclidean_distance
                index_min = i
            else:
                if euclidean_distance < min:
                    min = euclidean_distance
                    index_min = i

        print('index:', index_min)
        print('min:', min)

        # print the match
        imgPair = np.concatenate((sample_fake, real_img_matrix[index_min]), axis=1)
        imgPair = np.repeat(imgPair, 3, axis=2)
        plt.imshow(imgPair)
        plt.show()

    if all_fake:

        dist = np.zeros((fake_img_features.shape[0], real_img_features.shape[0]), dtype=np.float32)
        dist_min = np.zeros((fake_img_features.shape[0], ), dtype=np.float32)
        label_fake = np.zeros((fake_img_features.shape[0], ), dtype=np.int32)
        index_min = np.zeros((fake_img_features.shape[0], ), dtype=np.int32)
        for j in range(fake_img_features.shape[0]):
            if (j + 1) % 100 == 0:
                print('Processed {}-th fake image'.format(j))
            # Compare to all real ones
            min = False
            index_min = False
            for i in range(real_img_features.shape[0]):
                euclidean_distance = np.linalg.norm(fake_img_features[j] - real_img_features[i], ord=2)
                dist[j, i] = euclidean_distance
                if not min:
                    min = euclidean_distance
                    index_min = i
                else:
                    if euclidean_distance < min:
                        min = euclidean_distance
                        index_min = i

            # The index of the matrched real img
            idx_min[j] = index_min

            # The minimum distance of fake image to its real counterpart
            dist_min[j] = min

            # The label that the fakes are classfied with
            label_fake[j] = real_img_labels[int(index_min)]

        np.save(os.path.join(opt.expPATH, "evalDist.npy"), dist, allow_pickle=False)


if opt.evaluate:

    print('reading files...')
    # load features and labels for real images and also features for fake images
    real_img_features = np.load(os.path.join(opt.expPATH, "real_img_features.npy"), allow_pickle=False)
    real_img_matrix = np.load(os.path.join(opt.expPATH, "real_img_matrix.npy"), allow_pickle=False)
    real_img_labels = np.load(os.path.join(opt.expPATH, "real_img_labels.npy"), allow_pickle=False)
    fake_img_features = np.load(os.path.join(opt.expPATH, "fake_img_features.npy"), allow_pickle=False)
    fake_img_matrix = np.load(os.path.join(opt.expPATH, "fake_img_matrix.npy"), allow_pickle=False)
    evalDist = np.load(os.path.join(opt.expPATH, "evalDist.npy"), allow_pickle=False)
    print(evalDist.shape, real_img_features.shape, real_img_matrix.shape, real_img_labels.shape,
          fake_img_features.shape, fake_img_matrix.shape)

    # # UNCOMMENT TO SEE SOME FAKE SAMPLES

    ########################################
    ########### Display fake ###############
    ########################################
    # if opt.display_sample:
    #     # display fake
    #     N = 3
    #     idx_fake_rand = np.random.choice(fake_img_matrix.shape[0], N * N)
    #     sel_fake_imgs = fake_img_matrix[idx_fake_rand, :, :, :]
    #
    #     # matplotlib.pyplot.imshow() needs a 2D array, or a 3D array with the third dimension being of shape 3 or 4!
    #     # Do repeat to copy along the thrid axis for having an RGB.
    #     sel_fake_imgs = np.repeat(sel_fake_imgs, 3, axis=3)
    #     f, axarr = plt.subplots(N, N)
    #     for i in range(N):
    #         for j in range(N):
    #             axarr[i, j].imshow(sel_fake_imgs[i + j])
    #     plt.show()

    ########################################
    ########### Evaluate  ##################
    ########################################

    print('calculating minimum distances...')
    # Get the index associated with minimum distance
    idx_sel = np.argmin(evalDist, axis=1)
    dist_sel = evalDist[np.arange(evalDist.shape[0]),idx_sel]

    # Take the detected label
    real_img_labels_matrix = real_img_labels.reshape((1, real_img_labels.shape[0]))
    real_img_labels_matrix = np.repeat(real_img_labels_matrix, fake_img_features.shape[0], axis=0)
    label_fake = real_img_labels_matrix[np.arange(real_img_labels_matrix.shape[0]),idx_sel].astype(int)
    print('calculating minimum distances is finished!')

    ##########################################
    ######## Display one random match ########
    ##########################################
    # random sample
    rand_idx = int(np.random.randint(fake_img_features.shape[0], size=1))
    sample_fake_feature = fake_img_features[rand_idx]
    sample_fake = fake_img_matrix[rand_idx]

    # print the match
    label_detect = label_fake[rand_idx]
    dist_sel_temp = dist_sel[rand_idx]
    print('label_detect', label_detect)
    print('dist_sel', dist_sel_temp)
    imgPair = np.concatenate((sample_fake, real_img_matrix[idx_sel[rand_idx]]), axis=1)
    imgPair = np.repeat(imgPair, 3, axis=2)
    plt.imshow(imgPair)
    plt.show()

    ###############################
    ######## Check quality ########
    ###############################

    # Here we sort distances and see if the score makes sense.
    # For lower quality matches we should get a higher distance

    # Return the indexes that sort the array
    distance_sorted_idx = np.argsort(dist_sel, axis=None)
    min = dist_sel[distance_sorted_idx[0]]
    max = dist_sel[distance_sorted_idx[-1]]
    print(min, max)

    # Plot some samples
    N = 9
    # Skip the first 10 images for reducing bias (10 should be zero if starting from the lowest distance)
    index_range = np.arange(10,distance_sorted_idx.shape[0], int(distance_sorted_idx.shape[0] / float(N)))
    index_fake_sample = distance_sorted_idx[index_range]
    index_real_sample = idx_sel[distance_sorted_idx[index_range]]
    fake_img_list = []
    for i in range(index_fake_sample.shape[0]):
        fake_img = fake_img_matrix[index_fake_sample[i]]
        fake_img = np.repeat(fake_img, 3, axis=2)
        fake_img_list.append(fake_img)

    real_img_list = []
    for i in range(index_real_sample.shape[0]):
        real_img = real_img_matrix[index_real_sample[i]]
        real_img = np.repeat(real_img, 3, axis=2)
        real_img_list.append(real_img)

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                     axes_pad=0.5,  # pad between axes in inch.
                     )
    count = 0
    for ax, im in zip(grid, fake_img_list):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title('D: {:.2f}, L: {}'.format(dist_sel[index_fake_sample[count]],label_fake[index_fake_sample[count]]))
        count+= 1

    plt.show()

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                     axes_pad=0.5,  # pad between axes in inch.
                     )
    count = 0
    for ax, im in zip(grid, real_img_list):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        count += 1

    plt.show()



    sys.exit()

    # Load frps and tprs
    # The frp and tpr for different runs of experiments were previously saved.
    fpr_list = []
    tpr_list = []
    import os

    # Get all
    for fname in os.listdir(opt.expPATH):  # change directory as needed
        if 'fpr' in fname:
            fpr_file = np.load(os.path.join(opt.expPATH, fname), allow_pickle=False)
            fpr_list.append(fpr_file)

            # Same tpr file
            tpr_file_name = 'tpr_' + fname.split('_')[1]
            tpr_file = np.load(os.path.join(opt.expPATH, tpr_file_name), allow_pickle=False)
            tpr_list.append(tpr_file)

    from scipy import interp
    from sklearn.metrics import roc_curve, auc

    # #############################################################################
    # Data IO and generation
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    # Each experiment fpr and tpr are used the loop
    for ii in range(len(tpr_list)):
        fpr, tpr = fpr_list[ii], tpr_list[ii]
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC run %d (AUC = %0.2f)' % (i + 1, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Flipping the Coin', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
