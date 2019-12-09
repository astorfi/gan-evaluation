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

# Pytorch library
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch

# experimentName is the current file name without extension
evaluationName = 'wgancnnmimic'

parser = argparse.ArgumentParser()
parser.add_argument("--DATASETPATH", type=str,
                    default=os.path.expanduser('~/data/PhisioNet/MIMIC/processed/out_binary.matrix'),
                    help="Dataset file")

parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")
parser.add_argument("--num_pairs", type=int, default=100000, help="number of pairs")

parser.add_argument("--training", type=bool, default=False, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Resume training or not")
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/model/'+evaluationName),
                    help="Training status")
parser.add_argument("--modelPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/model/'+evaluationName+'/model'),
                    help="Training status")

parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="l2 regularization")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

parser.add_argument("--sample_interval", type=int, default=100, help="interval between samples")
parser.add_argument("--epoch_time_show", type=bool, default=False, help="interval betwen image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=100, help="number of epops per model save")


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
if not os.path.exists(opt.modelPATH):
    os.system('mkdir {0}'.format(opt.modelPATH))

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

##########################
### Dataset Processing ###
##########################

# ave synthetic data
trainData = np.load(os.path.join(opt.expPATH, "dataTrain.npy"), allow_pickle=False)
testData = np.load(os.path.join(opt.expPATH, "dataTest.npy"), allow_pickle=False)
synData = np.load(os.path.join(opt.expPATH, "synthetic.npy"), allow_pickle=False)

class Dataset:
    # TRIAINING SETUP: GEN: train-train IMP: train-test
    # TESTING SETUP: GEN: synthetic-train IMP: synthetic-test
    def __init__(self, trainData, testData, synData, trainstatus, num_pairs=1000, transform=None):

        # Transform
        self.transform = transform
        self.num_pairs = num_pairs

        # load data here
        self.trainData = trainData
        self.testData = testData
        self.synData = synData
        self.featureSize = trainData.shape[1]

        # Create data matrix
        self.data = np.zeros((self.num_pairs, self.featureSize, 2), dtype=trainData.dtype)
        self.label = np.zeros((self.num_pairs), dtype=trainData.dtype)


        if trainstatus:
            idx = 0
            while idx < int(self.num_pairs):

                # Genuine pair
                a = self.trainData[random.randint(0, self.trainData.shape[0]-1), :]
                b = self.trainData[random.randint(0, self.trainData.shape[0]-1), :]
                genPair = np.stack([a, b], axis=-1)
                self.data[idx] = genPair
                self.label[idx] = 0
                idx += 1

                # Imposter pair
                a = self.trainData[random.randint(0, self.trainData.shape[0]-1), :]
                b = self.testData[random.randint(0, self.testData.shape[0]-1), :]
                impPair = np.stack([a, b], axis=-1)
                self.data[idx] = impPair
                self.label[idx] = 1
                idx += 1

        else:
            idx = 0
            while idx < int(self.num_pairs):

                # Genuine pair
                a = self.trainData[random.randint(0, self.trainData.shape[0]-1), :]
                b = self.synData[random.randint(0, self.synData.shape[0]-1), :]
                genPair = np.stack([a, b], axis=-1)
                self.data[idx,:] = genPair
                self.label[idx] = 0
                idx += 1

                # Imposter pair
                a = self.synData[random.randint(0, self.synData.shape[0]-1), :]
                b = self.testData[random.randint(0, self.testData.shape[0]-1), :]
                impPair = np.stack([a, b], axis=-1)
                self.data[idx, :] = impPair
                self.label[idx] = 1
                idx += 1


    def return_data(self):
        return self.data, self.label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.data[idx]
        label = self.label[idx]
        pair = np.clip(pair, 0, 1)
        sample = {'pair': torch.from_numpy(np.asarray(pair)), 'label': torch.from_numpy(np.asarray(label))}

        if self.transform:
           pass

        return sample


# Train data loader
dataset_train_object = Dataset(trainData, testData, synData, num_pairs=opt.num_pairs, trainstatus=True, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
dataloader_train = DataLoader(dataset_train_object, batch_size=opt.batch_size,
                              shuffle=False, num_workers=2, drop_last=True, sampler=samplerRandom)

# Test data loader
dataset_test_object = Dataset(trainData, testData, synData, num_pairs=opt.num_pairs, trainstatus=False, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
dataloader_test = DataLoader(dataset_test_object, batch_size=opt.batch_size,
                             shuffle=False, num_workers=1, drop_last=True, sampler=samplerRandom)

# Generate random samples for test
random_samples = next(iter(dataloader_test))
feature_size = random_samples['pair'].size()[1]

####################
### Architecture ###
####################

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Discriminator's parameters
        self.disDim = 128

        self.model = nn.Sequential(
            nn.Linear(dataset_train_object.featureSize, 4 * self.disDim),
            nn.BatchNorm1d(4 * self.disDim, eps=0.001, momentum=0.01),
            nn.ReLU(True),
            nn.Linear(4 * self.disDim, 2 * self.disDim),
            nn.BatchNorm1d(2 * self.disDim, eps=0.001, momentum=0.01),
            nn.ReLU(True),
            nn.Linear(2 * self.disDim, self.disDim),
            nn.BatchNorm1d(self.disDim, eps=0.001, momentum=0.01),
            nn.Sigmoid()
        )

    def forward_pass(self, x):
        # Feeding the model
        output = self.model(x)
        return output

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
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=False)
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
Model = Model()

# Cost function
criterion = ContrastiveLoss()

# Define cuda Tensors
Tensor = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1


if torch.cuda.device_count() > 1 and opt.multiplegpu:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  Model = nn.DataParallel(Model, list(range(opt.num_gpu)))

if opt.cuda:
    """
    model.cuda() will change the model inplace while input.cuda()
    will not change input inplace and you need to do input = input.cuda()
    ref: https://discuss.pytorch.org/t/when-the-parameters-are-set-on-cuda-the-backpropagation-doesnt-work/35318
    """
    Model.cuda()
    one, mone = one.cuda(), mone.cuda()
    Tensor = torch.cuda.FloatTensor

# Weight initialization
Model.apply(weights_init)

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
        for i, samples in enumerate(dataloader_train):
            iter_count += 1

            pairs = samples['pair']
            labels = samples['label']

            # Configure input
            pairs = Variable(pairs.type(Tensor))
            labels = Variable(labels.type(Tensor))

            # Zero grads
            optimizer.zero_grad()

            # Generate a batch of images
            pair_left, pair_right = pairs[:, :, 0], pairs[:, :, 1]
            out1, out2 = Model(pair_left, pair_right)
            loss = criterion(out1, out2, labels)
            loss.backward()

            # read more at https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/4
            optimizer.step()


            if iter_count % opt.sample_interval == 0:
                print('TRAIN: [Epoch %d/%d] [Batch %d/%d] Loss: %.3f'
                      % (epoch + 1, opt.n_epochs, i, len(dataloader_train),
                         loss.item()), flush=True)

        with torch.no_grad():

            # Variables
            samples_test = next(iter(dataloader_test))

            # Configure input
            pairs_test = Variable(samples_test['pair'].type(Tensor))
            labels_test = Variable(samples_test['label'].type(Tensor))

            # Evaluate with model
            pairs_test_left, pairs_test_right = pairs_test[:, :, 0], pairs_test[:, :, 1]
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
            }, os.path.join(opt.modelPATH, "model_siamese_epoch_%d.pth" % (epoch + 1)))

            # keep only the most recent 10 saved models
            # ls -d -1tr /home/sina/experiments/pytorch/model/* | head -n -10 | xargs -d '\n' rm -f
            call("ls -d -1tr " + opt.modelPATH + "/*" + " | head -n -10 | xargs -d '\n' rm -f", shell=True)


    # When training is finished, the ROC outputs should be saved
    np.save(os.path.join(opt.expPATH, "fpr_"+str(opt.manualSeed)+".npy"), fpr, allow_pickle=False)
    np.save(os.path.join(opt.expPATH, "tpr_"+str(opt.manualSeed)+".npy"), tpr, allow_pickle=False)


else:
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
                 label='ROC run %d (AUC = %0.2f)' % (i+1, roc_auc))

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