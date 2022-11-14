import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def image_generator(images_csv, lbl=True):
    """
    Args:
        @input: opened csv with the images as cols 
        @lbl: if True the csv contains corresponding labels
        @returns: labels and images reshaped - numpy arrays

    Function that uses an already opened csv file and returns the images reshaped in (28,28,1)
    and (if any) the corresponding label    
    """
    labels = 0
    if (lbl == True):
        labels = images_csv[images_csv.columns[0]
                            ].to_numpy(dtype='uint8', copy=True)

    images = images_csv.loc[:, 'pixel0':'pixel783'].to_numpy(
        dtype=np.float32, copy=True)
    # Reshape 28x28x1
    images = images.reshape((len(images), 28, 28, 1))
    return images, labels


class kannadaDataset(torch.utils.data.Dataset):
    """
    Custom class for reading/loading and augmenting the given dataset        
    """

    def __init__(self, data, label=None, aug=False):
        """
        Args:
            data : images
            label (optional): labels of the images. Defaults to None.
            aug (bool, optional): if data augmentations should be performd - True for training otherwise default=False.

            Original given data and their images are copied 5 times. Each set will be augmented depending on the category
            aug1: Rotation 10 degrees 
            aug2: CenterCrop + Resize (Zoom)
            aug3: RandomAffine + Cloclwise Rotation 10 degrees
            aug4: crop + RandomAffine + RandomRotation +Resize
            aug5: Cloclwise Rotation 10 degrees
        """
        self.aug = aug
        if (label is not None) and aug == True:
            self.data = [(el, 'none') for el in data]
            self.label = [el_l for el_l in label]

            tmp_data = []
            tmp_labels = []
            for i in range(len(data)):
                el_data = data[i]
                el_l = label[i]
                tmp_data.append((el_data, 'augm1'))
                tmp_labels.append(el_l)
            self.data.extend(tmp_data)
            self.label.extend(tmp_labels)

            tmp_data = []
            tmp_labels = []
            for i in range(len(data)):
                el_data = data[i]
                el_l = label[i]
                tmp_data.append((el_data, 'augm2'))
                tmp_labels.append(el_l)
            self.data.extend(tmp_data)
            self.label.extend(tmp_labels)

            tmp_data = []
            tmp_labels = []
            for i in range(len(data)):
                el_data = data[i]
                el_l = label[i]
                tmp_data.append((el_data, 'augm3'))
                tmp_labels.append(el_l)
            self.data.extend(tmp_data)
            self.label.extend(tmp_labels)

            tmp_data = []
            tmp_labels = []
            for i in range(len(data)):
                el_data = data[i]
                el_l = label[i]
                tmp_data.append((el_data, 'augm4'))
                tmp_labels.append(el_l)
            self.data.extend(tmp_data)
            self.label.extend(tmp_labels)

            tmp_data = []
            tmp_labels = []
            for i in range(len(data)):
                el_data = data[i]
                el_l = label[i]
                tmp_data.append((el_data, 'augm5'))
                tmp_labels.append(el_l)
            self.data.extend(tmp_data)
            self.label.extend(tmp_labels)

        else:
            self.data = data
            self.label = label

        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor()
        ])

        self.transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(size=(18, 18)),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),

        ])
        self.transform3 = transforms.Compose([
            transforms.ToPILImage(),
            torchvision.transforms.RandomAffine(
                degrees=0, translate=(0.2, 0.2), shear=0.1),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.Resize((28, 28)),
            transforms.ToTensor()

        ])
        self.transform4 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(size=(18, 18)),
            torchvision.transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), shear=0.1),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),

        ])

        self.transform5 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(-10, -10)),
            transforms.ToTensor()
        ])

    def __getitem__(self, i):
        if (self.aug == True):

            img = self.data[i][0]

            if self.data[i][1] == 'augm1':
                img = self.transform1(img)

            if self.data[i][1] == 'augm2':
                img = self.transform2(img)

            if self.data[i][1] == 'augm3':
                img = self.transform3(img)

            if self.data[i][1] == 'augm4':
                img = self.transform4(img)

            if self.data[i][1] == 'augm5':
                img = self.transform5(img)

            if self.data[i][1] == 'none':
                img = transforms.ToTensor()(img)
        else:
            img = self.data[i]
            img = transforms.ToTensor()(img)

        if self.label is not None:
            label = self.label[i]
            return img, label  # train.val
        else:

            return img  # test

    def __len__(self):
        return len(self.data)


class CNN_Model(nn.Module):
    """
    Class that defines the model and it's layers.
    It's a CNN.
    """

    def __init__(self):
        super(CNN_Model, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(1, 64, 3),
            nn.ReLU(),  # 26
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3),
            nn.ReLU(),  # 24
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(0.1),  # 22
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.MaxPool2d(2, 2),  # 10
            nn.Dropout(p=0.2),

            nn.Conv2d(128, 256, 3),
            nn.ReLU(),  # 8
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3),
            nn.ReLU(),  # 6
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 5, padding='same'),
            nn.ReLU(),  # 6
            nn.BatchNorm2d(256),

            nn.MaxPool2d(2, 2),  # 3
            nn.Dropout(p=0.2),
        )

        self.classify = nn.Sequential(
            nn.Linear(2304, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 10),
        )

    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x


class Trainer(object):
    """
    Class defined for easier training 
    """

    def __init__(self,  folder, epochs=17, lr=0.001, batch_size=128):
        self.cnn = CNN_Model()
        self.folder = folder
        self.epochs = epochs
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.BATCH_SIZE = batch_size

    def read_data(self):
        # Data path
        train_set = os.path.join(self.folder, "train.csv")
        valid_set = os.path.join(self.folder, "Dig-MNIST.csv")
        test_set = os.path.join(self.folder, "test.csv")

        # Read data
        train_csv = pd.read_csv(train_set)
        valid_csv = pd.read_csv(valid_set)

        train_images, train_labels = image_generator(train_csv)
        valid_images, valid_labels = image_generator(valid_csv)

        train_data, validation_data, train_label, validation_label = train_test_split(np.concatenate((train_images, valid_images)),
                                                                                      np.concatenate(
                                                                                          (train_labels, valid_labels)),
                                                                                      test_size=0.2,
                                                                                      shuffle=True)

        # DatasetLoaders
        trainset = kannadaDataset(train_data, train_label, aug=True)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.BATCH_SIZE, shuffle=True)

        valset = kannadaDataset(validation_data, validation_label, aug=True)
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=self.BATCH_SIZE, shuffle=False)
        print("Train data: ", len(train_loader))
        print("Test data: ", len(val_loader))
        return train_loader, val_loader

    def train(self, verbose=False):
        """
        Main training and validation loop

        Args:
            verbose (bool, optional): if metrices should be printed per epochs. Defaults to False.

        Returns:
            cnn: the trained model
            nn_output : 2D list with all the metrics per epochs
        """

        print('Device: ', self.device)
        train_loader, val_loader = self.read_data()
        self.cnn.to(self.device)
        print('building model......')

        print('start training......')

        criterion = nn.CrossEntropyLoss()

        # optimizer=torch.optim.Adam(cnn.parameters(),lr=lr,weight_decay=0.001)
        # lr_scheduler=ReduceLROnPlateau(optimizer,mode='min',factor=0.75,patience=5)

        optimizer=torch.optim.SGD(self.cnn.parameters(),lr=self.lr,momentum=0.99,weight_decay=0.001)
        lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        # train
        nn_output = []
        for i in range(self.epochs):
            self.cnn.train()
            loss_total = 0
            train_correct = 0

            for idx, (img, label) in enumerate(train_loader):
                img = img.to(self.device)
                label = label.to(self.device)
                output = self.cnn(img)
                loss = criterion(output, label)
                loss_total += loss.item()
                prediction_tr = torch.max(output, dim=1)[1]
                train_correct += sum(prediction_tr == label).item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

               
            lr_scheduler.step()

            print('validate......')
            with torch.no_grad():
                self.cnn.eval()
                correct = 0
                val_loss = 0
                n_samples = 0

                for idx, (img, label) in enumerate(val_loader):
                    img = img.to(self.device)
                    label = label.to(self.device)
                    output = self.cnn(img)
                    vloss = criterion(output, label)
                    val_loss += vloss.item()
                    prediction = torch.max(output, dim=1)[1]
                    correct += sum(prediction == label).item()
                    n_samples += label.size(0)

            metrics = [i + 1,
                       loss_total/len(train_loader.dataset), train_correct /
                       len(train_loader.dataset)*100,
                       val_loss/len(val_loader.dataset), correct/len(val_loader.dataset)*100]
            nn_output.append(metrics)

            if (verbose == True):
                print('epoch: {} , train_loss: {}, train_acc: {}, val_loss: {}, val_acc:{} '.format(
                    metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))
        return self.cnn, nn_output


def loss_acc_graph(nn_output):
    """
    Returns 2 graphs the train/val loss and train/val accuracy 
    Used for overfitting check

    Args:
        nn_output (2D list): the 2D metrics list that is produced from training 
    """
    pd_results = pd.DataFrame(nn_output, columns=['epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc']
                              )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    axes[0].plot(pd_results['epoch'], pd_results['valid_loss'],
                 label='validation_loss')
    axes[0].plot(pd_results['epoch'],
                 pd_results['train_loss'], label='train_loss')

    axes[0].legend()

    axes[1].plot(pd_results['epoch'], pd_results['valid_acc'],
                 label='validation_acc')
    axes[1].plot(pd_results['epoch'],
                 pd_results['train_acc'], label='train_acc')

    axes[1].legend()
    fig.suptitle("Loss and Accuracy graphs", fontweight="bold")
    plt.show(block=True)


def test(cnn, test_data_path, BATCH_SIZE=128):
    """
    Function used for testing the test dataset and creating the required submission.csv file

    Args:
        cnn : the trained model
        test_data_path (string): the full path that the test.csv file is in
        BATCH_SIZE (int, optional): the batchsize. Defaults to 128.
    """
    
    testdata = pd.read_csv(test_data_path)
    testset = testdata.drop('id', axis=1)
    test_data = testset.to_numpy(
        dtype=np.float32).reshape(len(testset), 28, 28, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testset = kannadaDataset(test_data)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False)

    print('testing data......')

    with open('submission.csv', 'w', newline='') as f:
        submission = csv.writer(f)
        submission.writerow(['id', 'label'])
        with torch.no_grad():
            cnn.eval()
            for idx, img in enumerate(test_loader):
                img = img.to(device)
                img = img.float()
                output = cnn(img)
                _, predictions = torch.max(output, 1)
                # print(predictions.shape)
                for i in range(len(img)):
                    submission.writerow(
                        [idx*BATCH_SIZE+i, predictions[i].item()])

    print('finished......')
