import numpy as np 
import matplotlib.pyplot as plt

# MNIST data
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms

import timeit
import pickle
import os



class MLP(object):

    def __init__(self, sizes, learning_rate, init_scale=1):
        """
        Initializing weights and biases
        """
        self.weights = np.array([np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])], dtype=object)*init_scale
        self.biases = np.array([np.random.randn(y) for y in sizes[1:]], dtype=object)*init_scale
        self.lr = learning_rate
        self.c0 = np.array([])

    def forward_prop(self, data):

        self.z = []
        self.a = []

        self.a.append(data)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            self.z.append(np.dot(self.a[-1], w.T)+b)
            self.a.append(self.reLu(self.z[-1]))
        self.z.append(np.dot(self.a[-1], self.weights[-1].T)+self.biases[-1])
        self.a.append(self.reLu(self.z[-1]))

        

    def grad(self, data):

        # forward
        self.forward_prop(data[0])
        
        # cummulating cost of a batch
        self.c0 = np.append(self.c0,np.sum((self.a[-1]-data[1])**2)/len(data[0]))

        # back
        dw = np.empty_like(self.weights)
        db = np.empty_like(self.biases)
        n = self.a[-1]-data[1]
        dw[-1] = np.dot(n.T, self.a[-2])
        db[-1] = np.sum(n, axis=0)
        for i in range(1, len(self.weights)):
            n = np.dot(self.weights[-i].T, db[-i])*self.dreLu(self.z[-i-1])
            dw[-i-1] = np.dot(n.T, self.a[-i-2])
            db[-i-1] = np.sum(n, axis=0)

        return (dw, db)


    def reLu(self, x):
        return np.maximum(0, x)
    def dreLu(self, x):
        return (x>0)

    def train(self, training_data, training_labels, batchSize, epoch): 

        startTime = timeit.default_timer() # timing

        # batch split remainder of numBatches handler
        remainder = (len(training_data) % batchSize)
        if (remainder != 0):
            print('Remaining data left unsplit:' + str(remainder) + '\n')
            training_data = training_data[:-remainder]
            training_labels = training_labels[:-remainder]
        numBatches = ((len(training_data) - remainder)//batchSize)


        # splitting data into batches
        din = [training_data[k:k+batchSize] for k in range(0, numBatches*batchSize, batchSize)]
        labels = [training_labels[k:k+batchSize] for k in range(0, numBatches*batchSize, batchSize)]
        batchedData = list(zip(din, labels))

        for t in range(0, epoch):

            print('\n' + 'epoch: ' + str(t+1))
            print('\n' + '---------------------')

            correct = 0
            for batch_idx, batch in enumerate(batchedData):
                grad = self.grad(batch)
                # if (batchedData.index(batch) % 100 == 0): print('Cost per ith batch = ' + str((self.c0[-1])))
                self.weights = self.weights - grad[0]*self.lr
                self.biases = self.biases - grad[1]*self.lr

                # Total correct predictions
                predicted = np.argmax(self.a[-1], axis=1)
                target = np.argmax(batch[1], axis=1)
                correct += (predicted == target).sum()
                #print(correct)
                if batch_idx % 50 == 0:
                    print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                        epoch, batch_idx*len(batch), len(training_data), 100.*batch_idx / len(batchedData), self.c0[0], float(correct*100) / float(batch_size*(batch_idx+1))))


        print('\n' + 'time training = ' + str(timeit.default_timer() - startTime) + ' (s)') # timing
        plt.plot(np.linspace(0,len(self.c0), len(self.c0)), self.c0)
        plt.draw()


    def test(self, testing_data, testing_labels):

        # forward pass of testing data
        self.forward_prop(testing_data)
        # mse
        c0 = np.sum((self.a[-1]-testing_labels)**2)/len(testing_data)

        correct = 0
        # Total correct predictions
        predicted = np.argmax(self.a[-1], axis=1)
        target = np.argmax(testing_labels, axis=1)
        correct = (predicted == target).sum()
        #print(correct)
        print('({}/{})\t Loss:{:.3f}\t Accuracy:{:.3f}%'.format(correct, len(testing_data), self.c0[0], 100*(correct / len(testing_data))))
        

    def saveNetwork(self, weightPath, biasesPath):
        with open(weightPath, 'wb') as file:
            pickle.dump(self.weights, file)
        with open(biasesPath, 'wb') as file:
            pickle.dump(self.biases, file)
        print('\n'+'Network saved')

    def loadNetwork(self, weightPath, biasesPath):
        with open(weightPath, 'rb') as file:
            self.weights = pickle.load(file)
        with open(biasesPath, 'rb') as file:
            self.biases = pickle.load(file)
        print('\n'+'Network loaded')

### MNIST from torchvision ###

def get_mnist():
    
    DIR = os.getcwd()

    # if MNIST not found in directory, will download to directory
    dl = False
    if (not os.path.isdir("MNIST")):
        dl = True
    
    train_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(root=DIR, 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=dl), 
                                    shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(root=DIR, 
                                            train=False, 
                                            transform=transforms.ToTensor(),
                                            download=dl),
                                    shuffle=True) 
    return train_loader, test_loader
    

# integer label -> class vector
def vectorized_result(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e


# structure parameters
layers = [784,16,16,10]
learning_rate = .001
init_scale = .01

# training parameters
batch_size = 7
epochs = 2

# model initialization
model = MLP(layers, learning_rate, init_scale)

# getting mnist from pytorch datasets
train, test = get_mnist()

# flattening and normalizing 2d pixel data, hot encoded labels
training_data = train.dataset.data.numpy().reshape(len(train), 784) / 255
training_labels = [vectorized_result(i) for i in train.dataset.targets.numpy()]
# flattening and normalizing 2d pixel data, hot encoded labels
testing_data = test.dataset.data.numpy().reshape(len(test), 784) / 255
testing_labels = [vectorized_result(i) for i in test.dataset.targets.numpy()]


# training
model.train(training_data, training_labels, batch_size, epochs)
# test
model.test(testing_data, testing_labels)

