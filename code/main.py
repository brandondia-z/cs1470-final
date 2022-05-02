from lib2to3.pytree import LeafPattern
import torch
import torchvision
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from yellowbrick.classifier import ROCAUC
from preprocess import get_data
import numpy as np
from sklearn.metrics import roc_auc_score

def main():
    print("Hello world!")
    timbres, pitches = get_data(0, 100)
    np.savetxt('data/processed.txt', timbres)
    return 1

def train(model, inputs, labels):
    criterion=nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    batches=batch_data(inputs)
    for i in range(batches.length):

        forwardProp=model.call()
        loss=criterion(forwardProp, labels)
        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
    model.call()

def batch_data(allInputs):
    return allInputs
    #Here we re    

def test(model, inputs, labels):
    one_hot_labels=one_hot(labels, num_classes=50)
    probs=model.call(inputs)
    return roc_auc_score(one_hot_labels, probs, multi_class='ovr')


    # Create an array that is of size 2xnumber of labels
    # Run call on each of the inputs and use the argmax of the logits to get the most probable label. 
    # Let n=argmax(logits of inputs) and m be the actual label.  If  is the correct label, add one to 0 x m.  If not, add 1 to 1 x m
    # 
    #Here we do the AUC-ROC calculation.
    visualizer = ROCAUC(model, classes=[list_of_labels])
    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()  
    model.call()

if __name__ == "__main__":
    main()