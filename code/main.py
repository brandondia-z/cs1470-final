import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from yellowbrick.classifier import ROCAUC
from preprocess import get_data
from model import Model
from sklearn import metrics
import time
import sys
from preprocessed_parsed import get_parsed
import pickle

def visualize_loss(losses):
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  

def visualize_acc(accuracies): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(accuracies))]
    plt.plot(x, accuracies)
    plt.title('Accuracy per batch')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.show()  

def train(model, inputs, labels, device='cpu', loss_array=[]):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    batch_inputs = list(batch_data(inputs, 100))
    batch_labels= list(batch_labs(labels, 100))
    
    loss_sum = 0
    for i in range(len(batch_inputs)):
        batchin = batch_inputs[i]
        batchlab = batch_labels[i]
        inp = torch.from_numpy(batchin)
        inp = inp.type(torch.FloatTensor).to(device)
        lab = torch.FloatTensor(batchlab).to(device)
        one_hot= torch.nn.functional.one_hot(lab.to(torch.int64), num_classes=50)

        predictions = model.call(inp)
        loss = criterion(predictions.to(torch.float32), one_hot.to(torch.float32))
        loss_sum += loss

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum/=len(batch_inputs)
        loss_array.append(loss_sum)
    
def batch_data(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, l.shape[0], n))

def batch_labs(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))  

def test(model, inputs, labels, list_of_labels, device='cpu'):
    # Create an array that is of size 2xnumber of labels
    # Run call on each of the inputs and use the argmax of the logits to get the most probable label. 
    # Let n=argmax(logits of inputs) and m be the actual label.  If  is the correct label, add one to 0 x m.  If not, add 1 to 1 x m
    # 
    #Here we do the AUC-ROC calculation.
    inp = torch.from_numpy(inputs)
    inp = inp.type(torch.FloatTensor).to(device)
    lab = torch.FloatTensor(labels).to(device)
    
    batch_inputs = list(batch_data(inputs, 100))
    batch_labels= list(batch_labs(labels, 100))

    total = 0
    for i in range(len(batch_inputs)):
        batchin = batch_inputs[i]
        batchlab = batch_labels[i]
        inp = torch.from_numpy(batchin)
        inp = inp.type(torch.FloatTensor).to(device)
        lab = torch.FloatTensor(batchlab)

        predictions = model.call(inp)
        thing = torch.argmax(predictions, dim=1)
        corr = (np.array(thing.cpu()) == batchlab)
        res = corr.sum() / len(corr)
        total += res
    return total/len(batch_inputs)

def sort_result(tags, predictions):
  zipped = zip(tags, predictions)
  sorted_tags = sorted(zipped, key=lambda x: x[1], reverse=True)
  tag_list = []
  for tag, score in sorted_tags:
     tag_list += [tag,score]
  return tag_list

def main():
    tags = ['rock', 'pop', 'alternative', 'indie', 'electronic',
                'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
                'beautiful', 'metal', 'chillout', 'male vocalists',
                'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica',
                '80s', 'folk', '90s', 'chill', 'instrumental', 'punk',
                'oldies', 'blues', 'hard rock', 'ambient', 'acoustic',
                'experimental', 'female vocalist', 'guitar', 'Hip-Hop',
                '70s', 'party', 'country', 'easy listening',
                'sexy', 'catchy', 'funk', 'electro', 'heavy metal',
                'Progressive rock', '60s', 'rnb', 'indie pop',
                'sad', 'House', 'happy']

    model = Model()

    if sys.argv[len(sys.argv)-1] != "BIG":
        train_inputs, train_labels = get_data(0, 7000)
        test_inputs, test_labels = get_data(7000,10000)

        start = time.time()
        training = np.reshape(train_inputs, (-1, 1, 200, 24))
        testing = np.reshape(test_inputs, (-1, 1, 200, 24))
        predicted = train(model=model, inputs=training, labels=train_labels)
        print ("Training is done. It took %d seconds." % (time.time()-start))
        results = test(model=model, inputs=testing, labels=test_labels, list_of_labels=tags)
    
    elif sys.argv[len(sys.argv)-1] == "BIG":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        epochs = 20
        accuracy_array = []
        loss_array = []
        for i in range(epochs):
            num_megabatches = 32

            for i in range(num_megabatches):
                inputs, labels = get_parsed(i)
                train_inputs = inputs[0:7000]
                train_labels = labels[0:7000]
                test_inputs = inputs[7000:10000]
                test_labels = labels[7000:10000]

                training = np.reshape(train_inputs, (-1, 1, 200, 24))
                testing = np.reshape(test_inputs, (-1, 1, 200, 24))
                
                training = np.reshape(train_inputs, (-1, 1, 200, 24))
                testing = np.reshape(test_inputs, (-1, 1, 200, 24))
                start = time.time()
                predicted = train(model=model, inputs=training, labels=train_labels, device=device, loss_array=loss_array) ##TODO: inputs 3161,200,24

            results = test(model=model, inputs=testing, labels=test_labels, list_of_labels=tags, device=device)
            accuracy_array.append(results)
        
        with open (f'results/accuracy', 'wb') as fp:
            pickle.dump(accuracy_array, fp)
        with open (f'results/loss', 'wb') as fp:
            pickle.dump(loss_array, fp)
        
        # Not sure if you can use both of these on the same run, but if you want to comment one out and then run again with the other that works too
        visualize_loss(losses=loss_array)
        visualize_acc(accuracies=accuracy_array)
        #Also maybe we could play around with a visualize results function ?? Similar to the one in hw2
            
    return

if __name__ == "__main__":
    main()
