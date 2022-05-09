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

def train(model, inputs, labels):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    batch_inputs = list(batch_data(inputs, 100))
    batch_labels= list(batch_labs(labels, 100))
    
    for i in range(len(batch_inputs)):
        batchin = batch_inputs[i]
        batchlab = batch_labels[i]
        inp = torch.from_numpy(batchin)
        inp = inp.type(torch.FloatTensor)
        lab = torch.FloatTensor(batchlab)
        # breakpoint()
        one_hot= torch.nn.functional.one_hot(lab.to(torch.int64), num_classes=50)

        predictions = model.call(inp)  # TODO: Make sure we are passing in the batched inputs
        loss = criterion(predictions.to(torch.float32), one_hot.to(torch.float32))

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
def batch_data(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, l.shape[0], n))

def batch_labs(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))  

def test(model, inputs, labels, list_of_labels):
    # Create an array that is of size 2xnumber of labels
    # Run call on each of the inputs and use the argmax of the logits to get the most probable label. 
    # Let n=argmax(logits of inputs) and m be the actual label.  If  is the correct label, add one to 0 x m.  If not, add 1 to 1 x m
    # 
    #Here we do the AUC-ROC calculation.
    inp = torch.from_numpy(inputs)
    inp = inp.type(torch.FloatTensor)
    lab = torch.FloatTensor(labels)
    
    results = model.call(inp)
    score= metrics.roc_auc_score(lab, results, multi_class='ovr')
    print(score)

def sort_result(tags, predictions):
  zipped = zip(tags, predictions)
  sorted_tags = sorted(zipped, key=lambda x: x[1], reverse=True)
  tag_list = []
  for tag, score in sorted_tags:
     tag_list += [tag,score]
  return tag_list

def main():
    if sys.argv[len(sys.argv)-1] != "BIG":

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
        
        train_inputs, train_labels = get_data(0, 7000)
        test_inputs, test_labels = get_data(7000,10000)
        model = Model() ##TODO
        # model.summary()

        start = time.time()
        training = np.reshape(train_inputs, (-1, 1, 200, 24))
        testing = np.reshape(test_inputs, (-1, 1, 200, 24))
        predicted = train(model=model, inputs=training, labels=train_labels) ##TODO: inputs 3161,200,24
        print ("Training is done. It took %d seconds." % (time.time()-start))
        results = test(model=model, inputs=testing, labels=test_labels, list_of_labels=tags)
    

    else:
        print("hello world")

    return

if __name__ == "__main__":
    main()
