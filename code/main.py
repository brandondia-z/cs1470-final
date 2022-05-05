import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from yellowbrick.classifier import ROCAUC
from preprocess import get_data
from model import Model
import time

def train(model, inputs, labels):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    batches = batch_data(inputs)

    for i in range(len(batches)):
        predictions = model(inputs)  # TODO: Make sure we are passing in the batched inputs
        loss = criterion(predictions, labels)

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

def batch_data(allInputs):
    return allInputs
    #Here we re    

def test(model, inputs, labels, list_of_labels):
    # Create an array that is of size 2xnumber of labels
    # Run call on each of the inputs and use the argmax of the logits to get the most probable label. 
    # Let n=argmax(logits of inputs) and m be the actual label.  If  is the correct label, add one to 0 x m.  If not, add 1 to 1 x m
    # 
    #Here we do the AUC-ROC calculation.
    visualizer = ROCAUC(model, classes=[list_of_labels])
    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()  
    model(inputs)

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
    
    inputs, labels = get_data(0, 10000)
    model = Model(inputs.shape) ##TODO
    model.summary()

    start = time.time()
    predicted = train(model=model, inputs=inputs, labels=labels) ##TODO: inputs 200x24
    print ("Training is done. It took %d seconds." % (time.time()-start)) 
    results = test(model=model, inputs=inputs, labels=labels, list_of_labels=tags)
   

    for song_idx, audio_path in enumerate(audio_paths):
      sorted = sort_result(tags, predicted[song_idx:].tolist())
      print(audio_path)
      print(sorted[:10])

    return

if __name__ == "__main__":
    main()
