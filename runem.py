import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

# Save data for approximately the last 21 days 
test_data = data[-21*24:]

# Now remove the test data from the data set 
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]

from my_answers import NeuralNetwork
def MSE(y, Y):
    return np.mean((y-Y)**2)

import sys
import time
####################
### Set the hyperparameters in you myanswers.py file ###
####################

from multiprocessing import Pool

def get_loss(params):
    iterations, learning_rate, hidden_nodes = params
    N_i = train_features.shape[1]
    network = NeuralNetwork(N_i, hidden_nodes, 1, learning_rate)
    loss_values = []
    
    for ii in range(iterations + 1):
        batch = np.random.choice(train_features.index, size=128)
        X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']
          
        network.train(X, y)
        if ii % 100 == 0 and ii > 0:
          loss_values.append(MSE(network.run(val_features).T, val_targets['cnt'].values))

    return loss_values


def check_losses(iterations, learning_rate, hidden_nodes, pool, repeat=1):
    params = (iterations, learning_rate, hidden_nodes)
    return pool.map(get_loss, [params]*repeat)

if __name__ == '__main__':
  pool = Pool(processes=4)
  
  start = time.time()
  total_iterations = 0

  iterations = 3000
  node_range = np.arange(10, 13, 1)
  lr_range = np.linspace(0.7, 1.0, 16)
  trials = 16
  print('for {} trials of {} iterations:'.format(trials, iterations))

  print()
  print('final loss: ')
  print('lr\t' + '\t'.join('{:4}'.format(n) for n in node_range))

  with open('runs.csv', 'a') as outfile:
    for learning_rate in lr_range:
      print('{:0.3f}'.format(learning_rate), end='\t')
      for hidden_nodes in node_range:
        total_iterations += iterations * trials
        losses = check_losses(iterations, learning_rate, hidden_nodes, pool, trials)
        title = '{}, {}, '.format(learning_rate, hidden_nodes)
        for trial in losses:
          outfile.write(title + ', '.join(str(l) for l in trial) + '\n')
        avg = np.mean(np.array(losses), axis=0)[-1]
        print('{:0.4f}'.format(avg), end='\t')
        sys.stdout.flush()
      print('{:0.2f}m'.format((time.time() - start)/60))

  print()
  duration = time.time() - start
  print('{:.1f}m total, {:0.2f}s per 1000 iterations'.format(duration/60, 1000 * duration / total_iterations))
