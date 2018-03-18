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
    
    for ii in range(iterations):
        batch = np.random.choice(train_features.index, size=128)
        X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']
          
        network.train(X, y)
    
    return MSE(network.run(val_features).T, val_targets['cnt'].values)


def check_losses(iterations, learning_rate, hidden_nodes, pool, repeat=1):
    params = (iterations, learning_rate, hidden_nodes)
    return pool.map(get_loss, [params]*repeat)

if __name__ == '__main__':
  pool = Pool(processes=4)
  
  start = time.time()
  total_iterations = 0

  for iterations in range(500, 3000, 500):
    node_range = np.arange(4, 11, 1)
    lr_range = np.linspace(0.4, 2.4, 21)
    trials = 24
    print('for {} trials of {} iterations:'.format(trials, iterations))

    validations = []
    run_times = []

    print()
    print('arithmetic means: ')
    print('lr\t' + '\t'.join('{:4}'.format(n) for n in node_range))

    for learning_rate in lr_range:
      print('{:0.3f}'.format(learning_rate), end='\t')
      row_timings = []
      row_validations = []
      for hidden_nodes in node_range:
        total_iterations += iterations * trials
        start_run = time.time()
        validate = check_losses(iterations, learning_rate, hidden_nodes, pool, trials)
        row_validations.append(validate)
        row_timings.append(time.time() - start_run)
        print('{:0.4f}'.format(np.mean(validate)), end='\t')
        sys.stdout.flush()
      run_times.append(row_timings)
      validations.append(row_validations)
      print()
        #stats_of(validate, '{} nodes at {:0.2f}'.format(hidden_nodes, learning_rate))

    print()
    print('standard deviations: ')
    print('lr\t' + '\t'.join('{:4}'.format(n) for n in node_range))
    for learning_rate, vs_row in zip(lr_range, validations):
      print('{:0.3f}'.format(learning_rate), end='\t')
      for hidden_nodes, vs in zip(node_range, vs_row):
        print('{:0.4f}'.format(np.std(vs)), end='\t')
      print()

    print()
    print('medians: ')
    print('lr\t' + '\t'.join('{:4}'.format(n) for n in node_range))
    for learning_rate, vs_row in zip(lr_range, validations):
      print('{:0.3f}'.format(learning_rate), end='\t')
      for hidden_nodes, vs in zip(node_range, vs_row):
        print('{:0.4f}'.format(np.median(vs)), end='\t')
      print()

    print()
    print('seconds / 1000 iterations: ')
    print('\t'.join('{:4}'.format(n) for n in node_range))
    for rt in np.mean(np.array(run_times), axis=0):
      print('{:0.4f}'.format(1000 * rt / (trials * iterations)), end='\t')
    print()

  print()
  duration = time.time() - start
  print('{:.1f}m total, {:0.2f}s per 1000 iterations'.format(duration/60, 1000 * duration / total_iterations))
