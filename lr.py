import sys
import numpy as np

class LogisticRegression:
  """Code author: Ting-Sheng Lin (tingshel@andrew.cmu.edu)"""
  def __init__(self, dictionary_file):
    self.theta = { 'b': 0 } # bias term
    self.dictionary = {}
    self.step_rate = 0.1
    self.metrix = {}
    f_in = open(dictionary_file, 'r')
    i = 0
    while True:
      row = f_in.readline()
      if not row:
        break
      word, index = row.split()
      self.dictionary[word] = int(index)
      self.theta[i] = 0
      i += 1
    self.theta_len = i - 1
    
    

  def train(self, train_file, epoch, valid_file):
    X, y = [], []
    f_in = open(train_file, 'r')
    while True:
      row = f_in.readline()
      if not row:
        break
      features = row.split()
      label, features = features[0], features[1:]
      x = { 'b': 1 } # bias term
      for feature in features:
        i, val = feature.split(':')
        x[int(i)] = int(val)
      X.append(x)
      y.append(int(label))
    
    # for 1.4.1
    valid_X, valid_y = [], []
    f_in = open(valid_file, 'r')
    while True:
      row = f_in.readline()
      if not row:
        break
      features = row.split()
      label, features = features[0], features[1:]
      x = { 'b': 1 } # bias term
      for feature in features:
        i, val = feature.split(':')
        x[int(i)] = int(val)
      valid_X.append(x)
      valid_y.append(int(label))


    # full SGD
    for e in range(epoch):
      # 1 epoch
      for i in range(len(X)):
        # 1 SGD
        thetaTxi = 0 # theta transpose xi
        for j in X[i]:
          thetaTxi += self.theta[j] * X[i][j]
        e_thetaTxi = np.exp(thetaTxi)

        for j, xji in X[i].items(): # update all theta for ith datapoint
          gradient = xji * (y[i] - (e_thetaTxi / (1 + e_thetaTxi)))
          self.theta[j] = self.theta[j] + self.step_rate * gradient
      # for 1.4.1
      l = 0
      for i in range(len(X)):
        thetaTxi = 0
        for j in X[i]:
          thetaTxi += self.theta[j] * X[i][j]
        e_thetaTxi = np.exp(thetaTxi)
        
        l = l + np.log(1 + e_thetaTxi) - y[i] * thetaTxi
      l /= len(X)

      valid_l = 0
      for i in range(len(valid_X)):
        thetaTxi = 0
        for j in valid_X[i]:
          thetaTxi += self.theta[j] * valid_X[i][j]
        e_thetaTxi = np.exp(thetaTxi)
        
        valid_l = valid_l + np.log(1 + e_thetaTxi) - valid_y[i] * thetaTxi
      valid_l /= len(valid_X)
      print(f'{l:.4f}\t{valid_l:.4f}')
      # plot.append(l)
    # f_out = open(name,"w+")
    # f_out.write('\n'.join(str(plot)))

      
          


  def test(self, test_file, predict_file, error_name):
    X, y = [], []
    f_in = open(test_file, 'r')
    while True:
      row = f_in.readline()
      if not row:
        break
      features = row.split()
      label, features = features[0], features[1:]
      x = { 'b': 1 } # bias term
      for feature in features:
        i, val = feature.split(':')
        x[int(i)] = int(val)
      X.append(x)
      y.append(int(label))

    predictions_str = ''
    error_count = 0
    for i in range(len(X)):
      thetaTxi = 0
      for j, val in X[i].items():
        thetaTxi = thetaTxi + self.theta[j] * val
      prediction = 1 if thetaTxi >= 0 else 0
      if prediction != y[i]:
        error_count += 1
      predictions_str += f'{prediction}\n'
    if error_name:
      self.metrix[error_name] = error_count / len(X)

    f_out = open(predict_file,"w+")
    f_out.write(predictions_str)


  def metrics(self, metrics_file):
    out_str = ''
    for name, error_rate in self.metrix.items():
      out_str += f'error({name}): {error_rate:.6f}\n'
    f_out = open(metrics_file,"w+")
    f_out.write(out_str)
      



if __name__ == '__main__':
  formatted_train_input = sys.argv[1]
  formatted_validation_input = sys.argv[2]
  formatted_test_input = sys.argv[3]
  dict_input = sys.argv[4]
  train_out = sys.argv[5]
  test_out = sys.argv[6]
  metrics_out = sys.argv[7]
  num_epoch = int(sys.argv[8])
  
  lg = LogisticRegression(dict_input)
  lg.train(formatted_train_input, num_epoch, formatted_validation_input)
  # lg.train(formatted_validation_input, num_epoch, 'valid_L.txt')
  lg.test(formatted_train_input, train_out, 'train')
  lg.test(formatted_test_input, test_out, 'test')
  lg.metrics(metrics_out)
