import sys

if __name__ == '__main__':
  train_input = sys.argv[1]
  validation_input = sys.argv[2]
  test_input = sys.argv[3]
  dict_input = sys.argv[4]
  formatted_train_out = sys.argv[5]
  formatted_validation_out = sys.argv[6]
  formatted_test_out = sys.argv[7]
  feature_flag = int(sys.argv[8])

  f_in = open(dict_input, 'r')
  dictionary = {}
  while True:
    row = f_in.readline()
    if not row:
      break
    word, index = row.split()
    dictionary[word] = int(index)

  def format_data(in_file, out_file):
    f_in = open(in_file, 'r')
    f_out = open(out_file,"w+")
    total_str = ""
    while True:
      row = f_in.readline()
      if not row:
        break
      label, words_str = row.split('\t')
      words = words_str.split()
      features = {}
      for word in words:
        if word not in dictionary:
          continue
        if dictionary[word] not in features:
          features[dictionary[word]] = 1
        else:
          features[dictionary[word]] += 1

      total_str += f"{label}\t"
      if feature_flag == 1:
        for key in features:
          total_str += f"{key}:1\t"
      elif feature_flag == 2:
        for key, val in features.items():
          if val < 4:
            total_str += f"{key}:1\t"
      total_str = total_str[:-1]
      total_str += "\n"
    f_out.write(total_str)


  format_data(train_input, formatted_train_out)
  format_data(validation_input, formatted_validation_out)
  format_data(test_input, formatted_test_out)
