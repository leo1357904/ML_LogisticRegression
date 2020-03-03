# ML_LogisticRegression
Logistic Regression for example movie prediction.

***Env: Python 3.6.9***

***feature.py***: use dict.txt to generate formatted_train.tsv and formatted_valid.tsv

<code>$ python feature.py train_data.tsv valid_data.tsv test_data.tsv dict.txt formatted_train.tsv formatted_valid.tsv formatted_test.tsv 1</code>

***lr.py***: use generated data to predict the movie according to the movie description.

<code>$ python lr.py formatted_train.tsv formatted_valid.tsv formatted_test.tsv dict.txt train_out.labels test_out.labels metrics_out.txt 60</code>
(number 60 is the epoches)
