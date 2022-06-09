python3

from __future__ 
from __future__ 
from __future__ 
print 
import tensorflow as tf
import numpy as np

# Similarly to the example in: https://www.tensorflow.org/tutorials/tflearn/
# we create a model and test on our own TSA Baggage Claims data.

# separated train and test files from MapR-FS
TRAIN = "/mapr/demo.mapr.com/user/mapr/claims_train.csv"
TEST = "/mapr/demo.mapr.com/user/mapr/claims_test.csv"
MODEL_DIR = "/mapr/demo.mapr.com/user/mapr/model"

# load the data sets
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TRAIN,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TEST,
    target_dtype=np.int,
    features_dtype=np.float32)
