python3

import sys
import tensorflow as tf

def predictint(imvalue):

    
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "model.ckpt")
        #print ("Model restored.")
   
        prediction=tf.argmax(y,1)
        return prediction.eval(feed_dict={x: [imvalue]}, session=sess)


def imageprepare(argv):
    
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
