#!/usr/bin/anaconda3/bin/python3
import numpy as np
import tensorflow as tf
import os,sys
sys.path.append(os.getcwd())
from tensorflow.examples.tutorials.mnist import input_data
import pickle

MODEL_DIRECTORY = "./model"

def load_mnist_model(model_dir,sess,input_map=None):
    """
    Inputs : model_dir : Directory where model checkpt and meta files are stored
             sess : TF session where the model needs into which the model will be loaded

    Returns : Restored model, mean of MNIST dataset

    """

    model_path = os.path.expanduser(model_dir)
    print('Model directory is {}'.format(model_path))
    files = os.listdir(model_path)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) > 1:
        print('More than one meta file found, exiting')
        return

    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_path)

    sys.stdout.flush()

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file), input_map=input_map)
        saver.restore(sess, os.path.join(model_path, ckpt_file))
        print('Model loaded successfully')
    else:
        print('Bad checkpoint state')

    sys.stdout.flush()


def plot_embeddings(images,labels):
    """
    Inputs:
        images: MNIST images
        labels: Image lables (used to color embeddings)

    Plot MNIST 2-D embeddings to illustrate the effect of
    center loss

    """
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            load_mnist_model(model_dir=MODEL_DIRECTORY,sess=sess)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("images:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            feed_dict = {images_placeholder:images}
            emb = sess.run(embeddings, feed_dict=feed_dict)

    print('Embeddings generated')

    distinct_labels = set(list(labels))
    plot_dict = {}

    for label in distinct_labels:
        plot_dict[label] = []
        #Get indices for the current label
        l_idxs = [idx for idx,l in enumerate(list(labels)) if l==label]
        for idx in l_idxs:
            plot_dict[label].append(emb[idx,:])
    #Save the dict to plot offline
    with open('mnist_plot_dict.pkl','wb') as f:
        pickle.dump(plot_dict,f)



def feed_images(num_images=50):
    """
    Inputs:
        num_images : Number of images to feed
    Returns:
        Array of normalized,flattened images

    """
    mnist = input_data.read_data_sets('./mnist')
    image_mean = np.mean(mnist.train.images, axis=0)
    batch = mnist.train.next_batch(num_images)
    batch_images = normalize_batch(batch[0],image_mean=image_mean)
    batch_labels = batch[1]
    return batch_images,batch_labels

if __name__ == '__main__':
    images,labels = feed_images()
    plot_embeddings(images=images,labels=labels)


