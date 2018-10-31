#!/usr/bin/anaconda3/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Using the distance computed by the Inception ResNet model, find most similar faces in the training set
for images in the 'held-out' set. Used to check if GANs are "memorizing" faces """


# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'src'))
import argparse
import facenet
import align.detect_face
from plot_mnist_embeddings import load_mnist_model
from compare import *
import math
import pickle
from tensorflow.examples.tutorials.mnist import input_data

n_images = 1000


def create_test_image_paths(root_dir):
    """
    To maintain indexing, provide the images db folder
    """
    test_image_paths = []
    for idx in range(n_images):
        test_image_paths.append(os.path.join(root_dir,str(idx),'original.jpg'))
    return test_image_paths

def create_train_image_paths(training_images_dir):
    training_image_paths = [os.path.join(training_images_dir,f) for f in os.listdir(training_images_dir) if os.path.isfile(os.path.join(training_images_dir,f))]
    return training_image_paths

def create_inpaint_image_paths(images_dir):
    """
    Provide the 'completions_stochastic_center' as the root_dir

    """
    image_paths = []
    for idx in range(1000):
        image_paths.append(os.path.join(images_dir,str(idx),'gz','gz_1400.jpg'))
    return image_paths

def compute_embedding(sess,images_placeholder,phase_train_placeholder,embedding_compute_node,image_batch,dataset='celeba'):
    """
    Computes embeddings for an image batch

    """
    # Run forward pass to calculate embeddings
    if dataset == 'celeba':
        feed_dict = { images_placeholder: image_batch, phase_train_placeholder:False }
    else:
        feed_dict = { images_placeholder: image_batch}

    emb = sess.run(embedding_compute_node, feed_dict=feed_dict)

    return emb


def create_embeddings(args):

    """
    Given a folder of images, generates a dictionary with
    image_path:embedding map

    """

    if args.src == 'test':
        image_paths = create_test_image_paths(args.images_dir)
    elif args.src == 'train':
        image_paths = create_train_image_paths(args.images_dir)
    else: # (src == inpaint)
        images_dir = os.path.join(args.images_dir,'{}'.format(args.gan.lower()),args.dataset.lower())
        image_paths = create_inpaint_image_paths(images_dir)

    num_images = len(image_paths)

    print('Images found in dir : {}'.format(num_images))

    batch_size = 512
    num_batches = math.ceil(num_images/batch_size)

    embedding_dict = {}

    save_path = os.path.join(os.getcwd(),'embeddings')

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    if args.src == 'inpaint':
        fname = os.path.join(save_path,'{}_{}_emb_dict.pkl'.format(args.gan.lower(),args.dataset.lower()))
    else:
        fname = os.path.join(save_path,'{}_{}_emb_dict.pkl'.format(args.src.lower(),args.dataset.lower()))

    with tf.Graph().as_default():

        config = tf.ConfigProto()

        with tf.Session(config = config) as sess:
            # Load the model
            if args.dataset == 'celeba':
                facenet.load_model(args.model)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            else:
                load_mnist_model(model_dir=args.model,sess=sess)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("images:0")
                phase_train_placeholder = None
                mnist = input_data.read_data_sets('./mnist')
                image_mean = np.mean(mnist.train.images, axis=0)

            # Get input and output tensors
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            for batch_idx in range(num_batches):
                print('Calculating embeddings for batch {}/{} of images'.format(batch_idx,num_batches))
                sys.stdout.flush()
                image_batch = image_paths[batch_size*batch_idx:min(batch_size*(batch_idx+1),num_images)]
                if args.dataset == 'celeba':
                    images = load_and_align_data(image_batch,args.image_size, args.margin, args.gpu_memory_fraction,use_cnn=False)
                else:
                    images = create_image_list(image_batch,dataset='mnist',image_mean=image_mean)

                emb = compute_embedding(sess = sess,
                                        images_placeholder = images_placeholder,
                                        phase_train_placeholder = phase_train_placeholder,
                                        embedding_compute_node = embeddings,
                                        image_batch = images,
                                        dataset=args.dataset)

                # Save to dict
                for path,idx in zip(image_batch,range(len(image_batch))):
                    embedding_dict[path] = emb[idx,:]

                # Save dict to disk for every batch
                with open(fname,'wb') as f:
                    pickle.dump(embedding_dict,f)


def save_dict(fname,diff_dict):
    with open(fname,'wb') as f:
        pickle.dump(diff_dict,f)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()


    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
    parser.add_argument('--images_dir',type=str,help='Path containing held out set of images',default=None)
    parser.add_argument('--src',type=str,help='Type of images to generate embeddings for',default=None)
    parser.add_argument('--gan',type=str,help='GAN that performed the inpainting',default=None)
    parser.add_argument('--dataset',type=str,help='celeba/mnist',default='celeba')
    return parser.parse_args(argv)

if __name__ == '__main__':
    create_embeddings(parse_arguments(sys.argv[1:]))

