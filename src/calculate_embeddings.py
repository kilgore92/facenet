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
from compare import *
import math
import pickle

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

def compute_embedding(sess,images_placeholder,phase_train_placeholder,embedding_compute_node,image_batch):
    """
    Computes embeddings for an image batch

    """
    # Run forward pass to calculate embeddings
    feed_dict = { images_placeholder: image_batch, phase_train_placeholder:False }
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
        images_dir = os.path.join(args.images_dir,'{}'.format(args.gan.lower()),'celeba')
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
        fname = os.path.join(save_path,'{}_emb_dict.pkl'.format(args.gan.lower()))
    else:
        fname = os.path.join(save_path,'{}_emb_dict.pkl'.format(args.src.lower()))

    with tf.Graph().as_default():

        config = tf.ConfigProto()

        with tf.Session(config = config) as sess:
            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            for batch_idx in range(num_batches):
                print('Calculating embeddings for batch {}/{} of images'.format(batch_idx,num_batches))
                sys.stdout.flush()
                image_batch = image_paths[batch_size*batch_idx:min(batch_size*(batch_idx+1),num_images)]
                images = load_and_align_data(image_batch,args.image_size, args.margin, args.gpu_memory_fraction,use_cnn=False)
                emb = compute_embedding(sess = sess,
                                        images_placeholder = images_placeholder,
                                        phase_train_placeholder = phase_train_placeholder,
                                        embedding_compute_node = embeddings,
                                        image_batch = images)

                # Save to dict
                for path,idx in zip(image_batch,range(len(image_batch))):
                    embedding_dict[path] = emb[idx,:]

                # Save dict to disk for every batch
                with open(fname,'wb') as f:
                    pickle.dump(embedding_dict,f)


def create_nn_emb_dict(args):

    """
    Creates a distance dict of training images for each test image
    Used for the nearest neighbours experiment

    """

    test_image_paths = create_test_image_paths(args.test_images_dir)
    training_image_paths = create_train_image_paths(args.training_images_dir)

    num_test_images = len(test_image_paths)
    num_train_images = len(training_image_paths)

    batch_size = 512

    num_batches_train = math.ceil(num_train_images/batch_size)
    num_batches_test = math.ceil(num_test_images/batch_size)

    with tf.Graph().as_default():

        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = "0"

        with tf.Session(config = config) as sess:
            # Load the model
            facenet.load_model(args.model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            #Compute all embeddings for test images
            test_image_embeddings = {} # Indexed by path to file
            for batch_idx in range(num_batches_test):
                print('Calculating embeddings for batch {} of test images'.format(batch_idx))
                test_image_batch = test_image_paths[batch_size*batch_idx:min(batch_size*(batch_idx+1),num_test_images)]
                images = load_and_align_data(test_image_batch,args.image_size, args.margin, args.gpu_memory_fraction,device_id = "0")
                emb = compute_embedding(sess = sess,
                                        images_placeholder = images_placeholder,
                                        phase_train_placeholder = phase_train_placeholder,
                                        embedding_compute_node = embeddings,
                                        image_batch = images)
                for path,idx in zip(test_image_batch,range(len(test_image_batch))):
                    test_image_embeddings[path] = emb[idx,:]

            print('Embeddings calculated for {} test images'.format(len(test_image_embeddings)))

            # Compute embeddings for training images
            # Store the differences between train and test image embeddings in the "distances" dictionary
            distances = {} # {test_image_path : {training_image_path:distance}}

            #Init inner dictionary
            for test_image_path in test_image_paths:
                distances[test_image_path] = {}

            for batch_idx in range(num_batches_train):
                print('Calculating embeddings for batch {} of train images'.format(batch_idx))
                training_image_batch = training_image_paths[batch_size*batch_idx:min(batch_size*(batch_idx+1),num_train_images)]
                images = load_and_align_data(training_image_batch,args.image_size, args.margin, args.gpu_memory_fraction)
                emb = compute_embedding(sess = sess,
                                        images_placeholder = images_placeholder,
                                        phase_train_placeholder = phase_train_placeholder,
                                        embedding_compute_node = embeddings,
                                        image_batch = images)
                for test_image_path in test_image_embeddings:
                    emb_test = test_image_embeddings[test_image_path]
                    for train_image_path,idx in zip(training_image_batch,range(len(training_image_batch))):
                        train_image_emb = emb[idx,:]
                        dist = np.sqrt(np.sum(np.square(np.subtract(emb_test,train_image_emb))))
                        distances[test_image_path][train_image_path] = dist
            #Save the dictionary of differences
            save_dict('distance_dict.pkl',distances)

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
    return parser.parse_args(argv)

if __name__ == '__main__':
    create_embeddings(parse_arguments(sys.argv[1:]))

