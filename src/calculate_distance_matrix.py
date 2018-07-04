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
sys.path.append('/home/TUE/s162156/fid/TTUR') # For the FID/ImageNet Embeddings
import argparse
import facenet
import align.detect_face
from compare import *
import math
import pickle
import fid
from calculate_embeddings import *
from scipy.spatial.distance import cosine

n_images = 1000

def create_nn_emb_dict(args):

    """
    Creates a distance dict of training images for each test image
    Used for the nearest neighbours experiment

    """

    test_image_paths = create_test_image_paths(args.test_images)
    training_image_paths = create_train_image_paths(args.train_images)

    num_test_images = len(test_image_paths)
    num_train_images = len(training_image_paths)

    batch_size = 512

    num_batches_train = math.ceil(num_train_images/batch_size)
    num_batches_test = math.ceil(num_test_images/batch_size)

    print('Num batches test : {}'.format(num_batches_test))
    print('Num batches train : {}'.format(num_batches_train))

    sys.stdout.flush()

    with tf.Graph().as_default():

        config = tf.ConfigProto()

        with tf.Session(config = config) as sess:
            # Load the model
            if args.imagenet == False:
                facenet.load_model(args.model)
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            else:
                fid.create_inception_graph(args.model)

            #Compute all embeddings for test images
            test_image_embeddings = {} # Indexed by path to file

            for batch_idx in range(num_batches_test):
                print('Calculating embeddings for batch {} of test images'.format(batch_idx))
                test_image_batch = test_image_paths[batch_size*batch_idx:min(batch_size*(batch_idx+1),num_test_images)]
                if args.imagenet == False:
                    images = load_and_align_data(test_image_batch,args.image_size, args.margin, args.gpu_memory_fraction,device_id = "0")
                    emb = compute_embedding(sess = sess,
                                            images_placeholder = images_placeholder,
                                            phase_train_placeholder = phase_train_placeholder,
                                            embedding_compute_node = embeddings,
                                            image_batch = images)
                else:
                    images = load_and_align_data(test_image_batch,64,args.margin,args.gpu_memory_fraction,use_cnn=False)
                    emb = fid.get_activations(images=images,sess=sess,batch_size=1)

                for path,idx in zip(test_image_batch,range(len(test_image_batch))):
                    test_image_embeddings[path] = emb[idx,:]

            print('Embeddings calculated for {} test images'.format(len(test_image_embeddings)))
            sys.stdout.flush()

            # Compute embeddings for training images
            # Store the differences between train and test image embeddings in the "distances" dictionary
            distances = {} # {test_image_path : {training_image_path:distance}}

            #Init inner dictionary
            for test_image_path in test_image_paths:
                distances[test_image_path] = {}

            for batch_idx in range(num_batches_train):
                print('Calculating embeddings for batch {} of train images'.format(batch_idx))
                sys.stdout.flush()
                training_image_batch = training_image_paths[batch_size*batch_idx:min(batch_size*(batch_idx+1),num_train_images)]
                if args.imagenet == False:
                    images = load_and_align_data(training_image_batch,args.image_size, args.margin, args.gpu_memory_fraction)
                    emb = compute_embedding(sess = sess,
                                            images_placeholder = images_placeholder,
                                            phase_train_placeholder = phase_train_placeholder,
                                            embedding_compute_node = embeddings,
                                            image_batch = images)
                else:
                    images = load_and_align_data(training_image_batch,64,args.margin,args.gpu_memory_fraction,use_cnn=False)
                    emb_train = fid.get_activations(images=images,sess=sess,batch_size=1)


                for test_image_path in test_image_embeddings:
                    emb_test = test_image_embeddings[test_image_path]
                    norm_emb_test = np.divide(emb_test,np.linalg.norm(emb_test))
                    for train_image_path,idx in zip(training_image_batch,range(len(training_image_batch))):
                        train_image_emb = emb_train[idx,:]
                        norm_emb_train = np.divide(train_image_emb,np.linalg.norm(train_image_emb))
                        dist = cosine(norm_emb_test,norm_emb_train)
                        distances[test_image_path][train_image_path] = dist

            #Save the dictionary of differences
            if args.imagenet == False:
                save_dict('distance_dict.pkl',distances)
            else:
                save_dict('distance_dict_imagenet.pkl',distances)

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
    parser.add_argument('--test_images',type=str,default='/home/TUE/s162156/gans_compare/tf.gans-comparison/imagesdb')
    parser.add_argument('--train_images',type=str,default='/home/TUE/s162156/datasets/celebA/celebA')
    parser.add_argument('--imagenet',action='store_true',default=False)

    return parser.parse_args(argv)


if __name__ == '__main__':
    create_nn_emb_dict(parse_arguments(sys.argv[1:]))
