#!/usr/bin/anaconda3/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Performs face alignment and calculates L2 distance between the embeddings of images."""

""" Modified for comparing GAN images -- Ishaan """

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
from plot_mnist_embeddings import load_mnist_model
import argparse
import facenet
import align.detect_face
from compare import *
import pandas as pd
from scipy.spatial.distance import cosine
from tensorflow.examples.tutorials.mnist import input_data
import shutil

n_images = 1000

image_size = 160
mnist_image_size = 28

models = ['dcgan','dcgan-gp','dragan','dcgan-cons','wgan','wgan-gp','dragan_bn','dcgan_sim']



def compare_inpaintings(root_dir,idx,sess,images_placeholder,embeddings,phase_train_placeholder,dataset='celeba',image_mean=None):
    """
    Computes distances of in-paintings of a single original image (by different models) to the original image

    """

    image_dir = os.path.join(root_dir,str(idx))
    original_image_path =  os.path.join(image_dir,'original.jpg')
    gen_images_dir = os.path.join(image_dir,'gen')
    image_paths = []
    image_paths.append(os.path.join(image_dir,'original.jpg'))

    for model in models:
        path = os.path.join(gen_images_dir,'{}.jpg'.format(model.lower()))
        image_paths.append(path)

    # From the paths, read + whiten + resize the images to be fed to model
    images = create_image_list(image_paths,dataset=dataset,image_mean=image_mean)

    # Run forward pass to calculate embeddings
    if dataset == 'celeba':
        feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    else:
        feed_dict = {images_placeholder : images}
    emb = sess.run(embeddings, feed_dict=feed_dict)

    nrof_images = len(image_paths)


    # Print distance matrix
    print('Distances w.r.t. original : {}'.format(original_image_path))
    dist_list = []
    dist_list.append(original_image_path) # Add path for DB indexing
    for i in range(1,nrof_images):
        model_name = image_paths[i].split('/')[-1].split('.')[0]
        dist = cosine(emb[0,:],emb[i,:])
        dist_list.append(dist)
        print('{} :: {}'.format(model_name.upper(),dist))
    return dist_list

def create_database(root_dir,model=None,dataset='celeba'):
    db = []

    outDir = os.path.join('logs',dataset)

    if os.path.exists(outDir):
        shutil.rmtree(outDir)
    os.makedirs(outDir)

    with tf.Graph().as_default():
        config = tf.ConfigProto()
        with tf.Session(config = config) as sess:
            # Load the model
            if dataset == 'celeba':
                facenet.load_model(model)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                image_mean = None
            else:
                load_mnist_model(model_dir=model,sess=sess)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("images:0")
                mnist = input_data.read_data_sets('./mnist')
                image_mean = np.mean(mnist.train.images, axis=0)

            # Get input and output tensors
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            if dataset == 'celeba':
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            else:
                phase_train_placeholder = None

            for idx in range(n_images):
                db.append(compare_inpaintings(root_dir = root_dir,
                                              idx=idx,
                                              sess = sess,
                                              embeddings=embeddings,
                                              images_placeholder=images_placeholder,
                                              phase_train_placeholder=phase_train_placeholder,
                                              dataset=dataset,
                                              image_mean=image_mean))

                sys.stdout.flush()

            columns = []
            columns.append('Original Image')
            for model in models:
                columns.append(model.upper())

            df = pd.DataFrame(data = db,
                              columns = columns)
            df.to_csv(os.path.join(outDir,'gan_distances.csv'.format(dataset)))


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_files', type=str,default='/home/ibhat/gans_compare/tf.gans-comparison/images_db/', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
    parser.add_argument('--dataset',type=str,default='celeba',help='Allowed options : celeba/mnist')
    return parser

if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()
    create_database(root_dir = os.path.join(args.image_files,args.dataset), model = args.model, dataset=args.dataset)

