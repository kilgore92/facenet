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
import argparse
import facenet
import align.detect_face
from compare import *
import pandas as pd

n_images = 1000
image_size = 160

def create_image_list(image_paths):
    images = [facenet.prewhiten(misc.imresize(misc.imread(path,mode='RGB'),(image_size,image_size),interp='bilinear')) for path in image_paths]
    return np.stack(images)


def compare_inpaintings(root_dir,idx,sess,images_placeholder,embeddings,phase_train_placeholder):
    """
    Computes distances of in-paintings of a single original image (by different models) to the original image

    """

    image_dir = os.path.join(root_dir,str(idx))
    original_image_path =  os.path.join(image_dir,'original.jpg')
    gen_images_dir = os.path.join(image_dir,'gen')
    image_paths = []
    image_paths.append(os.path.join(image_dir,'original.jpg'))
    generated_image_paths = [os.path.join(gen_images_dir,f) for f in os.listdir(gen_images_dir) if os.path.isfile(os.path.join(gen_images_dir, f))]
    for path in generated_image_paths:
        image_paths.append(path)

    images = create_image_list(image_paths)

    # Run forward pass to calculate embeddings
    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    emb = sess.run(embeddings, feed_dict=feed_dict)

    nrof_images = len(image_paths)


    # Print distance matrix
    print('Distances w.r.t. original : {}'.format(original_image_path))
    dist_list = []
    dist_list.append(original_image_path) # Add path for DB indexing
    for i in range(1,nrof_images):
        model_name = image_paths[i].split('/')[-1].split('.')[0]
        dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[i,:]))))
        dist_list.append(dist)
        print('{} :: {}'.format(model_name.upper(),dist))
    return dist_list

def create_database(root_dir,model=None):
    db = []

    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = "1"
        with tf.Session(config = config) as sess:
            # Load the model
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            for idx in range(n_images):
                db.append(compare_inpaintings(root_dir = root_dir,
                                              idx=idx,
                                              sess = sess,
                                              embeddings=embeddings,
                                              images_placeholder=images_placeholder,
                                              phase_train_placeholder=phase_train_placeholder))

            columns = ['Original Image','DCGAN-CONS','WGAN-GP','DCGAN','WGAN','DCGAN-GP']
            df = pd.DataFrame(data = db,
                              columns = columns)
            df.to_csv('gan_distances.csv')


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_files', type=str,default='/home/ibhat/gans_compare/tf.gans-comparison/images_db/0', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
    return parser

if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()
    create_database(root_dir = args.image_files,model = args.model)

