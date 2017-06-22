from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from scipy import misc

def init_session_model():
    MODEL_DIR = '~/models/facenet/20170512-110547'
    seed = 666
    np.random.seed(seed=seed)
    print('Loading feature extraction model')
    facenet.load_model(MODEL_DIR)

def try_init_session():
    MODEL_DIR = '~/models/facenet/20170512-110547'
    # with tf.Graph().as_default():
    #     with tf.Session() as sess:
    g = tf.Graph()
    sess = tf.Session()
    seed = 666
    with g.as_default():
        with sess.as_default():
            np.random.seed(seed=seed)
            
            # dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            # for cls in dataset:
            #     assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            
                 
            # paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            # print('Number of classes: %d' % len(dataset))
            # print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(MODEL_DIR)
            
            # Get input and output tensors
            # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # embedding_size = embeddings.get_shape()[1]
            return g,sess 
            
def close_session(sess):
    sess.close()


def get_emb_array(sess,paths):
    IMG_SIZE = 160
        # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    # Run forward pass to calculate embeddings
    print('Calculating features for images')
    nrof_images = len(paths)
    # nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))
    images = facenet.load_data(paths,False,False,IMG_SIZE)
    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
    emb_array = sess.run(embeddings, feed_dict=feed_dict)
    # for i in range(nrof_batches_per_epoch):
    #     start_index = i*args.batch_size
    #     end_index = min((i+1)*args.batch_size, nrof_images)
    #     paths_batch = paths[start_index:end_index]
    #     images = facenet.load_data(paths_batch, False, False, args.image_size)
    #     feed_dict = { images_placeholder:images, phase_train_placeholder:False }
    #     emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
    return emb_array


def train(sess,data_dir,cam_classifier_file):
    dataset = facenet.get_dataset(data_dir)

    # Check that there are at least one training image per class
    for cls in dataset:
        assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            
         
    paths, labels = facenet.get_image_paths_and_labels(dataset)
    
    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))

    emb_array = get_emb_array(sess,paths)
    
    classifier_filename_exp = os.path.expanduser(cam_classifier_file)

    # Train classifier
    print('Training classifier')
    model = SVC(kernel='linear', probability=True)
    model.fit(emb_array, labels)

    # Create a list of class names
    class_names = [ cls.name.replace('_', ' ') for cls in dataset]

    # Saving classifier model
    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    print('Saved classifier model to file "%s"' % classifier_filename_exp)


# wrap all calls to classify with:
#
# with tf.Graph().as_default():
#       with tf.Session() as sess:
def classify(sess,img_path,classifier_file):
    #     # Classify images

    # print('Testing classifier')
    # args.authUser = args.authUser.replace('_',' ')
    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    nrof_images = 1
    IMG_SIZE = 160

    # Run forward pass to calculate embeddings
    print('Calculating features for images')
    # nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))
    images = np.zeros((nrof_images, IMG_SIZE, IMG_SIZE, 3)) # rgb -> 3 channels
    images[0,:,:,:] = misc.imread(img_path)
    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
    emb_array = sess.run(embeddings, feed_dict = feed_dict)
    # for i in range(nrof_batches_per_epoch):
        # start_index = i*args.batch_size
        # end_index = min((i+1)*args.batch_size, nrof_images)
        # paths_batch = paths[start_index:end_index]
        # img = misc.imread(image_paths[i])
        # images = facenet.load_data(paths_batch, False, False, image_size)
        # feed_dict = { images_placeholder:images, phase_train_placeholder:False }
        # emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
    
    classifier_filename_exp = os.path.expanduser(classifier_file)

    with open(classifier_filename_exp, 'rb') as infile:
        (model, class_names) = pickle.load(infile)
    # try:
    #     class_ind = class_names.index(args.authUser)
    # except:
    #     print("User",authUser," couldn't be found")
    #     return
    print('Loaded classifier model from file "%s"' % classifier_filename_exp)
    #assert(len(dataset) < 2) # only one class
    predictions = model.predict_proba(emb_array)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    #nFalseNeg = 0.0
    #nFalsePos = 0.0
    print('Probability',best_class_probabilities[0])
    return class_names[best_class_indices[0]]
    # for i in range(len(predictions)):
    #     print('Test %d : %.3f' % (i, predictions[i][class_ind]))
    #     if (predictions[i][class_ind] <= args.threshold):
    #         print('--> Image %d not authorized' % (i))
    #         if class_names[labels[i]]==class_names[class_ind]:
    #             nFalseNeg+=1
    #     elif class_names[labels[i]]!=class_names[class_ind]:
    #         nFalsePos+=1
    # for i in range(len(best_class_indices)):
    #     print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
        # if (best_class_probabilities[i] <= args.threshold or class_names[best_class_indices[i]]!=args.authUser):
    #print('labels ',labels)
    #print('class_names ',class_names)
    #print('best class indices ', best_class_indices)
    #accuracy = (len(predictions) - nFalseNeg - nFalsePos)/len(predictions)
    #print('Using threshold =',args.threshold)
    #print('Number of false positives:',nFalsePos)
    #print('Number of false negatives:',nFalseNeg)
    #print('Accuracy: %.3f' % accuracy)

