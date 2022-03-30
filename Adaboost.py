import numpy as np
import skimage.io as io
from skimage.transform import resize
import os
from functools import partial
import progressbar
from multiprocessing import Pool
from IPython.core.display import Image, display
from haar_features import *
from intagral_image import *

LOADING_BAR_LENGTH = 50

def _get_feature_vote(feature, image):
    return feature.vote(image)

#AdaBoost training function it takes 
#postive_iis -> faces images
#negative_iis -> non faces images
#num_classifiers -> number of returned classifiers
#features width and height -> determine range of feature sizes that will be created
def learn(positive_iis, negative_iis, num_classifiers=-1, min_feature_width=1, max_feature_width=-1, min_feature_height=1, max_feature_height=-1):
   
    
    num_pos = len(positive_iis)
    num_neg = len(negative_iis)
    num_imgs = num_pos + num_neg
    img_height, img_width = positive_iis[0].shape
    max_feature_height = img_height if max_feature_height == -1 else max_feature_height
    max_feature_width = img_width if max_feature_width == -1 else max_feature_width

    pos_weights = np.ones(num_pos) * 1. / (2 * num_pos)
    neg_weights = np.ones(num_neg) * 1. / (2 * num_neg)
    weights = np.hstack((pos_weights, neg_weights))
    labels = np.hstack((np.ones(num_pos), np.ones(num_neg) * -1))

    images = positive_iis + negative_iis
    features = featuresCreation(img_height, img_width, min_feature_width, max_feature_width, min_feature_height, max_feature_height)
    num_features = len(features)
    feature_indexes = list(range(num_features))

    num_classifiers = num_features if num_classifiers == -1 else num_classifiers

    print('Calculating scores for images..')
    
    votes = np.zeros((num_imgs, num_features))
    bar = progressbar.ProgressBar()
    pool = Pool(processes=None)
    for i in bar(range(num_imgs)):
        votes[i, :] = np.array(list(pool.map(partial(_get_feature_vote, image=images[i]), features)))


    classifiers = []

    print('Selecting classifiers..')
    bar = progressbar.ProgressBar()
    for _ in bar(range(num_classifiers)):

        classification_errors = np.zeros(len(feature_indexes))
        weights *= 1. / np.sum(weights)

        for f in range(len(feature_indexes)):
            f_idx = feature_indexes[f]
            error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0, range(num_imgs)))
            classification_errors[f] = error
        min_error_idx = np.argmin(classification_errors)
        best_error = classification_errors[min_error_idx]
        best_feature_idx = feature_indexes[min_error_idx]
        best_feature = features[best_feature_idx]
        feature_weight = 0.5 * np.log((1 - best_error) / best_error)
        best_feature.weight = feature_weight
        classifiers.append(best_feature)

        weights = np.array(list(map(lambda img_idx: weights[img_idx] * np.sqrt((1-best_error)/best_error) if labels[img_idx] != votes[img_idx, best_feature_idx] else weights[img_idx] * np.sqrt(best_error/(1-best_error)), range(num_imgs))))

        feature_indexes.remove(best_feature_idx)
    
    return classifiers

def ensemble_vote(int_img, classifiers):

    return 1 if sum([c.vote(int_img) for c in classifiers]) >= 0 else 0


def ensemble_vote_all(int_imgs, classifiers):

    vote_partial = partial(ensemble_vote, classifiers=classifiers)
    return list(map(vote_partial, int_imgs))

#upload images for training and resize it to 50X50
#trainingmax is number of training images to load
#testmax is number of test images to load
def load_images(path, trainingmax=400,testmax=70,istest=False):
    images = []
    count=0
    max=trainingmax
    if (istest):
      max=testmax
    for _file in os.listdir(path):
        count += 1
        if count>max:
          break
        if _file.endswith('.png') or _file.endswith('.jpg') or _file.endswith('.jpeg'):
            #read image as gray
            img_arr = io.imread(os.path.join(path, _file), as_gray = True)
            #resize image
            img_arr = resize(img_arr, (25,25))
            #normalize photo
            img_arr /= img_arr.max()
            #add image to the array
            images.append(img_arr)
    return images