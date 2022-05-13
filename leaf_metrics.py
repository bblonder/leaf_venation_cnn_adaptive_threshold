import os
import matlab.engine
import numpy as np
import random
import time
import os
import sys
import cv2
import imageio
from scipy import misc, ndimage, io, linalg, stats
from PIL import Image, ImageOps
from time import gmtime
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
import h5py
import shutil
from scipy.ndimage.filters import gaussian_filter
import itertools
import matlab.engine
import queue
import multiprocessing as mp
import matplotlib.pyplot as plt
from itertools import cycle
import copy
import math
import skimage.measure
import seaborn as sns
import sklearn

def leaf_metrics(fold="", test_folder="", test_case=""):
    #basedir = os.path.join('images', fold, 'test', test_folder)
    cwd = os.getcwd()
    
    #basedir = os.path.join(cwd)
    basedir = cwd
    #filecode = file_name + '_cnn.png'
    px_per_mm = 200
    #med_filt = median smoothing filter (= kernel size)
    med_filt = 5
    spur_length_max = 10
    #spur_length_max = 0 (keeps everything) potentially 1 also
    color_roi = [255, 255, 0] #yellow
    color_vein =[255, 0, 0] #red
    discard_boundary = 1  #if whole leaf, then 1
    plot_image = 1 #(0 if testing not needed)

    result_save_folder = basedir

    #threshold
    # if not os.path.exists(os.path.join(result_save_folder, test_case)):
    #     os.mkdir(os.path.join(result_save_folder, test_case))
    # img_test_sample = test_case + '_img.png'
    # roi_test_sample = test_case + '_roi.png'
    # seg_test_sample = test_case + '_seg.png'
    # cwd = os.getcwd()
    # # region of interest
    # roi = Image.open(os.path.join(cwd, 'images', fold, "test", test_folder, roi_test_sample)).convert("L")
    # roi = np.array(roi)
    # # original image
    # img = Image.open(os.path.join(cwd, 'images', fold, "test", test_folder, roi_test_sample)).convert("L")
    # img = ImageOps.autocontrast(img, cutoff = 0.01)
    # img = np.array(img)
    # # segmentation
    # seg = Image.open(os.path.join(cwd, 'images', fold, "test", test_folder, roi_test_sample)).convert("L")
    # seg = np.array(seg)
    # # because the red segmentation color is below 127.5 so set to 0 otherwise
    # seg[np.nonzero(seg)] = 255
    
    #filecode = img_test_sample
    filecode = "ax638.png"
    print(basedir)
    print(filecode)

    eng = matlab.engine.start_matlab()
    result = eng.calculate_vein_stats(basedir, filecode, px_per_mm, med_filt, spur_length_max, color_roi, color_vein, discard_boundary, plot_image, nargout=2)
    result_table = result[0]
    result_other = result[1]
    print(result_table)
    print(result_other)
    return

    roi[roi < 0] = 0
    roi[(roi > 0) & (roi <= 127.5)] = 0
    roi[(roi > 127.5)] = 1
    seg[seg < 0] = 0
    seg[(seg > 0) & (seg <= 127.5)] = 0
    seg[(seg > 127.5)] = 1
    seg[roi == 0] = 0

    patch_size = 256
    x = np.arange(0, img.shape[0] - patch_size, 50)
    y = np.arange(0, img.shape[1] - patch_size, 50)
    xx, yy = np.meshgrid(x, y, sparse=True)
    pred_mask = np.zeros(img.shape)
    pred_mask[0:xx[0][-1], 0:yy[-1][0]] = 1

    patch_samples = np.ndarray((len(xx[0, ...]) * len(yy[..., 0]), patch_size, patch_size, 1), dtype='float32')
    k = 0
    for xi in xx[0, ...]:
        for yi in yy[..., 0]:
            x_min = int(xi)
            x_max = int(xi + patch_size)
            y_min = int(yi)
            y_max = int(yi + patch_size)
            patch_samples[k, ..., 0] = img[x_min:x_max, y_min:y_max]
            k += 1
    
    model_location = os.path.join(cwd, 'f11-256-weights.24-0.383.h5')
    model = load_model(model_location)
    
    pred_sample = model.predict(patch_samples, batch_size=1)
    pred_seg = np.zeros(img.shape)
    weit_seg = np.zeros(img.shape)
    k = 0
    for xi in xx[0, ...]:
        for yi in yy[..., 0]:
            x_min = int(xi)
            x_max = int(xi + patch_size)
            y_min = int(yi)
            y_max = int(yi + patch_size)
            pred_seg[x_min:x_max, y_min:y_max] += pred_sample[k, ..., 0]
            weit_seg[x_min:x_max, y_min:y_max] += np.ones((patch_size, patch_size))
            k += 1
    weit_seg[weit_seg > 0] = 1 / weit_seg[weit_seg > 0]
    pred_res = np.multiply(pred_seg, weit_seg)
    Image.fromarray(pred_res * 255).convert('RGB').save(os.path.join(result_save_folder, 'forcv.png'))
    #this is arbitrary, just so we can separate true positives and true negatives
    f2betascores = {}
    prscores = {}
    seg[seg == 0] = 10
    threshold = list(np.linspace(0.0, 1.0, 51))
    pscores = []
    rscores = []
    # for i in range(len(threshold)):
        # pred_res_copy = copy.deepcopy(pred_res)
        # pred_res_copy[pred_res_copy >= threshold[i]] = 1
        # pred_res_copy[pred_res_copy < threshold[i]] = 0
        # pred_res_copy[roi == 0] = 0
        # comparison = np.add(pred_res_copy, -1 * seg)

        # true_positives = (comparison == 0).sum()
        # false_positives = (comparison == -9).sum()
        # false_negatives = (comparison == -1).sum()
        # precision = true_positives / (false_positives + true_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # f2beta = (1 + math.pow(2, 2)) * (precision * recall) / (math.pow(2, 2) * precision + recall)
        # f2betascores[f2beta] = threshold[i]
        # prscores[threshold[i]] = (recall, precision)
        # pscores.append(precision)
        # rscores.append(recall)

    img = cv2.imread(os.path.join(result_save_folder, 'forcv.png'),0)
    #img = cv2.medianBlur(img,5)

    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    #plt.imshow(th3,'gray')
   


    # maxthreshold = f2betascores[max(f2betascores.keys())]
    # #saving f2betascores (f2beta -> threshold)
    # np.save(os.path.join(result_save_folder, test_case, test_case + '-' + str(maxthreshold) + str(max(f2betascores.keys())) + 'f2betadict.npy'), np.array(f2betascores))
    # print(str(maxthreshold))
    # print(str(max(f2betascores.keys())))
    # thresh_res = np.copy(pred_res)
    # thresh_res[thresh_res >= maxthreshold] = 1
    # thresh_res[thresh_res < maxthreshold] = 0

    thresh_res = np.asarray(th3)

    #store previous value of precision and recall for trapezoid rule
    # prevp = 0
    # prevr = 0
    # AUC = 0
    # c = list(zip(rscores, pscores))
    # #sort by recall
    # sortedvals = sorted(c, key = lambda pair: pair[0])
    # srecall, sprecision = zip(*sortedvals)

    # for j in range(len(srecall)):
    #     deltax = srecall[j] - prevr
    #     yavg = (sprecision[j] + prevp) / 2
    #     AUC += deltax * yavg
    #     prevr = srecall[j]
    #     prevp = sprecision[j]
        
    # #plotting and saving plot
    # plt.scatter(rscores, pscores, label="AUC: " + str(AUC))
    # plt.axis([0, 1, 0, 1])
    # ax = plt.gca()
    # ax.set_autoscale_on(False)
    # plt.legend(loc='upper right')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(test_case)
    # plt.savefig(os.path.join(result_save_folder, test_case + 'prcurve.png'))
    # plt.close()
    #saving images
    Image.fromarray(pred_res * 255).convert('RGB').save(os.path.join(result_save_folder, test_case + '_cnn.png'))
    Image.fromarray(seg * 255).convert('RGB').save(os.path.join(result_save_folder, test_case + '_seg.png'))
    Image.fromarray(img).convert('RGB').save(os.path.join(result_save_folder, test_case + '_img.png'))
    Image.fromarray(pred_mask * 255).convert('RGB').save(os.path.join(result_save_folder, test_case + '_cnn_mask.png'))
    Image.fromarray(thresh_res * 255).convert('RGB').save(os.path.join(result_save_folder, test_case + '_cnn_thresh.png'))
    
    filecode = test_case + '_cnn_thresh.png'
#set veins to some color
#set roi to some other color (as per input image) 

    eng = matlab.engine.start_matlab()
    result = eng.calculate_vein_stats(basedir, filecode, px_per_mm, med_filt, spur_length_max, color_roi, color_vein, discard_boundary, plot_image, nargout=2)
    result_table = result[0]
    result_other = result[1]
    print(result_table)
    print(result_other)

leaf_metrics()
