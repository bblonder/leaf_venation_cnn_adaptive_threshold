#Note: Working directory is home folder

#Data Augmentation-------------------------------------
ROI_RATIO =  3/8 #Ratio of roi to non roi within a patch, near the edge of a leaf
SHADOW = 0.92 #1 - probability with which to draw a shadow
BUBBLE = 0.25 #1 - probability with which to draw a bubble
G_NOISE = 0.95 #1 - probability with which to add gaussian noise
BRIGHT_MAG = 40 #Magnitude of brightness deviation
FLIP_UD = 0.5  #1 - probability with which to flip up and down
FLIP_LR = 0.5  #1 - probability with which to flip left and right
FT_PROB = 0.15 #1 - probability with which to add a fourier trasform

#Data loading and separation for training CNNs---------
patches_per_image = 8 #Default:512 number of patches extracted from each image with data augmentation
patch_size = 256 #size of patch
numerator = 2 #Default: 16 number of validation patches
available_cpus = 2 # NOTE: MUST BE A FACTOR OF NUMERATOR allocate as many as possible

#Training CNNs-----------------------------------------
learning_rate = 1e-4 #learning rate for CNN in Adam opt
BATCH_SIZE = 2 #batch size for training CNNs in model.fit
start_epoch = 0 # epoch to start training at
end_epoch = 3 # Default: 33 (found it was good through testing)  last epoch to train until
num_folds = 2
test_img_per_fold = 1
convert_mat = True
exps = ['exp1']
folds = ['f1']


#ADD CONVERT_MAT BOOL

#Prediction
predict_folder = "images"
output_folder = "output"
model_patch_size = 256 # the patch size that predicting model was trained on
avg_vds = [0.2, 0.3] # the vein density to normalize each sliding window 
sliding_window_lengths =  [32, 128] #
model_location = "weights/f14-512-weights.30-0.368.h5"
voting = False # pixels across the predictions are averaged if false, majority vote if true
