from utils import *
from constants import *

gpu = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

tf.keras.backend.clear_session()

#num_folds = 2
#est_img_per_fold = 1
#convert_mat = True #Set this to True if you want to automatically convert models to matlab compatible
make_fold_folders(num_folds) #creates training and validation folds from images inside the images folder
separate_test(num_folds, test_img_per_fold)

#After creating the folds, determine which folds/experiments combinations should be run and place in folds/exps array
#Example: exps = ['exp1', 'exp2']
#         folds = ['f12', 'f14']
#exps = ['exp1']
#folds = ['f1']

for fold in folds:
    for exp in exps:
        train(fold, exp, start_epoch, end_epoch, numerator, patch_size, patches_per_image, int(gpu))
        #f_test(fold, exp)

if convert_mat:
    path = None #Specify path to folder with .h5 models
    # convert_to_mat(path)

