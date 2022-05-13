from utils import *
from constants import *
import warnings

warnings.filterwarnings('ignore')

gpu = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

tf.keras.backend.clear_session()

#After creating the folds, determine which folds/experiments combinations should be run and place in folds/exps array
#Example: exps = ['exp1', 'exp2']
#         folds = ['f12', 'f14']
#exps = ['exp1']
#folds = ['f1']
filenames = os.listdir(os.path.join(os.getcwd(), predict_folder))
total = len(filenames) * len(avg_vds) * len(sliding_window_lengths)
count = 0
for filename in filenames:
    if 'img.png' not in filename or 'cnn' in filename:
        print(f"name of image, {filename}, needs to be modified: must contain '_img.png', and must not be a CNN image")
        count += len(avg_vds) * len(sliding_window_lengths)
        print(f"{count}/{total}")
        continue
    for avg_vd in avg_vds:
        for sliding_window_length in sliding_window_lengths:
            count += 1
            print(f'{count}/{total}: {filename}, vein density {avg_vd}, sliding window length {sliding_window_length}')
            predict_with_vd_thresholding(predict_folder, output_folder, filename, model_patch_size, avg_vd, sliding_window_length, model_location)
        #f_test(fold, exp)


