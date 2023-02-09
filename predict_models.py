from utils import *
from constants import *
import warnings
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name',
                        help='')

    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    gpu = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    tf.keras.backend.clear_session()

    #After creating the folds, determine which folds/experiments combinations should be run and place in folds/exps array
    #Example: exps = ['exp1', 'exp2']
    #         folds = ['f12', 'f14']
    #exps = ['exp1']
    #folds = ['f1']
    #imagenames = os.listdir(os.path.join(os.getcwd(), predict_folder))
    imagenames = [args.name]
    #imagenames = ["Linaceae_Linum_monogynum_Hickey_Hickey_2435_crop_img.png"]
    total = len(imagenames) * len(avg_vds) * len(sliding_window_lengths)
    count = 0
    #raise Exception
    for imagename in imagenames:
        if 'mask.' in imagename:
            os.rename(os.path.join(os.getcwd(), predict_folder, imagename), os.path.join(os.getcwd(), predict_folder, imagename[:-8] + 'roi.png'))

    for imagename in imagenames:
        if 'roi' in imagename:
            continue
        # make name crop_img.png
        elif 'crop_crop.' in imagename:
            os.rename(os.path.join(os.getcwd(), predict_folder, imagename), os.path.join(os.getcwd(), predict_folder, imagename[:-8] + 'img.png'))
            imagename = imagename[:-8] + 'img.png'
        # make name crop_img.png
        elif 'crop.' in imagename:
            os.rename(os.path.join(os.getcwd(), predict_folder, imagename), os.path.join(os.getcwd(), predict_folder, imagename[:-4] + 'img.png'))
            imagename = imagename[:-4] + 'img.png'
        #filenames = os.listdir(os.path.join(os.getcwd(), predict_folder, imagename))
        output_folder_fold = os.path.join(os.getcwd(), output_folder, imagename[:-4])
        os.makedirs(output_folder_fold, exist_ok = True)

        if 'img.png' not in imagename or 'cnn' in imagename:
            print(f"name of image, {imagename}, needs to be modified: must contain 'img.png', and must not be a CNN image")
            print(f"{count}/{total}")
            continue
        #print(predict_folder, filename, output_folder_fold)
        for avg_vd in avg_vds:
            for sliding_window_length in sliding_window_lengths:
                count += 1
                print(f'{count}/{total}: {imagename}, vein density {avg_vd}, sliding window length {sliding_window_length}')
                predict_with_vd_thresholding(predict_folder, output_folder_fold, imagename, model_patch_size, avg_vd, sliding_window_length, voting, model_location)
            #f_test(fold, exp)


