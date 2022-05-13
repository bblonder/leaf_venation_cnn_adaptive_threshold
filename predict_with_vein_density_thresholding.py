from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

tf.keras.backend.clear_session()

files = []
#Example: files = ["P_ACL176"]
#Input arguements: 
for i in range(len(files)):
    convert_roi_to_grayscale(files[i])
    for vein_density in [0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
        for patch_size in [16, 32, 64, 128, 256, 512]:
            predict_with_vd_thresholding(files[i], "", 512, 24, vein_density, patch_size)