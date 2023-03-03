# LeafVeinCNN

This repository is a collection of work that helps researchers train machine learning models to segment leaves.

# How to Use
## Training

Create a folder called `images` and upload each leaf you want to include in your training set in an individual folder. Note that in order to properly work, this folder should contain the leaf image, the leaf ROI image, and the segmentation of the leaf image. The ROI image should be light-colored (preferably white) within the ROI and black or empty outside. The segmentation image should be white for veins and black for non-veins. Within that folder, make sure that the leaf image is formatted as `*_img.png`, the ROI image is formatted as `*_roi.png`, and the segmented leaf image as `*_seg.png`.

Check the `constants.py` file to ensure that you are fine with the other training parameters. More details are given in the ‘Customizing Training’ section below.

Once all the leaves you wish to train with have been uploaded and are formatted correctly, run `python3 train_models.py`.

## Segmenting

To segment with a given model, create a folder that contains all the leaf images you wish to segment, and modify the `predict_folder` variable in the `constants.py` file to point to this folder. Additionally, modify the `model_location` variable in the same file to point to the model you wish to segment the images with. This is the least you must do before predicting.

Then, run `python3 predict_models.py`.

## Customizing Training
There exist default hyperparameters for the data augmentation part of the pipeline which creates augmentations to increase the size of the training set. The below parameters can be changed from their default values:

### Data Augmentation
ROI_RATIO: Ratio of ROI to non-ROI within a patch, near the edge of a leaf. 

SHADOW: Probability with which to draw a shadow. 

BUBBLE : probability with which to draw a bubble. 

G_NOISE : probability with which to add gaussian noise. 


BRIGHT_MAG : Magnitude of brightness deviation. 


FLIP_UD : probability with which to flip up and down. 


FLIP_LR : probability with which to flip left and right. 


FT_PROB : probability with which to add a fourier trasform. 

### CNNs
Below are the hyperparameters for training the models. If the models do not perform well, then we recommend tuning end_epoch and learning_rate before the other hyperparaters (including the data augmentation hyperparaters).  

learning_rate : This is the learning rate for CNN in Adam optimizer. 

BATCH_SIZE : This is the batch size for training CNNs in model.fit. 

start_epoch : Change only if crash during training, to restart from the epoch where the last training ended. 

end_epoch : This is the epoch number after which the training will end, the default value is 33 (found it was good through testing)    


## Customizing Segmenting
The following parameters are used by the predicting script after the models have been trained. The models segment the input image while maintain a certain average vein density a certain window size. This window then slides across the image to normalize the image. NOTE: Very important to change model_patch_size to be consistent with the patch_size of the training data used to train the specific model being used.

Model_patch_size : The patch size that the predicting model was trained on. 

avg_vds : An array with the vein densities used to normalize within each window, where each element in the array is ratio between white to black pixels. 

Sliding_window_lengths: An array with the sizes of the sliding windows, each element is the side length of a square window. 


## Under the Hood

`train_models.py` iterates over the number of folds and experiments specified in the `constants.py` file and calls the `train` function defined in `utils.py`. 

Likewise, `predict_models.py` iterates over each file to be predicted on and calls the `predict_with_vd_thresholding` function, defined in `utils.py`, at all vein density and sliding window size combinations the user requests.


## Running Jobs on Savio
* On Google Drive, collect all images and rois into a new folder called 'images', and place this new folder within an outer folder on Google Drive (outer folder required due to some quirks of rclone).
* Log into your Savio account with ssh, and then navigate to Savio's scratch directory by entering 'cd ../../../scratch/users/[your username]' into the command line.
* git clone this repository with 'git clone https://github.com/bblonder/leaf_venation_cnn_adaptive_threshold.git' and then enter into the repository with 'cd leaf_venation_cnn_adaptive_threshold'.
* 'module load rclone', and then 'rclone copy' the outer folder onto Savio into scratch directory. The inner folder will be copied into the directory.
* If the new folder is not named 'images', change the path in make_jobs_scripts.py line 5 to be the name of the folder.
* Set the number of jobs according to the number of images following roughly a 2 image : 1 job ratio for larger images and 4:1 for smaller images. This is done in make_jobs_scripts.py line 3.
* Run the following command: 'module load python' and then 'python3 make_job_scripts.py'
* Navigate into the 'jobs' folder and launch each job script by running the following command for each script: 'sbatch {job name}.sh'
* Run "squeue -u $USER" to ensure each job is in the queue.


*Remember the following tradeoff: Increasing the number of jobs means more jobs have to be scheduled which increases time to process all images. However, having too few jobs means having more images per job which also increases time to process all images.
