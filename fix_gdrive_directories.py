import os
import shutil

# seems to not be a good organization structure

# let's just document the workflow of using rclone to download, moving images to 
# modify the way the predict_with_vd_thresholding fn works to just pick up any images int he images folder, regardless of how they are named - but wait, we need masks too so we need to document this in the README as well

def find_all_imgs_and_rois_in_folder(folder_path):
    files_in_folder = os.listdir(folder_path)
    imgs = []
    rois = []
    for filename in files_in_folder:
        file_path = os.path.join(folder_path, filename)
        # check if we have subdirectory
        if os.path.isdir(file_path):
            sub_imgs, sub_rois = find_all_imgs_and_rois_in_folder(file_path)
            imgs += sub_imgs
            rois += sub_rois
        # case if image
        elif ('crop.png' in filename) or ('crop.jpg' in filename) or ('img.jpg' in filename) or ('img.png' in filename):
            imgs.append(file_path)
            # do logic to get img and mask
        # case if ROI
        elif ('mask.png' in filename) or ('mask.jpg' in filename) or ('roi.png' in filename) or ('roi.jpg' in filename):
            rois.append(file_path)
    
    return imgs, rois

# assumes run from same folder that has "images"
def move_all_imgs_and_rois(folder_path):
    imgs, rois = find_all_imgs_and_rois_in_folder(folder_path)
    images_folder = os.path.join(os.getcwd(), 'images')
    for img_path in imgs:
        if os.path.isfile(os.path.join(images_folder, img_path.split('/')[-1])):
            continue
        shutil.move(img_path, images_folder)
    for roi_path in rois:
        if os.path.isfile(os.path.join(images_folder, roi_path.split('/')[-1])):
            continue
        shutil.move(roi_path, images_folder)
        

move_all_imgs_and_rois(os.path.join(os.getcwd(), "wilf"))
