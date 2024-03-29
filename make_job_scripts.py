import os
import numpy as np
number_of_jobs = 3

images = os.listdir("./images")
images = [x for x in images if "_roi." not in x]
if len(images) < number_of_jobs:
    number_of_jobs = len(images)
partitions = np.array_split(np.array(images), number_of_jobs)
print(partitions)
for i in range(len(partitions)):
    image_names = partitions[i].tolist()
    res = " ".join([str(item) for item in image_names])
    os.makedirs("./jobs", exist_ok=True)
    file_name = "./jobs/" + "job_" + str(i) + ".sh"
    with open (file_name, 'w') as rsh:
        rsh.write(f'''\
#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --account=fc_mel
#SBATCH --partition=savio2_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00

module load ml/tensorflow/2.5.0-py37
pip install --user opencv-python

cd leaf_venation_cnn_adaptive_threshold
for i in {res}
do
python predict_models.py --name $i
done
    ''')
