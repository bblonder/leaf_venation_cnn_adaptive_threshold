import os
import numpy as np
number_of_jobs = 3

images = os.listdir("leaf_venation_cnn_adaptive_threshold/images")
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
    #SBATCH --job-name=test
    #SBATCH --account=fc_mel
    #SBATCH --partition=savio2_gpu
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=8
    #SBATCH --time=80:10:00

    module load python

    pip install --upgrade pip
    pip install opencv-python
    pip install tensorflow

    cd leaf_venation_cnn_adaptive_threshold
    for i in {res}
    do
    python predict_models.py --name $i
    done
    ''')
