#!/bin/bash


#SBATCH --partition main

#SBATCH --job-name scene_recognition_model

#SBATCH --output scene_rec-%J.out

#SBATCH --mail-user=kovachev@post.bgu.ac.il

#SBATCH --mail-type=FAIL

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=16

module load anaconda

source activate scene_recognition

echo 'Finished Copying , start training'

python /home/kovachev/SceneRecognition/train.py --ephocs 100 --dataset /home/kovachev/Dataset --output_file /home/kovachev/model_perf.pkl --batch_size 800 --num_processes 16 --model_save_path /home/kovachev --model_class Base_Model > /home/kovachev/output.log
