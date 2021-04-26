#!/bin/bash


#SBATCH --partition main

#SBATCH --job-name scene_recognition_model_lstm_base

#SBATCH --output scene_rec_lstm_base-%J.out

#SBATCH --mail-user=kovachev@post.bgu.ac.il

#SBATCH --mail-type=FAIL

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

module load anaconda

source activate scene_recognition

echo 'Finished Copying , start training'

python /home/kovachev/SceneRecognition/train.py --ephocs 100 --dataset /home/kovachev/Dataset --output_file /home/kovachev/model_lstm_base_perf.pkl --batch_size 64 --num_processes 8 --model_save_path /home/kovachev --model_class LSTM_Base_Model --model_prefix lstm_base_model > /home/kovachev/output.log
