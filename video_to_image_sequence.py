import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description='train a lstm over a videos dataset')
parser.add_argument('--dataset', help='the videos folder')
parser.add_argument('--output_folder', help='the output folder of the videos')

args = parser.parse_args()

dataset = args.dataset
output_folder = args.output_folder

for subdir, dirs, files in os.walk(dataset):
    for file in files:
        output_dir_for_video = os.path.join(output_folder, file.split('.')[0])
        os.mkdir(output_dir_for_video)
        subprocess.call(['C:/Users/Dani/Documents/ffmpeg/bin/ffmpeg', '-i', os.path.join(dataset, file), '-r', '10', os.path.join(output_dir_for_video, 'video-%09d.png')])
