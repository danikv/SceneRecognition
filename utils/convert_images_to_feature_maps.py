import torch
import argparse
import os
import cv2
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a lstm over a videos dataset')
    parser.add_argument('--dataset', help='the dataset folder which contains 2 folders, videos and labels')
    parser.add_argument('--output_folder', help='the dataset folder which contains 2 folders, videos and labels')
    parser.add_argument('--batch_size', type=int, help='batch size')

    args = parser.parse_args()


    dataset_folder = args.dataset
    output_folder = args.output_folder
    videos_folder = os.path.join(dataset_folder, 'Videos-Images-1fps')
    batch_size = args.batch_size

    resnet101 = models.resnet101(pretrained=True)
    model = nn.Sequential(*list(resnet101.children())[:-1])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for subdir, dirs, files in os.walk(videos_folder):
            for dir in dirs:
                current_dir = os.path.join(videos_folder, dir)
                for subdir, dirs, files in os.walk(current_dir):
                    #create directory
                    output_dir_for_video = os.path.join(output_folder, dir)
                    os.mkdir(output_dir_for_video)
                    for file in files:
                        image_path = os.path.join(current_dir, file)
                        image = cv2.imread(image_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = transform(image).view(1,3,224,224)
                        image = image.to(device)
                        feature_map = model(image)
                        new_path = os.path.join(output_dir_for_video, file.split('.')[0])
                        torch.save(feature_map, new_path)



