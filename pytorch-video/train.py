from UCF_Crime_Data_Module import UCFCrimeDataModule
from models.resnet_3d_model import VideoClassificationLightningModule
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
  
def train(min_ephocs, dataset_folder, batch_size, num_workers, stats_file, clip_duration, model_save_dir, subsampled_frames, learning_rate):
    model = VideoClassificationLightningModule(learning_rate)
    data_module = UCFCrimeDataModule(dataset_folder, clip_duration, batch_size, num_workers, subsampled_frames)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                        dirpath=f"{model_save_dir}",
                        filename='resnet-3d-{clip_duration}-{subsampled_frames}-{epoch:02d}-{val_loss:.2f}'.format(clip_duration=clip_duration, subsampled_frames=subsampled_frames),
                        save_top_k=3,
                        mode='min')
    logger = TensorBoardLogger(stats_file, name=f"resnet-3d-{clip_duration}-{subsampled_frames}-{learning_rate}")
    trainer = pytorch_lightning.Trainer(logger=logger, gpus=1, callbacks=[checkpoint_callback],  min_epochs=min_ephocs)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a lstm over a videos dataset using grid search')
    parser.add_argument('--epochs', type=int, help='the min number of epochs')
    parser.add_argument('--dataset', help='the dataset folder which contains 2 folders, videos and labels')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--num_processes', type=int, help='number of processes to use during trainging')
    parser.add_argument('--model_save_path', help='path for model saving')
    parser.add_argument('--stats_file', help='path for tensor board')
    parser.add_argument('--clip_duration', type=int, help='size of submpled clip from the original video')
    parser.add_argument('--subsampled_frames', type=int, help='size of sub submpled frames from the clip')
    parser.add_argument('--lr', type=float, help="learning rate of the model")

    args = parser.parse_args()

    dataset_folder = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    num_processes = args.num_processes
    stats_file = args.stats_file
    prefix_path = args.model_save_path
    clip_duration = args.clip_duration
    subsampled_frames = args.subsampled_frames
    lr = args.lr
    train(epochs, dataset_folder, batch_size, num_processes, stats_file, clip_duration, prefix_path, subsampled_frames, lr)
