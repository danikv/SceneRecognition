from UCF_Crime_Features_Dataset import UCFCrimeFeatureDataModule
from models.resnet_3d_with_lstm import VideoClassificationLightningModule
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
  
def train(min_ephocs, dataset_folder, batch_size, num_workers, stats_file, clip_duration, model_save_dir, subsampled_frames, learning_rate, check_val_every_n_epoch, anomaly_classification, hidden_dim, gradient_clip):
    model = VideoClassificationLightningModule(learning_rate, anomaly_classification, hidden_dim)
    data_module = UCFCrimeFeatureDataModule(dataset_folder, batch_size, num_workers)
    base_filename = f'resnet-3d-lstm-{clip_duration}-{subsampled_frames}-{hidden_dim}'
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                        dirpath=f"{model_save_dir}",
                        filename= base_filename + '-{epoch:02d}-{val_loss:.2f}',
                        save_top_k=3,
                        mode='min')
    logger = TensorBoardLogger(stats_file, name=f"resnet-3d-lstm-{clip_duration}-{subsampled_frames}-{learning_rate}-{hidden_dim}-{anomaly_classification}")
    trainer = pytorch_lightning.Trainer(logger=logger, gpus=1, callbacks=[checkpoint_callback],  min_epochs=min_ephocs, check_val_every_n_epoch=check_val_every_n_epoch, gradient_clip_val=gradient_clip)
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
    parser.add_argument('--check_val', type=int, help="how many train epochs until we check on the val data set")
    parser.add_argument('--anomaly_classification', type=bool, help="classify as classes or anomelies")
    parser.add_argument('--hidden_dim', type=int, help="hidden dim of the model")
    parser.add_argument('--clip', type=float, help="gradient clipping norm")

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
    check_val_every_n_epoch = args.check_val
    anomaly_classification = args.anomaly_classification
    hidden_dim = args.hidden_dim
    clip = args.clip
    train(epochs, dataset_folder, batch_size, num_processes, stats_file, clip_duration, prefix_path, subsampled_frames, lr, check_val_every_n_epoch, anomaly_classification, hidden_dim, clip)
