from UCF_Crime_Features_Dataset import UCFCrimeFeatureDataModule
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
  
def train(min_ephocs, dataset_folder, batch_size, num_workers, stats_file, model_save_dir, check_val_every_n_epoch, gradient_clip, base_filename):
    
    data_module = UCFCrimeFeatureDataModule(dataset_folder, batch_size, num_workers)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                        dirpath=f"{model_save_dir}",
                        filename= base_filename + '-{epoch:02d}-{val_loss:.2f}',
                        save_top_k=3,
                        mode='min')
    logger = TensorBoardLogger(stats_file, name=base_filename)
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
    parser.add_argument('--hidden_dim', type=int, help="hidden dim for the lstm model", required=False)
    parser.add_argument('--lstm_layers', type=int, help="hidden dim for the lstm model", required=False)
    parser.add_argument('--num_heads', type=int, help="number of head of the attention model", required=False)
    parser.add_argument('--clip', type=float, help="gradient clipping norm", required=False)
    parser.add_argument('--type', type=str, help='attation or lstm')

    args = parser.parse_args()

    dataset_folder = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    num_processes = args.num_processes
    stats_file = args.stats_file
    prefix_path = args.model_save_path
    clip_duration = args.clip_duration
    subsampled_frames = args.subsampled_frames
    learning_rate = args.lr
    check_val_every_n_epoch = args.check_val
    anomaly_classification = args.anomaly_classification
    type = args.type
    if type == 'lstm':
        from models.resnet_3d_with_lstm import VideoClassificationLightningModule
        hidden_dim = args.hidden_dim
        lstm_layers = args.lstm_layers
        clip = args.clip
        model = VideoClassificationLightningModule(learning_rate, anomaly_classification, hidden_dim)
        base_filename = f'resnet-3d-lstm-{clip_duration}-{subsampled_frames}-{hidden_dim}-{learning_rate}-{anomaly_classification}-{clip}'
    else:
        from models.resnet_3d_with_attention import VideoClassificationLightningModule
        num_heads = args.num_heads
        clip = None
        model = VideoClassificationLightningModule(learning_rate, anomaly_classification, num_heads)
        base_filename = f'resnet-3d-attention-{clip_duration}-{subsampled_frames}-{num_heads}-{learning_rate}-{anomaly_classification}'
    train(epochs, dataset_folder, batch_size, num_processes, stats_file, prefix_path, check_val_every_n_epoch, clip, base_filename)
