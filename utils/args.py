from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("--input_dir", type=str, default=None, help="Original BraTS Files Directory")
    arg("--output_dir", type=str, default=None, help="Preprocessed Files Directory")
    arg("--base_dir", type=str, default="BraTS2020_Preprocessed", help="Train Data Directory")
    arg("--pred_dir", type=str, default=None, help="Model Predictions Save Directory")
    arg("--patch_size", type=int, default=128, help="Shape of Training Patches")
    arg("--val_size", type=int, default=224, help="Shape of Validation Patches")
    arg("--batch_size", type=int, default=2, help="batch size")
    arg("--samples_per_epoch", type=int, default=300, help="Number of Samples Used Per Epoch")
    arg("--in_channels", type=int, default=4, help="Network Input Channels")
    arg("--out_channels", type=int, default=3, help="Network Output Channels")
    arg("--seed", type=int, default=26012022, help="Random Seed")
    arg("--num_workers", type=int, default=1, help="Number of DataLoader Workers")
    arg("--learning_rate", type=float, default=1e-4, help="Learning Rate")
    arg("--weight_decay", type=float, default=1e-5, help="Weight Decay")
    arg("--kernels", type=list, default=[[3, 3, 3]] * 4, help="Convolution Kernels")
    arg("--strides", type=list, default=[[1, 1, 1]] +  [[2, 2, 2]] * 3, help="Convolution Strides")
    arg("--augment", type=bool, default=False, help="Apply Data Augmentation")
    arg("--num_epochs", type=int, default=10, help="Number of Epochs")
    arg("--exec_mode", type=str, default='train', help='Execution Mode')
    arg("--ckpt_path", type=str, default=None, help='Checkpoint Path')
    arg("--save_path", type=str, default='./', help='Saves Path')

    return parser.parse_args()