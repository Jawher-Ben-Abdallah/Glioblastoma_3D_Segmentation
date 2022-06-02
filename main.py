from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, LightningDataModule

from utils.args import get_main_args
from data_loading.data_module import BraTS20DataModule
from nnunet.model import NNUnet

if __name__ == "__main__":
    args = get_main_args()
    callbacks = []
    model = NNUnet(args)
    model_ckpt = ModelCheckpoint(dirpath="./", filename="best_model",
                                monitor="dice_mean", mode="max", save_last=True)
    callbacks.append(model_ckpt)
    dm = BraTS20DataModule(samples_per_epoch=400)
    trainer = Trainer(callbacks=callbacks, enable_checkpointing=True, max_epochs=1, 
                    enable_progress_bar=True, gpus=1, accelerator="gpu", amp_backend='apex')

    trainer.fit(model, dm)

    #args = get_main_args()
    #dm = BraTS20DataModule(args)
    #dm.setup()
    #img, lbl = next(iter(dm.train_dataloader()))
    #print(img.shape, lbl.shape)