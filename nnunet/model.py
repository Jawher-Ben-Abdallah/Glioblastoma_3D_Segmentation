import gc
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from monai.networks.nets import DynUNet

from nnunet.losses import LossBraTS
from nnunet.metrics import Dice


class NNUnet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.best_mean, self.best_mean_epoch = (0,) * 2
        self.best_dice, self.best_epoch, self.best_mean_dice = (args.out_channels * [0],) * 3
        self.build_model()
        self.loss = LossBraTS()
        self.dice = Dice(n_class=args.out_channels)
    
    def training_step(self, batch, batch_idx):
        img, lbl, _ = batch
        logits = self.model(img)
        loss = self.loss(logits, lbl)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, lbl, _ = batch
        logits = self.model(img)
        loss = self.loss(logits, lbl)
        self.dice.update(logits, lbl[:, 0], loss)
        
    def predict_step(self, batch, batch_idx):
        img, lbl, patient_id = batch
        preds = self.model(img)
        preds = (nn.Sigmoid()(preds) > 0.5).int()
        lbl_np = lbl.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        np.save(f'{self.args.pred_dir}/{patient_id[0]}-prediction.npy', preds_np)
        np.save(f'{self.args.pred_dir}/{patient_id[0]}-label.npy', lbl_np)
        
    def training_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        gc.collect()
        
    def validation_epoch_end(self, outputs):
        dice, loss = self.dice.compute()
        self.dice.reset()
        dice_mean = torch.mean(dice)
        
        if dice_mean >= self.best_mean:
            self.best_mean = dice_mean
            self.best_mean_dice = dice[:]
            self.best_mean_epoch = self.current_epoch
            
        for i, dice_i in enumerate(dice):
            if dice_i > self.best_dice[i]:
                self.best_dice[i], self.best_epoch[i] = dice_i, self.current_epoch
                
        metrics = {}
        metrics.update({"Mean_Dice": round(torch.mean(dice).item(), 2)})
        metrics.update({"Highest": round(torch.mean(self.best_mean_dice).item(), 2)})
        metrics.update({f"L{i+1}": round(m.item(), 2) for i, m in enumerate(dice)})
        metrics.update({"val_loss": round(loss.item(), 4)})
        
        print(f"Epoch: {self.current_epoch + 1}")
        print(f"Validation Performace: Mean Dice {metrics['Mean_Dice']}, Validation Loss {metrics['val_loss']}")
        self.log("dice_mean", dice_mean)
        
        torch.cuda.empty_cache()
        gc.collect()
        
    def build_model(self):
        self.model = DynUNet(
            spatial_dims=3,
            in_channels=self.args.in_channels,
            out_channels=self.args.out_channels,
            kernel_size=self.args.kernels,
            strides=self.args.strides,
            upsample_kernel_size=self.args.strides[1:],
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01})
        )
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)