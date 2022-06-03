import numpy as np
from torch.utils.data import DataLoader, Sampler, BatchSampler
import pytorch_lightning
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

from data_loading.dataset import BraTS20Dataset
from utils.utils import get_patients

class BraTS20DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.prepare_data_per_node = True
        self.args = args

    def setup(self, stage=None):
        patients = get_patients(self.args.base_dir)
        patients_train, patients_val_test = train_test_split(patients, test_size=25, random_state=self.args.seed)
        patients_val, patients_test = train_test_split(patients_val_test, test_size=5, random_state=self.args.seed)
        self.brats_train = BraTS20Dataset(patients_train, self.args, mode='train')
        self.brats_val = BraTS20Dataset(patients_val, self.args, mode='validate')
        self.brats_test = BraTS20Dataset(patients_test, self.args, mode='validate')
        
    def train_dataloader(self):
        train_sampler = self.PatientSampler(len(self.brats_train), self.args.samples_per_epoch)
        train_bSampler = BatchSampler(train_sampler, batch_size=self.args.batch_size, drop_last=True)
        return DataLoader(self.brats_train, batch_sampler=train_bSampler, num_workers=self.args.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.brats_val, batch_size=1, num_workers=self.args.num_workers, drop_last=False)
    
    def predict_dataloader(self):
        return DataLoader(self.brats_test, batch_size=1, num_workers=self.args.num_workers, drop_last=False)
    
    class PatientSampler(Sampler):
        def __init__(self, num_patients=300, num_samples=500):
            self.num_patients = num_patients
            self.num_samples = num_samples

        def generate_iteration_list(self):
            return np.random.randint(0, self.num_patients, self.num_samples)

        def __iter__(self):
            return iter(self.generate_iteration_list())

        def __len__(self):
            return self.num_samples