import numpy as np
import SimpleITK as sitk
import os
import pickle
from multiprocessing import Pool

from utils.args import get_main_args


def get_list_of_files(base_dir):
    list_of_lists = []
    patients = sorted(os.listdir(base_dir))[:-2]
    patients.pop(354) # damaged file
    for p in patients:
        patient_directory = os.path.join(base_dir, p)
        t1_file = os.path.join(patient_directory, p + "_t1.nii")
        t1ce_file = os.path.join(patient_directory, p + "_t1ce.nii")
        t2_file = os.path.join(patient_directory, p + "_t2.nii")
        flair_file = os.path.join(patient_directory, p + "_flair.nii")
        seg_file = os.path.join(patient_directory, p + "_seg.nii")
        this_case = [t1_file, t1ce_file, t2_file, flair_file, seg_file]
        assert all((os.path.isfile(i) for i in this_case)), "some file is missing for patient %s;" \
                                                            "make sure the following files are there: %s" % (
                                                            p, str(this_case))
        list_of_lists.append(this_case)
    print("Found %d patients" % len(list_of_lists))
    return list_of_lists

def load_and_preprocess(case, patient_name, output_folder):
    # load SimpleITK files
    imgs_sitk = [sitk.ReadImage(i) for i in case]

    # get arrays from SimpleITK files
    imgs_npy = [sitk.GetArrayFromImage(i) for i in imgs_sitk]

    # get some metadata
    spacing = imgs_sitk[0].GetSpacing()
    spacing = np.array(spacing)[::-1]

    direction = imgs_sitk[0].GetDirection()
    origin = imgs_sitk[0].GetOrigin()

    original_shape = imgs_npy[0].shape

    # stack images to 4d array
    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)

    # crop nonzero region
    nonzero = [np.array(np.where(i != 0)) for i in imgs_npy]
    nonzero = [[np.min(i, 1), np.max(i, 1)] for i in nonzero]
    nonzero = np.array([np.min([i[0] for i in nonzero], 0), np.max([i[1] for i in nonzero], 0)]).T

    imgs_npy = imgs_npy[:,
               nonzero[0, 0]: nonzero[0, 1] + 1,
               nonzero[1, 0]: nonzero[1, 1] + 1,
               nonzero[2, 0]: nonzero[2, 1] + 1,
               ]

    nonzero_masks = [i != 0 for i in imgs_npy[:-1]]
    brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
    for i in range(len(nonzero_masks)):
        brain_mask = brain_mask | nonzero_masks[i]

    # normalize brain region
    for i in range(len(imgs_npy) - 1):
        mean = imgs_npy[i][brain_mask].mean()
        std = imgs_npy[i][brain_mask].std()
        imgs_npy[i] = (imgs_npy[i] - mean) / (std + 1e-8)
        imgs_npy[i][brain_mask == 0] = 0

    # fix labels
    imgs_npy[-1][imgs_npy[-1] == 4] = 3

    # save images as npz
    np.save(os.path.join(output_folder, patient_name + ".npy"), imgs_npy)

    # save metadata as .pkl
    metadata = {
        'spacing': spacing,
        'direction': direction,
        'origin': origin,
        'original_shape': original_shape,
        'nonzero_region': nonzero
    }

    with open(os.path.join(output_folder, patient_name + ".pkl"), 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    args = get_main_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    list_of_lists = get_list_of_files(args.input_dir)[:-68] # Preprocessing only 300 MRI scans due to Storage Capacity limitations
    patient_names = ['_'.join(i[0].split("/")[-1].split("_")[:-1]) for i in list_of_lists]

    p = Pool()
    p.starmap(load_and_preprocess, zip(list_of_lists, patient_names, [args.output_dir] * len(list_of_lists)))
    p.close()
    p.join()
