import os
import numpy as np
from glob import glob


def get_patients(base_dir, suffix='.npy'):
    list_of_patients = glob(os.path.join(base_dir, '*' + suffix))
    list_of_patients.sort()
    return [i[:-len(suffix)] for i in list_of_patients]

def pad_image(image, new_shape=None, mode="constant"):
    """ Pad image if patch size > image size,
        leave image intact if patch size < image size"""
    old_shape = np.array(image.shape[-len(new_shape):])
    num_axes_nopad = len(image.shape) - len(new_shape)
    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]] * num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        return np.pad(image, pad_list, mode, constant_values=0)
    else:
        return image

def crop(data, seg, crop_size=128):
    data_shape = tuple([len(data)] + list(data[0].shape))
    data_dtype = data[0].dtype
    dim = len(data_shape) - 2

    seg_shape = tuple([len(seg)] + list(seg[0].shape))
    seg_dtype = seg[0].dtype
    assert all([i == j for i, j in zip(seg_shape[2:], data_shape[2:])]), "data and seg must have the same spatial " \
                                                                         "dimensions. Data: %s, seg: %s" % \
                                                                         (str(data_shape), str(seg_shape))

    crop_size = [crop_size] * dim
    data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype)
    seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype)


    for b in range(data_shape[0]):
        data_shape_here = [data_shape[0]] + list(data[b].shape)
        seg_shape_here = [[seg_shape[0]]] + list(seg[0].shape)

        lbs = []
        for i in range(len(data_shape_here) - 2):
            lbs.append(np.random.randint(0, data_shape_here[i+2] - crop_size[i]))

        ubs = [lbs[d] + crop_size[d] for d in range(dim)]

        slicer_data = [slice(0, data_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        data_cropped = data[b][tuple(slicer_data)]

        slicer_seg = [slice(0, seg_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        seg_cropped = seg[b][tuple(slicer_seg)]

        data_return[b] = data_cropped
        seg_return[b] = seg_cropped

    return data_return, seg_return