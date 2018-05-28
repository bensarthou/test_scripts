import numpy as np


def flatten_wave(x):
    """ Flatten list an array.

    Parameters
    ----------
    x: list of dict or ndarray
        the input data

    Returns
    -------
    y: ndarray 1D
        the flatten input list of array.
    shape: list of dict
        the input list of array structure.
    """

    # Flatten the dataset
    if not isinstance(x, list):
        x = [x]
    elif len(x) == 0:
        return None, None

    # Flatten the dataset
    y = x[0].flatten()
    shape_dict = [x[0].shape]
    for x_i in x[1:]:
        dict_lvl = {}
        for key in x_i.keys():
            dict_lvl[key] = x_i[key].shape
            y = np.concatenate((y, x_i[key].flatten()))
        shape_dict.append(dict_lvl)

    return y, shape_dict


def unflatten_wave(y, shape):
    """ Unflatten a flattened array.

    Parameters
    ----------
    y: ndarray 1D
        a flattened input array.
    shape: list of dict
        the output structure information.

    Returns
    -------
    x: list of ndarray
        the unflattened dataset.
    """
    # Unflatten the dataset
    start = 0
    stop = np.prod(shape[0])
    x = [y[start:stop].reshape(shape[0])]
    offset = stop
    for shape_i in shape[1:]:
        sublevel = {}
        for key in shape_i.keys():
            start = offset
            stop = offset + np.prod(shape_i[key])
            offset = stop
            sublevel[key] = y[start: stop].reshape(shape_i[key])
            print(sublevel.keys())
        x.append(sublevel)
    return x


if __name__ == '__main__':

    test_dict = np.load('/volatile/bsarthou/datas/flatten/coeffs_dict_1.npy')
    res_list_old = np.load('/volatile/bsarthou/datas/flatten/res_list_2.npy')

    res_list, s = flatten_wave(list(test_dict))
    list_dict = list(test_dict)
    unflat = unflatten_wave(res_list_old, s)

    for i in range(1, len(list_dict)):
        for k in list_dict[i].keys():
            print((list_dict[i][k] - unflat[i][k]).sum())
