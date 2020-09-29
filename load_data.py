import numpy as np
from skimage import transform, io, color
from os import walk
from os import path

def loadCovid19ClassificationData(path_covid19, path_notcovid, im_shape):
    X, Y = [], []

    i = 0
    for (cxr_dirpath, cxr_dirnames, cxr_filenames) in walk(path_covid19):
        for cxr_filename in cxr_filenames:
            cxr_fullfilepath = path.join(path_covid19, cxr_filename)
            cxr = io.imread(cxr_fullfilepath)
            cxr = transform.resize(cxr, im_shape)
            cxr = color.gray2rgb(cxr)
            label = "COVID-19"
            X.append(cxr)
            Y.append(label)
            i = i + 1
        break

    print(str(i)+" COVID-19 CXR IMAGES LOADED")

    i = 0
    for (cxr_dirpath, cxr_dirnames, cxr_filenames) in walk(path_notcovid):
        for cxr_filename in cxr_filenames:
            cxr_fullfilepath = path.join(path_notcovid, cxr_filename)
            cxr = io.imread(cxr_fullfilepath)
            cxr = transform.resize(cxr, im_shape)
            cxr = color.gray2rgb(cxr)
            label = "NÃO COVID-19"
            X.append(cxr)
            Y.append(label)
            i = i + 1
        break

    print(str(i)+" NOT COVID-19 CXR IMAGES LOADED")

    X = np.array(X)
    Y = np.array(Y)
    X -= X.mean()
    X /= X.std()

    return X, Y



