
import os
import pandas as pd, numpy as np, pickle as pkl
from collections import defaultdict
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
import splines_utils

patient_id = "*"    # patient id goes here
set = "Train"       # 'Train' or 'Test'
path = "*"          # path goes here

if set == "Test":

    files = os.listdir(path+"spectra/")
    files.sort()

    for _, val in enumerate(files):

        print("opening data: ", val)
        with open(path+"spectra/"+val, "rb") as file:
            spectra = pkl.load(file)

        normalized_spectra = spectra["normalized_psd"]

        W = np.zeros((normalized_spectra.shape[0],normalized_spectra.shape[1],1))
        H = np.zeros((normalized_spectra.shape[0],1,normalized_spectra.shape[2]))

        W_predict = np.zeros((normalized_spectra.shape[0],normalized_spectra.shape[1],1))
        H_predict = np.zeros((normalized_spectra.shape[0],1,normalized_spectra.shape[2]))

        W_coeff = np.zeros((normalized_spectra.shape[0],3))
        H_coeff = np.zeros((normalized_spectra.shape[0],24))

        dictionary = defaultdict(dict)

        for idx in range(normalized_spectra.shape[0]):

            model = NMF(n_components=1)

            w_current = model.fit_transform(X=normalized_spectra[idx, :, :])
            h_current = model.components_

            W[idx, :, :] = w_current
            H[idx, :, :] = h_current

            x_w = np.linspace(0,len(w_current),len(w_current))
            x_w_predict = np.linspace(0,len(w_current),len(w_current))

            W_model = make_pipeline(PolynomialFeatures(2), HuberRegressor())
            W_model.fit(x_w.reshape(-1,1), w_current.ravel())
            W_predict[idx,:,:] = np.expand_dims(W_model.predict(x_w_predict.reshape(-1,1)), axis=1)

            W_coeff[idx,:] = W_model.steps[1][1].coef_

            x_h = np.linspace(0,len(h_current.T),len(h_current.T))
            x_h_predict = np.linspace(0,len(h_current.T),len(h_current.T))

            knots = splines_utils.make_knots(size=H.shape[2], num_knots=10)
            bspline_features = splines_utils.BSplineFeatures(knots=knots, degree=6)

            H_model = make_pipeline(bspline_features, HuberRegressor())
            H_model.fit(x_h[:,np.newaxis], h_current.T.ravel())
            H_predict[idx,:,:] = H_model.predict(x_h_predict.reshape(-1,1))

            H_coeff[idx,:] = H_model.steps[1][1].coef_

        dictionary["W"] = W
        dictionary["H"] = H

        dictionary["W_predict"] = W_predict
        dictionary["H_predict"] = H_predict

        dictionary["W_coeff"] = W_coeff
        dictionary["H_coeff"] = H_coeff

        filename = val.split("spec")[0]+"model.pkl"

        print("saving data: ", filename)
        with open(path+"nmf_models/"+filename, "wb") as file:
            pkl.dump(dictionary, file)

else:

    labels = ['interictal', 'preictal']

    for label in labels:

        files = os.listdir(path+"spectra/"+label+"/")
        files.sort()

        if label == "interictal":
            files = files[:-1]

        for _, val in enumerate(files):

            print("opening data: ", val)

            with open(path+"spectra/"+label+"/"+val, "rb") as file:
                spectra = pkl.load(file)

            normalized_spectra = spectra["normalized_psd"]

            W = np.zeros((normalized_spectra.shape[0],normalized_spectra.shape[1],1))
            H = np.zeros((normalized_spectra.shape[0],1,normalized_spectra.shape[2]))

            W_predict = np.zeros((normalized_spectra.shape[0],normalized_spectra.shape[1],1))
            H_predict = np.zeros((normalized_spectra.shape[0],1,normalized_spectra.shape[2]))

            W_coeff = np.zeros((normalized_spectra.shape[0],3))
            H_coeff = np.zeros((normalized_spectra.shape[0],24))

            dictionary = defaultdict(dict)

            for idx in range(normalized_spectra.shape[0]):

                model = NMF(n_components=1)

                w_current = model.fit_transform(X=normalized_spectra[idx, :, :])
                h_current = model.components_

                W[idx, :, :] = w_current
                H[idx, :, :] = h_current

                x_w = np.linspace(0,len(w_current),len(w_current))
                x_w_predict = np.linspace(0,len(w_current),len(w_current))

                W_model = make_pipeline(PolynomialFeatures(2), HuberRegressor())
                W_model.fit(x_w.reshape(-1,1), w_current.ravel())
                W_predict[idx,:,:] = np.expand_dims(W_model.predict(x_w_predict.reshape(-1,1)), axis=1)

                W_coeff[idx,:] = W_model.steps[1][1].coef_

                x_h = np.linspace(0,len(h_current.T),len(h_current.T))
                x_h_predict = np.linspace(0,len(h_current.T),len(h_current.T))

                knots = splines_utils.make_knots(size=H.shape[2], num_knots=10)
                bspline_features = splines_utils.BSplineFeatures(knots=knots, degree=6)

                H_model = make_pipeline(bspline_features, HuberRegressor())
                H_model.fit(x_h[:,np.newaxis], h_current.T.ravel())
                H_predict[idx,:,:] = H_model.predict(x_h_predict.reshape(-1,1))

                H_coeff[idx,:] = H_model.steps[1][1].coef_

            dictionary["W"] = W
            dictionary["H"] = H

            dictionary["W_predict"] = W_predict
            dictionary["H_predict"] = H_predict

            dictionary["W_coeff"] = W_coeff
            dictionary["H_coeff"] = H_coeff

            filename = val.split("spec")[0]+"model.pkl"

            print("saving data: ", filename)
            with open(path+"nmf_models/"+label+"/"+filename, "wb") as file:
                pkl.dump(dictionary, file)
