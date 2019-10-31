
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import bspline.splinelab as splinelab
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin
from sklearn.linear_model import HuberRegressor

class BSplineFeatures(TransformerMixin):
    def __init__(self, knots, degree=6):
        self.bsplines = get_bspline_basis(knots, degree)
        self.nsplines = len(self.bsplines)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, nfeatures * self.nsplines))
        for ispline, spline in enumerate(self.bsplines):
            istart = ispline * nfeatures
            iend = (ispline + 1) * nfeatures
            features[:, istart:iend] = si.splev(X, spline)
        return features

def make_knots(size=30, num_knots=4, order=7):

    knots = np.round(-1+np.logspace(np.log10(1), np.log10(size-1), num_knots))
    knots = np.round(np.sqrt(knots)*(size-1)/max(np.sqrt(knots)))
    knots = np.unique(knots)
    knots = splinelab.augknt(knots, order)

    return knots

def get_bspline_basis(knots, degree=3):

    nknots = len(knots)
    ncoeffs = len(knots)+degree+1
    bsplines = []

    for ispline in range(nknots):
        coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)]
        bsplines.append((knots, coeffs, degree))

    return bsplines
