''' File to model prepayments. Here, we implement a two-covariate prepayment model following Schwartz and Torous (1989) '''
import numpy as np
from utils import *

def lamba_nut(t, p, gamma):
  '''
  :param t: time, can be a scalar of a 1D numpy array of times
  :param p, gamma: parameters (we want to fit for)
  :return: an array of same shape as t, where each element is corresponding lambda_nut
  '''
  return gamma * p * (gamma * t) ** (p - 1) / (1 + (gamma * t) ** p)

def lamda(t, v, beta, p, gamma):
  '''
  Hazard rate
  :param t:
  :param v: covariates, can be a
        * 1D np array of shape beta, e.g. [coupon gap (covariate 1), indicator variable for summer months (covariate 2)]
        * 2D np array of shape (num_obs, len(beta)), where each row is an obserbation of the covariates
  :param beta: parameter, 1D np array of shape v, coefficients on the covariates
  :param p, gamma: parameters for lambda_nut
  :return: scalar id v is 1D array, 1D array if v is 2D array (where each element corresponds to the estimated hazard
        rate for that observation)
  '''
  lambda_nut = lamba_nut(t, p, gamma)
  lam = lambda_nut * np.exp(v.dot(beta))
  return lam

def survival(t, v, beta, p, gamma):
    '''
    :param t:
    :param v: 2D np array of shape (nobs, len(beta))
    :param beta:
    :param p:
    :param gamma:
    :return: The survival  value, 1D array of shape (nobs), where each element is the estimated survival for that obs
    '''
    cov_portion = np.exp(v.dot(beta)) # depends on covariates only
    time_portion = np.log(1 + (gamma * t) ** p)
    survival = np.exp(-cov_portion * time_portion)
    return survival

def log_likelihood_schwatz_torous(t, expir_type, v, beta, p, gamma):
    '''
    :param t: 1D np.array of times when the loan has expired, or time of the end of the observation if the loan has not
        expired
    :param expir_type: 1D np.array of shape (len(t)), where the element is 1 iif the bond defaults during our observation
    :param v: a np 2D array of shape (num_obs, num_covariates), where each row is a set covariates
    :param beta: parameter to estimate (as before)
    :param p: parameter to estimate (as before)
    :param gamma: parameter to estimate (as before)
    :return: log-likelihood function to estimate a prepayment model as in Schwartz and Tourous
    '''
    hazard_part = np.log(lamda(t, v, beta, p, gamma)).dot(expir_type)
    survival_part = np.log(survival(t, v, beta, p, gamma)).sum()
    return hazard_part + survival_part