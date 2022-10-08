import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

directory = "/".join(__file__.split("/")[:-1])

# Uncertainty Calculations Based on Documentation for U1270 Series Handheld
# Digital Multimeters


def calculate_uncertainty(value, res, percentage, multiplier):
    return value * percentage / 100 + res * multiplier


# Model Functions.
def linear_model(values, a, b):
    return a * values + b


def nonlinear_model(values, a, b):
    return a * values ** b


def error_propagation_division(value1, value2, uncertainty):
    return value1 * uncertainty / value2


def error_propagation_exponential(value, uncertainty):
    return value * uncertainty


def model_light(a, values):
    return a * values ** (3 / 5)


def calculatechisquared(
    xvalues, yvalues, yuncertainty, model, *modelparameters
):
    return (1 / (len(xvalues) - len(modelparameters))) * np.sum(
        ((yvalues - model(xvalues, *modelparameters)) / yuncertainty) ** 2
    )
