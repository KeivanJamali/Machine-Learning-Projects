import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score


# noinspection PyTypedDict,PyTypeChecker
class GreenShield_EXP:
    def __init__(self, k_j, u, r, q):
        self.predicted_q = None
        self.parameters = {
            "a": None,
            "b": None,
            "c": None
        }
        self.Error = {
            "MSE": None,
            "R2": None
        }
        self.k_j = k_j
        self.u = u
        self.r = r
        self.q = q

    def fit(self, p0=None) -> str:
        """
        params:
        p0: list = initial values - must be 3
        returns: some messages about the result.
        """
        if p0 is None:
            p0 = [0, 0, 0]
        try:
            fit_results = curve_fit(self._greenshield_exp, [self.k_j, self.u, self.r], self.q, p0=p0, full_output=True)
            msg = fit_results[3]
            self.parameters["a"] = fit_results[0][0]
            self.parameters["b"] = fit_results[0][1]
            self.parameters["c"] = fit_results[0][2]

            self.predicted_q = self._greenshield_exp([self.k_j, self.u, self.r], self.parameters["a"],
                                                     self.parameters["b"], self.parameters["c"])
            return msg
        except:
            return "[ERROR] There is something wrong in fitting!"

    @staticmethod
    def _greenshield_exp(x, a: float, b: float, c: float) -> float:
        """
        params:
        x: list: include [k_j, u, r]
        k_j: float: Jam density of the traffic flow.
        u: float: Space mean speed of the traffic flow. (km/hour)
        r: float: Rainfall intensity. (mm)

        a: float: calibration parameter of model.
        b: float: calibration parameter of model.
        c: float: calibration parameter of model.
        returns: q: flow: Traffic flow. (veh/hour)
        """
        q = x[0] * (x[1] - (x[1] ** 2) * (np.exp(a * (x[2] ** b) - c)))
        return q

    def plot(self):
        plt.scatter(self.u, self.q, label='Data')
        plt.plot(self.u, self._greenshield_exp([self.k_j, self.u, self.r], self.parameters["a"], self.parameters["b"],
                                               self.parameters["c"]), color='red', label='Fit')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def get_results(self):
        print("Fitted Parameters:")
        print("a =", self.parameters["a"], end="  ===  ")
        print("b =", self.parameters["b"], end="  ===  ")
        print("c =", self.parameters["c"])
        print("____________________________________________________")
        print("Error Parameters:")
        print("R2 =", self.Error["R2"], end="  ===  ")
        print("MSE =", self.Error["MSE"])

    def error(self, metric=None):
        if metric is None:
            metric = ["R2", "MSE"]
        for i in metric:
            if i in "R2":
                self.Error["R2"] = self._r2_score()
            if i in "MSE":
                self.Error["MSE"] = self._mse_score()

    def _r2_score(self):
        return r2_score(self.q, self.predicted_q)

    def _mse_score(self):
        return mean_squared_error(self.q, self.predicted_q)
