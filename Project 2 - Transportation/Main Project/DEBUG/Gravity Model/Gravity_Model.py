import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


class GM:
    def __init__(self) -> None:
        """
        This class will find the trips with GM method.
        """
        self.mse_t = None
        self.mae_t = None
        self.r2 = None
        self.rmse = None
        self.mse = None
        self.mae = None
        self.real_od_shaped = None
        self.predict_od_shaped = None
        self.real_od_matrix = None
        self.od_matrix_predicted = None
        self.od_matrix_calibrated = None
        self.od_matrix_init = None
        self.parameter_f_final = None
        self.parameter_f_init = None
        self.approach = None
        self.attraction = None
        self.production = None
        self.travel_time = None
        self.show_od = None
        self.show_everything = None
        self.show_f = None

    def _guess_od_matrix(self) -> None:
        """
        This matrix trying to guess the results.
        """
        for zone in self.od_matrix_calibrated.index:
            od_temp = self.od_matrix_init.copy()
            od_temp.replace("False", 0, inplace=True)
            od_temp = od_temp.astype(float)
            remained_prod = self.production[zone] - sum(od_temp.loc[zone])

            column_index = []
            attraction_sum = 0
            column_attraction = []
            for i in self.od_matrix_init.columns:
                if self.od_matrix_init.loc[zone, i] == "False":
                    column_index.append(i)

            for i in column_index:
                attract = self.attraction[i]
                column_attraction.append(attract)
                attraction_sum += attract

            ratio = np.array(column_attraction) / attraction_sum
            replacing_num = remained_prod * ratio
            self.od_matrix_calibrated.loc[zone, column_index] = replacing_num

    def _guess_time_scale(self, time_period: int) -> int:
        """
        This function will split data.
        :param time_period: How much time the period will take.
        :return: The number of periods.
        """
        travel_time_matrix = self.travel_time.copy().values
        max_item = max(max(travel_time_matrix, key=lambda x: max(x)))
        min_item = min(min(travel_time_matrix, key=lambda x: min(x)))
        n_period = int(np.ceil((max_item - min_item) / time_period))
        self.period = []
        for i in range(n_period + 1):
            self.period.append(i * time_period + min_item)

        if self.show_everything:
            print("---------------------------------------------------------------")
            print(f"you chose   {n_period}   period to be made, which means we have   {n_period + 1}   bins:")
            print(f"===>   {self.period}   <===")
            print()

        return n_period

    def _guess_find_time_dependent_origin(self) -> None:
        """
        It will categorize every cell according to the split that we made.
        :return: It makes the parameter self.dependent_origin.
        """
        dependent_origin = self.od_matrix_calibrated.copy()

        for i in range(len(self.period) - 1):
            if i == len(self.period) - 2:
                for j in self.travel_time.index:
                    for k in self.travel_time.columns:
                        if self.period[i] <= self.travel_time.loc[j, k] <= self.period[i + 1]:
                            dependent_origin.loc[j, int(k)] = i
            else:
                for j in self.travel_time.index:
                    for k in self.travel_time.columns:
                        if self.period[i] <= self.travel_time.loc[j, k] < self.period[i + 1]:
                            dependent_origin.loc[j, int(k)] = i

        self.dependent_origin = dependent_origin.astype(int)

        if self.show_everything:
            print("---------------------------------------------------------------")
            print("dependent_origin according to the previews split...")
            print("pay attention: every list has two value.")
            print("one: address of the node")
            print("two: value of it")
            print(f"===>   {dependent_origin}   <===")
            print()

    def _guess_sigma_a_f(self, i) -> float:
        """
        Here we calculate the Denominator
        :param i: i of the node
        :return: Denominator
        """
        temp = 0
        for j in range(len(self.attraction)):
            temp += self.attraction.iloc[j] * self.parameter_f_init[self.dependent_origin.iloc[i, j]]
        return temp

    def _guess_t(self, i, j) -> float:
        """
        Here we calculate cells
        :param i: i of the node
        :param j: j of the node
        :return: The cell value
        """
        result = (self.production.iloc[i] * self.attraction.iloc[j] * self.parameter_f_init[
            self.dependent_origin.iloc[i, j]]) / self._guess_sigma_a_f(i)
        return result

    def _guess_update_f(self) -> None:
        """
        It will update the new f as parameter_f_new.
        Also, if you want, it will print for you all f steps.
        """
        for i_period in range(len(self.period) - 1):
            mask = self.dependent_origin == i_period
            temp = sum(self.od_matrix_init[mask])
            self.parameter_f_final[i_period] = self.parameter_f_init[i_period] * (temp / self.g_list[i_period])

        if self.show_f or self.show_everything:
            print("---------------------------------------------------------------")
            print("The value of all f in every step in the function is:")
            print("===>   ", end="")
            for i in range(len(self.parameter_f_final)):
                if i + 1 == 1:
                    print(f"your {i + 1}st f is {self.parameter_f_final[i]:.5f}   ", end="")
                elif i + 1 == 2:
                    print(f"your {i + 1}ed f is {self.parameter_f_final[i]:.5f}   ", end="")
                else:
                    print(f"your {i + 1}th f is {self.parameter_f_final[i]:.5f}   ", end="")
            print("   <===")
            print()

    def _guess_check_parameter_f(self) -> float:
        """
        It will check the difference of new_f and the old one (sum of difference of all f)
        :return: Error
        """
        temp = 0
        for i in range(len(self.parameter_f_init)):
            temp += abs(self.parameter_f_init[i] - self.parameter_f_final[i])
        return temp

    def _guess_new_od_update(self, error, _s=2) -> None:
        """
        It will update the od matrix according to correction of rows and columns sum.
        :param error: How much the error of od_matrix can be.
        """
        while _s > error:
            _s = 0
            new_a = self.od_matrix_predicted.copy().sum(axis=0)
            for i in self.attraction.index:
                temp = (self.attraction[i] / new_a[i])
                self.od_matrix_predicted.loc[:, i] = self.od_matrix_predicted.loc[:, i].apply(lambda x: temp * x)
                _s += abs(temp - 1)

            new_p = self.od_matrix_predicted.sum(axis=1)
            for i in self.production.index:
                temp = self.production.loc[i] / new_p.loc[i]
                self.od_matrix_predicted.loc[i, :] = self.od_matrix_predicted.loc[i, :].apply(lambda x: temp * x)
                _s += abs(temp - 1)

    def _guess_main(self, error_f, error_od, iteration_max) -> None:
        """
        Here we perform the gravity model according to guess strategy.
        :param error_f: How much the error of f can be.
        :param error_od: How much error of od_matrix can be.
        :param iteration_max: How many times to continue at most?
        :return: It will produce the new od and set it as od_matrix_final
        """
        iteration = 0
        while self._guess_check_parameter_f() > error_f and iteration < iteration_max:
            if sum(self.parameter_f_final) < 0.5:
                raise TimeoutError("it will not converge. Sorry! :(")
            iteration += 1
            self.parameter_f_init = self.parameter_f_final.copy()
            g_list = []
            od = self.od_matrix_predicted.copy()

            for i in range(len(od.index)):
                for j in range(len(od.columns)):
                    self.od_matrix_predicted.iloc[i, j] = self._guess_t(i, j)

            for k in range(len(self.period) - 1):
                mask = self.dependent_origin == k
                g_list.append(sum(self.od_matrix_predicted[mask]))

            self.g_list = g_list.copy()
            self._guess_update_f()

        self._guess_new_od_update(error_od)

        if self.show_od or self.show_everything:
            print("---------------------------------------------------------------")
            print("here is what we guess at first")
            print(self.od_matrix_calibrated)
            print()
            print(f"and here is the new OD matrix after {iteration} step")
            print(self.od_matrix_predicted)

    def _least_square(self) -> None:
        """
        :return: It will set everything for you.
        """
        if self.approach == "exp":
            self._model_exp()
        elif self.approach == "power":
            self._model_power()
        elif self.approach == "tanner":
            self._model_tanner()

    def _model_exp(self) -> None:
        """
        Method of exponential.
        :return: t(i,j) = beta * P(i) * A(j) * exp( -u * C(i,j) )
        """
        mask = ~self.od_matrix_init.isin(["False", "No_connection"])

        y_list = np.log(
            self.od_matrix_init.values[mask.values].astype(float) /
            (self.attraction.values[:] * self.production.values[:, None])[mask])
        x_list = self.travel_time.values[mask]

        slop, constant = self._least_square_calculator(x_list, y_list)
        beta = np.exp(constant)
        u = -slop

        mask = self.od_matrix_init == "False"
        t = beta * self.production[:].values[:, np.newaxis] * self.attraction[:].values[np.newaxis, :] * np.exp(
            -u * self.travel_time.values)
        self.od_matrix_calibrated.iloc[:, :] = np.where(mask, t, self.od_matrix_init.values)

    def _model_power(self) -> None:
        """
        
        Method of power.
        :return: t(i,j) = beta * P(i) * A(j) * ( C(i,j) ^ (-u) )
        """
        mask = ~self.od_matrix_init.isin(["False", "No_connection"])
        y_list = np.log(
            self.od_matrix_init.values[mask].astype(float) /
            (self.attraction.values[:] * self.production.values[:, None])[mask])
        x_list = np.log(self.travel_time.values[mask])

        slope, constant = self._least_square_calculator(x_list, y_list)
        beta = np.exp(constant)
        u = -slope
        mask = self.od_matrix_init == "False"
        t = (beta * self.production[:].values[:, np.newaxis] * self.attraction[:].values[np.newaxis, :]
             * self.travel_time.values ** -u)
        self.od_matrix_calibrated.iloc[:, :] = np.where(mask, t, self.od_matrix_init.values)

    def _model_tanner(self) -> None:
        """
        Method of tanner.
        :return: t = P(i) * A(j) * [( C(i,j) ) ^ (-beta)] * exp( -u * C(i,j) )
        """
        mask = ~self.od_matrix_init.isin(["False", "No_connection"])
        y_list = (np.log(
            self.od_matrix_init.values[mask].astype(float) /
            (self.attraction.values[:] * self.production.values[:, None])[mask])) / np.log(
            self.travel_time.values[mask])
        x_list = self.travel_time.values[mask] / np.log(self.travel_time.values[mask])

        slope, constant = self._least_square_calculator(x_list, y_list)
        u = -slope
        beta = -constant

        mask = self.od_matrix_init == "False"
        t = self.production[:].values[:, np.newaxis] * self.attraction[:].values[np.newaxis, :] * (
                self.travel_time.values ** (-beta)) * np.exp(-u * self.travel_time.values)
        self.t = t
        self.od_matrix_calibrated.iloc[:, :] = np.where(mask, t, self.od_matrix_init.values)

    @staticmethod
    def _least_square_calculator(x: np.array, y: np.array) -> list:
        """
        Calculating the least square of 2 np.array.
        :param x: First array.
        :param y: Second array.
        :return: List of [slope, constant]
        """
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_dev = x - x_mean
        y_dev = y - y_mean
        prod_sum = np.sum(x_dev * y_dev)
        x_dev_square = np.sum(x_dev ** 2)
        slope = prod_sum / x_dev_square
        constant = y_mean - slope * x_mean
        return [slope, constant]

    def _repr_matrix(self) -> None:
        """
        It represents the initial matrix and the final one.
        """
        print("not calibrate")
        print(self.od_matrix_init)
        print("Calibrated OD_matrix:")
        print(self.od_matrix_calibrated)
        if self.approach == "guess":
            print("Predicted od_matrix:")
            print(self.od_matrix_predicted)

    def _save_results_file(self, folder: str) -> None:
        """
        This function will export errors in csv file.
        :param folder: the folder you want to store the data.
        """
        file_name = f"GM_{self.approach}.csv"
        if not os.path.exists(folder):
            os.makedirs(folder)

        if os.path.exists(f"{folder}/{file_name}"):
            results = pd.read_csv(f"{folder}/{file_name}")
            results = results.iloc[:, 1:]
            new_data = pd.DataFrame({"RMSE": [self.rmse], "MAE": [self.mae], "R^2": [self.r2]})
            results = pd.concat([results, new_data], ignore_index=True)
            results.to_csv(f"{folder}/{file_name}")

        else:
            results = pd.DataFrame({
                "RMSE": [self.rmse],
                "MAE": [self.mae],
                "R^2": [self.r2]})
            results.to_csv(f"{folder}/{file_name}")

    def _save_data(self, folder, miss):
        if not os.path.exists(f"{folder}/data_GM_{self.approach}.csv"):
            data_test = pd.DataFrame({"Predict1": self.predict_od_shaped, "Real1": self.real_od_shaped})
            data_test.to_csv(f"{folder}/data_GM_{self.approach}.csv")
        else:
            data_last = pd.read_csv(f"{folder}/data_GM_{self.approach}.csv")
            data_last = data_last.iloc[:, 1:]
            data_last[f"Predict{int(miss * 10)}"] = self.predict_od_shaped
            data_last[f"Real{int(miss * 10)}"] = self.real_od_shaped
            data_last.to_csv(f"{folder}/data_GM_{self.approach}.csv")

    def _error(self, real_od: pd.DataFrame, od_test: pd.DataFrame, print_it: bool = True) -> None:
        """
        Calculating Errors. [ MAE, MSE, RMSE, R2 ]
        :param real_od: The Real_OD_Matrix.
        :param print_it: If you don't want anything to be printed, False it.
        :return: MAE / MSE / RMSE / R2
        """
        mask = ~(od_test.isin(["False", "No_connection"]))

        self.predict_od_shaped = self.od_matrix_predicted.values[mask]
        self.real_od_shaped = real_od.values[mask]
        # self.mae = np.abs(self.predict_od_shaped - self.real_od_shaped)
        # self.mae_t = sum(self.mae)
        # self.mse = np.power(self.predict_od_shaped - self.real_od_shaped, 2) / len(self.predict_od_shaped)
        # self.mse = self.mse.astype(float)
        # self.mse_t = sum(self.mse)
        # self.rmse = np.sqrt(self.mse_t)
        self.mae = mean_absolute_error(self.real_od_shaped, self.predict_od_shaped)
        self.mse = mean_squared_error(self.real_od_shaped, self.predict_od_shaped, squared=False)
        self.rmse = self.mse ** 0.5

        self.r2 = r2_score(self.real_od_shaped, self.predict_od_shaped)

        if print_it:
            print("_________________")
            print(f" MAE Error is {self.mae:.3f}")
            print(f" MSE Error is {self.mse:.3f}")
            print(f"RMSE Error is {self.rmse:.3f}")
            print(f"R2 Error is {self.r2:.2f}")
            print("_________________")

    def _fit(self, production: pd.DataFrame, attraction: pd.DataFrame, travel_time: pd.DataFrame,
             od_matrix: pd.DataFrame, approach: str, scale_time_period: int = 5,
             show_f_and_number_of_attempt: bool = False, show_every_parameter: bool = False, show_od: bool = True,
             error_f: float = 0.01,
             error_od: float = 0.01, iteration_max: int = 100):
        """
        This function will fit the data for you and will make all these parameters which can be used if needed.
        According to the approach you chose, it will calibrate the miss_cells_od_matrix for you and will save the final
        one to the parameter od_matrix_final.

        :param production: Panda Series, which shows the production matrix. With only one column for it. No column index
        :param attraction: Panda Series, which shows the attraction matrix. With only one column for it. No column index
        :param travel_time: DataFrame shows the travel_time. Rows and columns should be named with (int) Same as OD
        :param od_matrix: It will be replaced with nd.Array... index from 0.
        :param approach:What is the approach to work with? guess or exp or power or tanner
        :param scale_time_period: Optional. Its default is 30min period. Only for guess approach.
        :param show_f_and_number_of_attempt: If you want to get a print of your f. Only for guess approach.
        :param show_every_parameter: If you want to print all steps, turn ir to True. For all approaches.
        :param show_od: It will only show the initial and final OD. For all approaches.
        :param error_od: How much error for od is validated. Only for guess approach.
        :param error_f: How much error for f is validated. Only for guess approach.
        :param iteration_max: Limit for iteration. Only for guess approach.
        :return: Final calibrated od_matrix.
        """
        self.show_f = show_f_and_number_of_attempt
        self.show_everything = show_every_parameter
        self.show_od = show_od

        self.travel_time = travel_time
        self.production = production
        self.attraction = attraction
        self.approach = approach

        if self.approach == "guess":
            self.od_matrix_init = od_matrix.copy()
            self.od_matrix_calibrated = od_matrix.copy()
            self._guess_od_matrix()
            self.od_matrix_predicted = self.od_matrix_calibrated.copy()
            n_period = self._guess_time_scale(scale_time_period)
            self.parameter_f_init = [0 for _ in range(n_period)]
            self.parameter_f_final = [1 for _ in range(n_period)]
            self._guess_find_time_dependent_origin()
            self._guess_main(error_f, error_od, iteration_max)

        elif self.approach in ["exp", "power", "tanner"]:
            self.od_matrix_init = od_matrix.copy()
            self.od_matrix_calibrated = od_matrix.copy()
            self._least_square()
            self.od_matrix_predicted = self.od_matrix_calibrated.copy()

        else:
            raise Exception("You must enter valid approach. guess / exp / power / tanner")

        if self.show_everything or self.show_od:
            self._repr_matrix()

    def pass_data_from_folder_not_scaled(self, folder: str, missing_rate: list, approach: str,
                                         scale_time_period: int = 5,
                                         show_f_and_number_of_attempt: bool = False, show_every_parameter: bool = False,
                                         show_od: bool = False, error_f: float = 0.01, error_od: float = 0.01,
                                         iteration_max: int = 100, print_error: bool = False,
                                         model_of_plot: list = None,
                                         save_plot: bool = True, show_plot: bool = False, saved_plot_folder: str = None,
                                         save_data: bool = True, saved_data_folder: str = None) -> None:
        """
        If you have your data already, this function will read and fit the data for you.

        :param folder: Name of the destination folder to read the data from.
        :param missing_rate: missing rate list.
        :param approach: What is the approach to work with? guess or exp or power or tanner
        :param scale_time_period: Optional. Its default is 30min period. Only for guess approach.
        :param show_f_and_number_of_attempt: If you want to get a print of your f. Only for guess approach.
        :param show_every_parameter: If you want to print all steps, turn ir to True. For all approaches.
        :param show_od: It will only show the initial and final OD. For all approaches.
        :param error_od: How much error for od is validated. Only for guess approach.
        :param error_f: How much error for f is validated. Only for guess approach.
        :param iteration_max: Limit for iteration. Only for guess approach.
        :param print_error: If you want us to print error for you.
        :param model_of_plot: What kind of plot you want to be plotted? Write them into one list: [MAE, MSE, RMSE, R2]
        :param save_plot: If you want to save these plots, enter True.
        :param show_plot: If you want to see the plot, enter True.
        :param saved_plot_folder: Name of the folder to save plots to.
        :param save_data: If you want to save the data or not
        :param saved_data_folder: The folder you want to store the data.
        :return: Final calibrated od_matrix.
        """
        if model_of_plot is None:
            model_of_plot = ["R2", "MSE", "RMSE", "MAE", "Hist"]
        for i in missing_rate:
            od = pd.read_csv(f"{folder}/train_data/at_miss{i:.2f}_train_od_matrix.csv", encoding='latin-1')
            od.index = od.iloc[:, 0]
            od = od.iloc[:, 1:]
            od.columns = od.columns.astype(int)
            od = od.applymap(lambda x: float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x)

            test_od = pd.read_csv(f"{folder}/test_data/at_miss{i:.2f}_test_od_matrix.csv", low_memory=False)
            test_od.index = test_od.iloc[:, 0]
            test_od = test_od.iloc[:, 1:]
            test_od.columns = test_od.columns.astype(int)
            test_od = test_od.applymap(
                lambda x: float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x)

            production = pd.read_csv(f"{folder}/production.csv")
            production.index = production.iloc[:, 0]
            production = production.iloc[:, 1:]
            production.columns = [1]
            production = production.squeeze()

            attraction = pd.read_csv(f"{folder}/attraction.csv")
            attraction.index = attraction.iloc[:, 0]
            attraction = attraction.iloc[:, 1:]
            attraction = attraction.squeeze()

            od_real = pd.read_csv(f"{folder}/real_od_matrix.csv")
            od_real.index = od_real.iloc[:, 0]
            od_real = od_real.iloc[:, 1:]
            od_real.columns = od_real.columns.astype(int)
            self.real_od_matrix = od_real.astype(float)

            travel_time = pd.read_csv(f"{folder}/travel_time_matrix.csv")
            travel_time.index = travel_time.iloc[:, 0]
            travel_time = travel_time.iloc[:, 1:]

            self._fit(production, attraction, travel_time, od, approach=approach, scale_time_period=scale_time_period,
                      show_f_and_number_of_attempt=show_f_and_number_of_attempt,
                      show_every_parameter=show_every_parameter,
                      show_od=show_od, error_od=error_od, error_f=error_f, iteration_max=iteration_max)

            self._error(real_od=self.real_od_matrix, od_test=test_od, print_it=print_error)
            self.plotting_configuration(model=model_of_plot, save=save_plot, show=show_plot, missing_rate=i,
                                        folder=saved_plot_folder)
            if save_data:
                self._save_results_file(folder=saved_data_folder)
                self._save_data(folder=saved_data_folder, miss=i)

    def pass_data_from_folder_scaled(self, folder: str, missing_rate: list, approach: str, scale_time_period: int = 5,
                                     show_f_and_number_of_attempt: bool = False, show_every_parameter: bool = False,
                                     show_od: bool = False, error_f: float = 0.01, error_od: float = 0.01,
                                     iteration_max: int = 100, print_error: bool = False, model_of_plot: list = None,
                                     save_plot: bool = True, show_plot: bool = False, saved_plot_folder: str = None,
                                     save_data: bool = True, saved_data_folder: str = None, bins=None):
        """
        If you have your scaled data already, this function will read and fit the data for you.

        :param folder: Name of the destination folder to read data from.
        :param missing_rate: missing rate list.
        :param approach:What is the approach to work with? [ guess, exp, power, tanner ]
        :param scale_time_period: Optional. Its default is 5min period. Only for guess approach.
        :param show_f_and_number_of_attempt: If you want to get a print of your f. Only for guess approach.
        :param show_every_parameter: If you want to print all steps, turn ir to True. For all approaches.
        :param show_od: It will only show the initial and final OD. For all approaches.
        :param error_od: How much error for od is validated. Only for guess approach.
        :param error_f: How much error for f is validated. Only for guess approach.
        :param iteration_max: Limit for iteration. Only for guess approach.
        :param print_error: If you want us to print error for you.
        :param model_of_plot: What kind of plot you want to be plotted? Write them into one list: [MAE, MSE, RMSE, R2]
        :param save_plot: If you want to save these plots, enter True.
        :param show_plot: If you want to see the plot, enter True.
        :param saved_plot_folder: Name of the folder to save plots to.
        :param save_data: If you want to save the data or not
        :param saved_data_folder: The folder you want to store the data.
        :param bins: in histogram that will be plotted, what is the range of x_axis.
        :return: Final calibrated od_matrix.
        """
        if model_of_plot is None:
            model_of_plot = ["R2", "MSE", "RMSE", "MAE", "Hist"]
        for i in missing_rate:
            sub_folder = int(i * 100)
            od = pd.read_csv(f"{folder}/{sub_folder}/at_miss{i:.2f}_train_od_matrix.csv", low_memory=False)
            od.index = od.iloc[:, 0]
            od = od.iloc[:, 1:]
            od.columns = od.columns.astype(int)
            od = od.applymap(lambda x: float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x)
            self.test = od

            test_od = pd.read_csv(f"{folder}/{sub_folder}/at_miss{i:.2f}_test_od_matrix.csv", low_memory=False)
            test_od.index = test_od.iloc[:, 0]
            test_od = test_od.iloc[:, 1:]
            test_od.columns = test_od.columns.astype(int)
            test_od = test_od.applymap(
                lambda x: float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x)

            production = pd.read_csv(f"{folder}/{sub_folder}/production.csv")
            production.index = production.iloc[:, 0]
            production = production.iloc[:, 1:]
            production.columns = [1]
            production = production.squeeze()

            attraction = pd.read_csv(f"{folder}/{sub_folder}/attraction.csv")
            attraction.index = attraction.iloc[:, 0]
            attraction = attraction.iloc[:, 1:]
            attraction = attraction.squeeze()

            od_real = pd.read_csv(f"{folder}/{sub_folder}/real_od_matrix.csv")
            od_real.index = od_real.iloc[:, 0]
            od_real = od_real.iloc[:, 1:]
            od_real.columns = od_real.columns.astype(int)
            self.real_od_matrix = od_real.astype(float)

            travel_time = pd.read_csv(f"{folder}/{sub_folder}/travel_time_matrix.csv")
            travel_time.index = travel_time.iloc[:, 0]
            travel_time = travel_time.iloc[:, 1:]

            self._fit(production, attraction, travel_time, od, approach=approach, scale_time_period=scale_time_period,
                      show_f_and_number_of_attempt=show_f_and_number_of_attempt,
                      show_every_parameter=show_every_parameter,
                      show_od=show_od, error_od=error_od, error_f=error_f, iteration_max=iteration_max)

            self._error(real_od=self.real_od_matrix, od_test=test_od, print_it=print_error)
            self.plotting_configuration(model=model_of_plot, save=save_plot, show=show_plot, missing_rate=i,
                                        folder=saved_plot_folder, bins=bins)
            if save_data:
                self._save_results_file(folder=saved_data_folder)
                self._save_data(folder=saved_data_folder, miss=i)

    def plotting_configuration(self, model: list, save: bool = False, show: bool = True, missing_rate: int = 0,
                               folder: str = None, bins=None) -> None:
        """
        This function will plot the error of real and predicted data together to show the difference.
        It will be stored if you want in folders with the same name as the model.
        :param model: What kind of plot you want to be plotted? Write them into one list: [MAE, MSE, RMSE, R2]
        :param save: If you want to save these plots, enter True.
        :param show: if you want to see the plot, enter True.
        :param missing_rate: if you want to save, enter the missing rate.
        :param folder: Name of the folder to save to.
        :param bins: What range the histogram should be at.
        :return: Plots
        """
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except:
            print(f"{folder} already existed.")

        if "R2" in model:
            fig_r2 = plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(self.real_od_shaped, self.predict_od_shaped)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Scatter Plot of Actual vs Predicted Values (R2 score: {self.r2:.2f})')
            x_min, x_max = plt.xlim()
            y_min, y_max = plt.ylim()
            plt.plot([x_min, x_max], [y_min, y_max], ls="--", c=".3")
            plt.subplot(1, 2, 2)
            plt.hist(self.predict_od_shaped - self.real_od_shaped, bins=bins)
            plt.xlabel('Predict - Real')
            plt.ylabel('#')
            plt.title(f'Histogram for diversity')
            if save:
                directory = f"{folder}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                fig_r2.savefig(f"{folder}/R2-{self.approach}Method-{missing_rate}missing-rate.png")
            if show:
                plt.show()
            else:
                plt.close()
