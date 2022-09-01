import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, \
    median_absolute_error, r2_score
from tqdm import tqdm
from joblib import Parallel, delayed
import os.path

def Evaluation(y: np.array, y_pred: np.array):
    error = abs(y - y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    std = np.std(y, y_pred)

    return error, mae, rmse, mape, r2, std


def AnalysisResults(results: np.array, method: str):
    """

    :param results:
    :param method:
    :return:
    """
    summarys = []
    for result in results:
        summary = []

        y_majority_train = result[1]
        y_majority_train_pred = result[2]
        y_majority_validation = result[4]
        y_majority_validation_pred = result[5]
        y_majority_test = result[7]
        y_majority_test_pred = result[8]

        match method:
            case "basis1":
                y_minority_train = np.zeros(5)
                y_minority_train_pred = np.zeros(5)
                y_minority_validation = np.zeros(5)
                y_minority_validation_pred = np.zeros(5)
                y_minority_test = result[10]
                y_minority_test_pred = result[11]
                y_train = y_majority_train
                y_train_pred = y_majority_train_pred
                y_validation = y_majority_validation
                y_validation_pred = y_majority_validation_pred
                y_test = np.append(y_majority_test, y_minority_test)
                y_test_pred = np.append(y_majority_test_pred, y_minority_test_pred)

            case _:
                y_minority_train = result[10]
                y_minority_train_pred = result[11]
                y_minority_validation = result[13]
                y_minority_validation_pred = result[14]
                y_minority_test = result[16]
                y_minority_test_pred = result[17]
                y_train = np.append(y_majority_train, y_minority_train)
                y_train_pred = np.append(y_majority_train_pred, y_minority_train_pred)
                y_validation = np.append(y_majority_validation, y_minority_validation)
                y_validation_pred = np.append(y_majority_validation_pred, y_minority_validation_pred)
                y_test = np.append(y_majority_test, y_minority_test)
                y_test_pred = np.append(y_majority_test_pred, y_minority_test_pred)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_std = np.std(abs(y_train - y_train_pred))
        train_majority_mae = mean_absolute_error(y_majority_train, y_majority_train_pred)
        train_majority_std = np.std(abs(y_majority_train - y_majority_train_pred))
        train_minority_mae = mean_absolute_error(y_minority_train, y_minority_train_pred)
        train_minority_std = np.std(abs(y_minority_train - y_minority_train_pred))
        train_var = median_absolute_error(y_train, y_train_pred)
        train_rmse = mean_squared_error(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        v_mae = mean_absolute_error(y_validation, y_validation_pred)
        v_std = np.std(abs(y_validation - y_validation_pred))
        v_majority_mae = mean_absolute_error(y_majority_validation, y_majority_validation_pred)
        v_majority_std = np.std(abs(y_majority_validation - y_majority_validation_pred))
        v_minority_mae = mean_absolute_error(y_minority_validation, y_minority_validation_pred)
        v_minority_std = np.std(abs(y_minority_validation - y_minority_validation_pred))
        v_var = median_absolute_error(y_validation, y_validation_pred)
        v_rmse = mean_squared_error(y_validation, y_validation_pred)
        v_mape = mean_absolute_percentage_error(y_validation, y_validation_pred)
        v_r2 = r2_score(y_validation, y_validation_pred)
        t_mae = mean_absolute_error(y_test, y_test_pred)
        t_std = np.std(abs(y_test - y_test_pred))
        t_majority_mae = mean_absolute_error(y_majority_test, y_majority_test_pred)
        t_majority_std = np.std(abs(y_majority_test - y_majority_test_pred))
        t_minority_mae = mean_absolute_error(y_minority_test, y_minority_test_pred)
        t_minority_std = np.std(abs(y_minority_test - y_minority_test_pred))
        t_var = median_absolute_error(y_test, y_test_pred)
        t_rmse = mean_squared_error(y_test, y_test_pred)
        t_mape = mean_absolute_percentage_error(y_test, y_test_pred)
        t_r2 = r2_score(y_test, y_test_pred)

        summary.append(train_mae)
        summary.append(train_std)
        summary.append(train_majority_mae)
        summary.append(train_majority_std)
        summary.append(train_minority_mae)
        summary.append(train_minority_std)
        summary.append(train_var)
        summary.append(train_rmse)
        summary.append(train_mape)
        summary.append(train_r2)
        summary.append(v_mae)
        summary.append(v_std)
        summary.append(v_majority_mae)
        summary.append(v_majority_std)
        summary.append(v_minority_mae)
        summary.append(v_minority_std)
        summary.append(v_var)
        summary.append(v_rmse)
        summary.append(v_mape)
        summary.append(v_r2)
        summary.append(t_mae)
        summary.append(t_std)
        summary.append(t_majority_mae)
        summary.append(t_majority_std)
        summary.append(t_minority_mae)
        summary.append(t_minority_std)
        summary.append(t_var)
        summary.append(t_rmse)
        summary.append(t_mape)
        summary.append(t_r2)

        summarys.append(summary)

    return summarys


def AnalysisSummarys(summarys: list):
    summarys = pd.DataFrame(summarys)
    df = []
    for i in range(summarys.shape[1]):
        df.append(summarys.iloc[:, i].mean())

    return df


def SaveMetrics(metrics: list, csvfile: str="summary.xlsx"):
    print_metrics = []
    for i in range(3):
        print_metric = []
        print_metric.append("%.6f ± %.6f" % (metrics[0 + 10 * i], metrics[1 + 10 * i]))
        print_metric.append("%.6f ± %.6f" % (metrics[2 + 10 * i], metrics[3 + 10 * i]))
        print_metric.append("%.6f ± %.6f" % (metrics[4 + 10 * i], metrics[5 + 10 * i]))
        print_metric.append("%.6f" % metrics[6 + 10 * i])
        print_metric.append("%.6f" % metrics[7 + 10 * i])
        print_metric.append("%.6f" % metrics[8 + 10 * i])
        print_metric.append("%.6f" % metrics[9 + 10 * i])
        print_metrics.append(print_metric)

    header = ["Training", "Validation", "Test"]
    print_metrics = pd.DataFrame(print_metrics).T
    print_metrics.to_excel(csvfile, sheet_name="summary", header=header, index=None)


def RealResults(data: pd.DataFrame, name_index: list):
    df = pd.DataFrame(data.iloc[:, 0])
    name = pd.DataFrame(df.iloc[name_index, :].reset_index(drop=True)).T
    name = name.to_numpy().flatten()

    return name


def SaveResult(results: pd.DataFrame, filename: str, iteration: int=0):
    for i in range(len(results)):
        result = pd.DataFrame(results[i]).T
        if os.path.exists(filename):
            writer = pd.ExcelWriter(path=filename, mode="a", if_sheet_exists="replace")
        else:
            writer = pd.ExcelWriter(path=filename, mode="w")
        result.to_excel(excel_writer=writer, sheet_name="iteration-"+str(iteration*25+i), header=None, index=None)    
        writer.save()
    return True


def SaveResults(results: pd.DataFrame, filename: str):
    iterations = len(results)
    if iterations > 25:
        filename = filename[:-5]
        outs = (
            Parallel(n_jobs=25)
            (delayed(SaveResult)(results[i:i+25], filename+"_"+str(i)+".xlsx", i)
            for i in tqdm(range(int(iterations/25))))
        )
    else:    
        SaveResult(results, filename)
    print("Save all results into excel files successfully.")
        
