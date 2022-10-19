from email import iterators
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, \
    median_absolute_error, r2_score
from tqdm import tqdm
from joblib import Parallel, delayed
import os.path, time

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
        train_majority_rmse = mean_squared_error(y_majority_train, y_majority_train_pred)
        train_minority_rmse = mean_squared_error(y_minority_train, y_minority_train_pred)
        v_majority_rmse = mean_squared_error(y_majority_validation, y_majority_validation_pred)
        v_minority_rmse = mean_squared_error(y_minority_validation, y_minority_validation_pred)
        t_majority_rmse = mean_squared_error(y_majority_test, y_majority_test_pred)
        t_minority_rmse = mean_squared_error(y_minority_test,y_minority_test_pred)

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
        summary.append(train_majority_rmse)
        summary.append(train_minority_rmse)
        summary.append(v_majority_rmse)
        summary.append(v_minority_rmse)
        summary.append(t_majority_rmse)
        summary.append(t_minority_rmse)

        summarys.append(summary)

    return summarys


def SaveHyperparameterMetrics(op_results: np.array, op_object: str = "C", filename: str = "metrics.xlsx", save_result: bool=True):
    op_metrics = []
    for op_result in op_results:
        result_filename = ""
        op_objects = HandleKeyValue(op_result)
        method = str(op_objects["method"][0])
        cv_method = str(op_objects["cv_method"][0])
        sampling_method = str(op_objects["sampling_method"][0])
        epsilon = str(op_objects["epsilon"][0])
        C = str(op_objects["C"][0])
        data_sheet = str(op_objects["data_sheet"][0])
        
        metrics = []
        metrics = np.append(np.full([10, 1], op_objects[op_object]), np.arange(0, 10).reshape(-1, 1), axis = 1)
        metrics = np.append(metrics, AnalysisResults(op_result, "op"), axis = 1)
        if len(op_metrics) == 0:
            op_metrics = metrics
        else:
            op_metrics = np.vstack((op_metrics, metrics))
            
        if sampling_method == "None":
            result_filename = method + "_" + cv_method + "_" + epsilon + "_" + C + "_" + data_sheet + "_results.xlsx"
        else:
            result_filename = method + "_" + cv_method + "_" + sampling_method + "_" + epsilon + "_" + C + "_" + data_sheet + "_results.xlsx"
            
        if save_result:
            SaveResults(op_result, result_filename)
        
    op_metrics_columns = [op_object, "iteration", 
                        "train_mae", "train_std", "train_majority_mae", "train_majority_std", "train_minority_mae", "train_minority_std", "train_var", "train_rmse", "train_mape", "train_r2", 
                        "validation_mae", "validation_std", "validation_majority_mae", "validation_majority_std", "validation_minority_mae", "validation_minority_std", "validation_var", "validation_rmse", "validation_mape", "validation_r2", 
                        "test_mae", "test_std", "test_majority_mae", "test_majority_std", "test_minority_mae", "test_minority_std", "test_var", "test_rmse", "test_mape", "test_r2", "train_majority_rmse", "train_minority_rmse", "validation_majority_rmse", "validation_minority_rmse", "test_majority_rmse", "test_minority_rmse"
                        ]
    op_metrics = pd.DataFrame(op_metrics, columns=op_metrics_columns)
    
    # By MAE
    select_columns = ["C", "iteration", "test_mae", "test_majority_mae", "test_minority_mae", "test_rmse", "test_majority_rmse", "test_minority_rmse"]
    total_mae_results = op_metrics.iloc[op_metrics.groupby("iteration")["test_mae"].idxmin()].loc[:, select_columns]
    majority_mae_results = op_metrics.iloc[op_metrics.groupby("iteration")["test_majority_mae"].idxmin()].loc[:, select_columns]
    minority_mae_results = op_metrics.iloc[op_metrics.groupby("iteration")["test_minority_mae"].idxmin()].loc[:, select_columns]
    
    # By RMSE
    select_columns = ["C", "iteration", "test_rmse", "test_majority_rmse", "test_minority_rmse", "test_mae", "test_majority_mae", "test_minority_mae"]
    total_rmse_results = op_metrics.iloc[op_metrics.groupby("iteration")["test_rmse"].idxmin()].loc[:, select_columns]
    majority_rmse_results = op_metrics.iloc[op_metrics.groupby("iteration")["test_majority_rmse"].idxmin()].loc[:, select_columns]
    minority_rmse_results = op_metrics.iloc[op_metrics.groupby("iteration")["test_minority_rmse"].idxmin()].loc[:, select_columns]

    # Write metrics to excel
    writer = pd.ExcelWriter(filename)
    op_metrics.to_excel(excel_writer=writer, sheet_name="op_metrics")   
    
    total_mae_results.to_excel(excel_writer=writer, sheet_name="by_total_mae")   
    majority_mae_results.to_excel(excel_writer=writer, sheet_name="by_majoritiy_mae")   
    minority_mae_results.to_excel(excel_writer=writer, sheet_name="by_minority_mae")  
    total_mae_results.insert(0, "by", "total_mae")
    majority_mae_results.insert(0, "by", "majority_mae")
    minority_mae_results.insert(0, "by", "minority_mae")
    summary = pd.concat([total_mae_results, majority_mae_results, minority_mae_results])
    summary = summary.sort_values(by=["iteration", "by"], ascending=(True, True)) 
    summary.to_excel(excel_writer=writer, sheet_name="summary_mae")
            
    total_rmse_results.to_excel(excel_writer=writer, sheet_name="by_total_rmse")   
    majority_rmse_results.to_excel(excel_writer=writer, sheet_name="by_majoritiy_rmse")   
    minority_rmse_results.to_excel(excel_writer=writer, sheet_name="by_minority_rmse")  
    total_rmse_results.insert(0, "by", "total_rmse")
    majority_rmse_results.insert(0, "by", "majority_rmse")
    minority_rmse_results.insert(0, "by", "minority_rmse")
    summary = pd.concat([total_rmse_results, majority_rmse_results, minority_rmse_results])
    summary = summary.sort_values(by=["iteration", "by"], ascending=(True, True)) 
    summary.to_excel(excel_writer=writer, sheet_name="summary_rmse")
    
    writer.save()
    writer.close()

    return op_metrics


def HandleKeyValue(op_result: np.array):
    keys = op_result[0][-2]
    values = op_result[0][-1]       
    d = {}
    for i in range(len(keys)):
        if keys[i] not in d:
            d[keys[i]] = []
        d[keys[i]].append(values[i])    
    return d


def AnalysisHyperparameter(op_results: np.array, op_object: str = "C"):
    """_summary_

    Args:
        op_results (np.array): _description_
    """
    
    op_summarys = []
    for op_result in op_results:
        op_summary = []
        op_obejcts = HandleKeyValue(op_result)      
        op_summary.append(op_obejcts[op_object])
        summary = AnalysisResults(op_result, "op")
        metrics = np.mean(summary, axis=0)
        op_summary.append(metrics)
        op_summary.append(summary)
        op_summarys.append(op_summary)
    
    return op_summarys
    

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


def SaveResult(results: pd.DataFrame, filename: str, iteration: int=0, header=None, index=None):
    for i in range(len(results)):
        result = pd.DataFrame(results[i]).T
        if os.path.exists(filename):
            writer = pd.ExcelWriter(path=filename, mode="a", if_sheet_exists="replace")
        else:
            writer = pd.ExcelWriter(path=filename, mode="w")
        result.to_excel(excel_writer=writer, sheet_name="iteration-"+str(iteration*25+i), header=header, index=index)    
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
    print("Save all results into excel files for `%s` successfully." % filename)
       
       
def LoadResult(data_results: pd.DataFrame, i: int):
    sheet_name = "iteration-" + str(i)
    shape = data_results[sheet_name].shape[1]
    result = []
    for j in range(shape):
        result.append(data_results[sheet_name].T.iloc[j].dropna(axis=0).values)
    return result 


def LoadResults(filename: str, parallel: bool=False):
    if os.path.exists(filename):
        data_results = pd.read_excel(io=filename, sheet_name=None, header=None)
        length = len(data_results)
        results = []
        start_at = time.time()
        if parallel:
            results = (
                Parallel(n_jobs=10)
                (delayed(LoadResult)(data_results, i)
                # )
                # for i in tqdm(range(length)))
                for i in range(length))
            )
        else:
            for i in range(length):
                result = LoadResult(data_results, i)
                results.append(result)
        end_at = time.time()
        if parallel:
            print('Loading %s used %s s with parallel' % (filename, (end_at - start_at)))
        else:
            print('Loading %s used %s s without parallel' % (filename, (end_at - start_at)))
        return results
    else:
        print("%s not found" % filename)


def LoadSummarys(metrics_filename: str):
    data_metrics = pd.read_excel(io=metrics_filename, sheet_name="op_metrics", index_col=0)
    C_list = pd.unique(data_metrics["C"])
    data_metrics_mean = data_metrics.groupby("C").mean().iloc[:, 1:]

    op_summarys = []
    for C in C_list:
        op_summary = []
        op_summary.append([C])
        op_summary.append(data_metrics_mean.loc[C].values)
        op_summarys.append(op_summary)
    
    return op_summarys

# def AnalysisMetrics(op_metrics):

def SaveAnalyzedResult(results: pd.DataFrame, filename: str, iteration: int=0):
    for i in range(len(results)):
        result = pd.DataFrame(results[i])
        if os.path.exists(filename):
            writer = pd.ExcelWriter(path=filename, mode="a", if_sheet_exists="replace")
        else:
            writer = pd.ExcelWriter(path=filename, mode="w")
        result.to_excel(excel_writer=writer, sheet_name="iteration-"+str(iteration*25+i), index=None)    
        writer.save()
    return True

def AnalysisCosts(siml_results: pd.DataFrame, results: pd.DataFrame, filename: str="costs.xlsx"):
    iteration1 = len(siml_results)
    iteration2 = len(siml_results)
    if iteration1 == iteration2:
        iteration = iteration1
        l12_summary = []
        for i in range(iteration):
            l1 = pd.DataFrame({
                'Bandgap'   : siml_results[i][10], 
                'Prediction': siml_results[i][11],
                'Material'  : siml_results[i][9],
                'Weights'   : siml_results[i][19]
                })
            l2 = pd.DataFrame({
                'Bandgap' : results[i][1], 
                'Material': results[i][2],
                'Origin'  : results[i][3]
                })
            l12 = pd.merge(l1, l2, how="left", on=["Bandgap", "Material"])
            l12["Difference"] = l12["Weights"] - l12["Origin"]
            l12.insert(0, 'No', range(1, 1 + len(l12)))
            l12_summary.append(l12)
            
        SaveAnalyzedResult(l12_summary, filename)
        return l12_summary
    else:
        print("Two results are not match! Please check them firstly!")