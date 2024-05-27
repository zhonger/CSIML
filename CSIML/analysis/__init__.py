"""Initialize for analysis module"""

from ._base import handle_key_value, index_to_name, metrics2array, metrics2msg
from ._dataclass import ResData
from .common import evaluation, evaluation_batch, evaluation_metric, evaluation_metrics
from .cv import AnalysisCV
from .hyperparameter import (
    analyze_hyperparameter,
    cal_hypers,
    cal_single_hyper,
    get_hyperparameter_metrics,
    load_hypers,
    load_single_hyper,
    order_by_metric,
    save_hyperparameter_metrics,
    save_ordered_metrics,
)
from .io import (
    load_pkl,
    load_result,
    load_results,
    load_summarys,
    save_result,
    save_results,
    write_excel,
    write_pkl,
)
from .main import (
    Analysis,
    analyze_results,
    analyze_summarys,
    load_metrics,
    save_analyzed_result,
    save_metrics,
    save_metrics_batch,
)

__all__ = [
    "index_to_name",
    "handle_key_value",
    "metrics2array",
    "metrics2msg",
    "ResData",
    "write_excel",
    "write_pkl",
    "load_pkl",
    "load_result",
    "load_results",
    "load_summarys",
    "save_result",
    "save_results",
    "evaluation",
    "evaluation_batch",
    "evaluation_metric",
    "evaluation_metrics",
    "Analysis",
    "analyze_results",
    "analyze_summarys",
    "load_metrics",
    "save_analyzed_result",
    "save_metrics",
    "save_metrics_batch",
    "analyze_hyperparameter",
    "cal_hypers",
    "cal_single_hyper",
    "get_hyperparameter_metrics",
    "load_hypers",
    "load_single_hyper",
    "order_by_metric",
    "save_hyperparameter_metrics",
    "save_ordered_metrics",
    "AnalysisCV",
]
