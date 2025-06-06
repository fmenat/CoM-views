import numpy as np

from .metric_predictions import OverallAccuracy,F1Score,Precision,Recall,Kappa, ConfusionMatrix,AverageAccuracy,ROC_AUC
from .metric_predictions import R2Score, MAPE, RMSE, MAE, BIAS, rRMSE,MedAE,PCorr
from .metric_predictions import get_n_data, get_n_true, get_n_pred
from .metric_predictions import CatEntropy,LogP,P_max, P_true
from .metric_robustness import DeformanceRobustnessScore, PerformanceRobustnessScore, ClassesUnChanged, SpreadScore, DeformationScore

class BaseMetrics(object):
	"""central metrics class to provide standard metric types"""

	def __init__(self, task_type="classification", metric_types=[]):
		"""build BaseMetrics

		Parameters
		----------
		metric_types : list of str, optional
		    declaration of metric types to be used, by default all are used
		"""
		self.task_type = task_type
		self.metric_types = [v.lower() for v in metric_types]
		self.metric_dict = {}

	def __call__(self, prediction, target):
		"""call forward for each metric in collection

		Parameters
		----------
		prediction : array_like (n_samples, n_outputs)
		    prediction tensor
		target : array_like     (n_samples, n_outputs)
		    ground truth tensor, in classification: n_outputs=1, currently working only =1
		"""
		if not isinstance(prediction, np.ndarray):
			prediction = np.asarray(prediction)
		if not isinstance(target, np.ndarray):
			target = np.asarray(target)

		if self.task_type=="classification" and hasattr(self,"aux_metric"):
			self.n_samples = self.aux_metric(prediction,target)
		else:
			self.n_samples = []

		#forward over all metrics
		return {name: func(prediction, target) for (name, func) in self.metric_dict.items()}

	def get_metric_types(self):
		"""return list of metric types inside collection

		Returns
		-------
		list of strings
		"""
		return list(self.metric_dict.keys())

	def reverse_forward(self, target,prediction):
		return self(prediction, target)



class ClassificationMetrics(BaseMetrics):
    def __init__(self, metric_types=["OA","AA","KAPPA", "F1 MACRO","P MACRO","ENTROPY","LOGP"]):
        """build ClassificationMetrics

        Parameters
        ----------
        metric_types : list of str, optional
            declaration of metric types to be used, by default all are used
        """
        super(ClassificationMetrics,self).__init__("classification", metric_types)
        self.aux_metric = get_n_data
        for metric in self.metric_types:
            if "oa"==metric:
                self.metric_dict["OA"] = OverallAccuracy()
            elif "aa"==metric or "r macro" == metric or "recall"==metric:
                self.metric_dict["AA"] = Recall("macro") #AverageAccuracy()
            elif "f1" in metric:
                avg_mode = metric.split(" ")[1] if len(metric.split(" "))!=1 else "macro"
                self.metric_dict[f"F1 {avg_mode.upper()}"] = F1Score(avg_mode)
            elif "p " in metric or "precision"==metric:
                avg_mode = metric.split(" ")[1] if len(metric.split(" "))!=1 else "macro"
                self.metric_dict[f"P {avg_mode.upper()}"] = Precision(avg_mode)
            elif "r " in metric or "recall"==metric :
                avg_mode = metric.split(" ")[1] if len(metric.split(" "))!=1 else "macro"
                self.metric_dict[f"R {avg_mode.upper()}"] = Recall(avg_mode)
            elif "kappa"==metric:
                self.metric_dict["KAPPA"] = Kappa()
            elif "confusion" in metric or "matrix" in metric:
                self.metric_dict["MATRIX"] = ConfusionMatrix()
            elif "ntrue"==metric:
                self.metric_dict["N TRUE"] = get_n_true
            elif "npred"==metric:
                self.metric_dict["N PRED"] = get_n_pred


class SoftClassificationMetrics(BaseMetrics):
    def __init__(self, metric_types=["AUC MACRO", "ENTROPY","LOGP", "PMAX", "PTRUE"]):
        """build SoftClassificationMetrics

        Parameters
        ----------
        metric_types : list of str, optional
            declaration of metric types to be used, by default all are used
        """
        super(SoftClassificationMetrics,self).__init__("classification", metric_types)
        for metric in self.metric_types:
            if "auc" in metric or "roc-auc"==metric:
                avg_mode = metric.split(" ")[1] if len(metric.split(" "))!=1 else "macro"
                self.metric_dict[f"AUC {avg_mode.upper()}"] = ROC_AUC(avg_mode)
            elif "entropy" == metric:
                self.metric_dict["ENTROPY"] = CatEntropy()
            elif "logp"==metric:
                self.metric_dict["LOGp"] = LogP()
            elif "pmax"==metric:
                self.metric_dict["Pmax"] = P_max()
            elif "ptrue"==metric:
                self.metric_dict["Ptrue"] = P_true()



class RegressionMetrics(BaseMetrics):
    def __init__(self, metric_types=["R2","RMSE", "rRMSE", "MAE", "MedAE", "MAPE", "BIAS", "PCorr"]):
        """build RegressionMetrics

        Parameters
        ----------
        metric_types : list of str, optional
            declaration of metric types to be used, by default all are used
        """
        super(RegressionMetrics,self).__init__("regression", metric_types)

        for metric in self.metric_types:
            if "r2"==metric:
                self.metric_dict["R2"] = R2Score()
            elif "mae"==metric:
                self.metric_dict["MAE"] = MAE()
            elif "medae"==metric:
                self.metric_dict["MedAE"] = MedAE()
            elif "rmse"==metric:
                self.metric_dict["RMSE"] = RMSE()
            elif "rrmse"==metric:
                self.metric_dict["rRMSE"] = rRMSE()
            elif "mape"==metric:
                self.metric_dict["MAPE"] = MAPE()
            elif "bias"==metric:
                self.metric_dict["BIAS"] = BIAS()
            elif "pcorr" == metric:
                self.metric_dict["PCorr"] = PCorr()


#-------------------Robustness Metrics-------------------
class BaseMetrics_3args(object):
    """central metrics class to provide standard metric types"""

    def __init__(self, metric_types=[]):
        """build BaseMetrics

        Parameters
        ----------
        metric_types : list of str, optional
            declaration of metric types to be used, by default all are used
        """
        self.metric_types = [v.lower() for v in metric_types]
        self.metric_dict = {}

    def __call__(self, real_prediction, noise_prediction, ground_truth):
        """call forward for each metric in collection

        Parameters
        ----------
        real_prediction : array_like (n_samples, n_outputs)
            real prediction tensor
        noise_prediction: array_like (n_samples, n_outputs)
            noise prediction tensor
        ground_truth : array_like     (n_samples, n_outputs)
            ground truth tensor, in classification: n_outputs=1, currently working only =1
        """
        if not isinstance(real_prediction, np.ndarray):
            real_prediction = np.asarray(real_prediction)
        if not isinstance(noise_prediction, np.ndarray):
            noise_prediction = np.asarray(noise_prediction)
        if not isinstance(ground_truth, np.ndarray):
            ground_truth = np.asarray(ground_truth)

        #forward over all metrics
        return {name: func(real_prediction, noise_prediction, ground_truth) for (name, func) in self.metric_dict.items()}

    def get_metric_types(self):
        """return list of metric types inside collection

        Returns
        -------
        list of strings
        """
        return list(self.metric_dict.keys())


class RobustnessMetrics(BaseMetrics_3args):
    def __init__(self, metric_types=["PRS", "PRS_diff", "DRS", "DRS_diff", "DS","DS_diff" ,"SS"], task_type=""):
        """build RobustnessMetrics

        Parameters
        ----------
        metric_types : list of str, optional
            declaration of metric types to be used, by default all are used
        """
        super(RobustnessMetrics,self).__init__(metric_types)
        for metric in self.metric_types:
            if "prs" == metric:
                self.metric_dict["PRS"] = PerformanceRobustnessScore()
            elif "prs_mae" == metric:
                self.metric_dict["PRS_MAE"] = PerformanceRobustnessScore(diff="MAE")
            elif "prs_diff" == metric:
                self.metric_dict["PRS_diff"] = PerformanceRobustnessScore(normalize=False)
            elif "drs" == metric:
                self.metric_dict["DRS"] = DeformanceRobustnessScore()
            elif "drs_mae" == metric:
                self.metric_dict["DRS_MAE"] = DeformanceRobustnessScore(diff="MAE")
            elif "drs_diff" == metric:
                self.metric_dict["DRS_diff"] = DeformanceRobustnessScore(normalize=False)
            elif "ds" == metric:
                self.metric_dict["DS"] = DeformationScore()
            elif "ds_diff" == metric:
                self.metric_dict["DS_diff"] = DeformationScore(normalize=False)
            elif "ss" == metric:
                self.metric_dict["SS"] = SpreadScore()

            if len(metric.split(" ")) > 1:
                metric_name, reduce = metric.split(" ")
                if "prs"==metric_name and "none" ==reduce:
                    self.metric_dict["PRS"] = PerformanceRobustnessScore(reduce="none")
                elif "prs_mae" ==metric_name and "none" ==reduce:
                    self.metric_dict["PRS_MAE"] = PerformanceRobustnessScore(diff="MAE",reduce="none")
                elif "prs_diff" ==metric_name and "none" ==reduce:
                    self.metric_dict["PRS_diff"] = PerformanceRobustnessScore(normalize=False, reduce="none")
                elif "drs" ==metric_name and "none" ==reduce:
                    self.metric_dict["DRS"] = DeformanceRobustnessScore(reduce="none")
                elif "drs_mae" ==metric_name and "none" ==reduce:
                    self.metric_dict["DRS_MAE"] = DeformanceRobustnessScore(diff="MAE", reduce="none")
                elif "drs_diff" ==metric_name and "none" ==reduce:
                    self.metric_dict["DRS_diff"] = DeformanceRobustnessScore(normalize=False, reduce="none")
                elif "ds" == metric:
                    self.metric_dict["DS"] = DeformationScore(reduce="none")
                elif "ds_diff" == metric:
                    self.metric_dict["DS_diff"] = DeformationScore(normalize=False, reduce="none")
                elif "ss" == metric:
                    self.metric_dict["SS"] = SpreadScore(reduce="none")

        if task_type == "classification":
            self.metric_dict["CC"] = ClassesUnChanged(inverse=True)
            self.metric_dict["CCraw"] = ClassesUnChanged(normalize=False, inverse=True)
            self.metric_dict["CU"] = ClassesUnChanged()
            self.metric_dict["CUraw"] = ClassesUnChanged(normalize=False)
