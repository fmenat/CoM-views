import numpy as np

def PerformanceRobustnessScore(diff="RMSE", normalize=True, reduce="mean"):
    def metric(real_prediction, noise_prediction, ground_truth):
        if diff.lower()=="rmse":
            diff_func = lambda x,y: np.sqrt( np.mean((x -y)**2, axis=0) )
        elif diff.lower() == "mae":
            diff_func = lambda x,y: np.median( np.abs(x -y), axis=0)
        ratio = diff_func(ground_truth, noise_prediction) / (diff_func(ground_truth, real_prediction) + 1e-10)
        if normalize:
             ratio = np.minimum(1, np.exp(1- ratio) )
        
        if reduce == "mean":
            return np.mean(ratio)
        else:
            return ratio
    return metric

def DeformanceRobustnessScore(diff="RMSE", normalize=True, reduce="mean"):
    def metric(real_prediction, noise_prediction, ground_truth):
        #std_ = np.std(ground_truth, axis=0) 
        std_ = np.std(real_prediction, axis= 0) #similar as executing y expect = y real pred + epsilon * std (real pred)
        if diff.lower()=="rmse":
            diff_func = lambda x,y: np.sqrt( np.mean((x -y)**2, axis=0) )
        elif diff.lower() == "mae":
            diff_func = lambda x,y: np.median( np.abs(x -y), axis=0)
        ratio = diff_func(real_prediction, noise_prediction) / std_  #denonp.minator is just a noise based on std: diff(y pred, y pred + std)
        if normalize:
             ratio = np.minimum(1, np.exp(1- ratio) )
        
        if reduce == "mean":
            return np.mean(ratio)
        else:
            return ratio
    return metric

def DeformationScore(diff="RMSE", normalize=True, reduce="mean"):
    def metric(real_prediction, noise_prediction, ground_truth):
        mean_ = np.mean(real_prediction, axis= 0)
        if diff.lower()=="rmse":
            diff_func = lambda x,y: np.sqrt( np.mean((x -y)**2, axis=0) )
        elif diff.lower() == "mae":
            diff_func = lambda x,y: np.median( np.abs(x -y), axis=0)
        ratio = diff_func(real_prediction, noise_prediction)
        if normalize:
             ratio /= mean_

        if reduce == "mean":
            return np.mean(ratio)
        else:
            return ratio
    return metric

def SpreadScore(diff="RMSE", normalize=True, reduce="mean"):
    def metric(real_prediction, noise_prediction, ground_truth):
        std_ = np.std(real_prediction, axis= 0) #similar as executing y expect = y real pred + epsilon * std (real pred)
        if diff.lower()=="rmse":
            diff_func = lambda x,y: np.sqrt( np.mean((x -y)**2, axis=0) )
        elif diff.lower() == "mae":
            diff_func = lambda x,y: np.median( np.abs(x -y), axis=0)
        ratio = diff_func(real_prediction, noise_prediction)
        if normalize:
             ratio /= 3*std_

        if reduce == "mean":
            return np.mean(ratio)
        else:
            return ratio
    return metric

def ClassesUnChanged(normalize=True, inverse = False):
    def metric(real_prediction, noise_prediction, ground_truth):
        real_class_pred = real_prediction.argmax(axis=-1)
        noise_class_pred = noise_prediction.argmax(axis=-1)

        if inverse:
            total_score = np.sum(real_class_pred != noise_class_pred)
        else:
            total_score = np.sum(real_class_pred == noise_class_pred)
        
        if normalize:
            total_score /= len(real_class_pred)
        
        return total_score
    return metric