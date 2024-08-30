import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_conf_matrix(ax, cf_matrix, add_title=""):
	ax = sns.heatmap(cf_matrix/np.sum(cf_matrix,axis=1, keepdims=True),
		annot=True, fmt='.2%', cmap='Blues',  vmin=0, vmax=1, cbar=False)
	ax.set_xlabel('\nPredicted Values')
	ax.set_ylabel('Actual Values ')
	ax.set_title(f"Confusion {add_title}")


def plot_prob_dist_bin(ax, y_pred_prob, y_true, add_title=""):
    mask_non_crop = y_true == 0
    mask_crop = y_true == 1
    binwidth = 0.04
    bins = np.arange(0, 1+ binwidth, binwidth)
    ax.hist(y_pred_prob[mask_non_crop,1], label="Negative", alpha=0.5,bins=bins,edgecolor='white', linewidth=1.2)
    ax.hist(y_pred_prob[mask_crop,1], label="Target Crop", alpha=0.5,bins=bins, edgecolor='black', linewidth=1.2)
    ax.set_xlim(0,1)
    ax.axvline(0.5, ls="dashed", lw=2, label="Probability Threshold", color="black")
    ax.set_title(f"Histogram of the predicted probability {add_title}")
    ax.legend(loc="upper center")
    ax.set_xlabel("Target Crop Probability")
    ax.set_ylabel("Count")


def plot_dist_bin(ax, y_pred_cont, y_true, add_title=""):
    binwidth = 5 if np.max(y_true) > 50  else 0.5
    bins = np.arange(0, np.max(y_true)+ binwidth, binwidth)
    ax.hist(y_true, label="Ground Truth", alpha=0.6, bins=bins, edgecolor='black', linewidth=1.2)
    ax.hist(y_pred_cont, label="Prediction", alpha=0.35, bins=bins,edgecolor='black', linewidth=1.2 )
    ax.set_title(f"Histogram of target values {add_title}")
    ax.legend(loc="upper right")
    ax.set_xlabel("Target value")
    ax.set_ylabel("Count")
    ax.set_xlim(0)

def plot_true_vs_pred(ax, y_pred_cont, y_true, add_title=""):
    y = np.arange(np.min(y_true), np.max(y_true))
    ax.plot(y, y, "-", color="red")
    ax.scatter(y_true, y_pred_cont, marker="o", edgecolors='black', s=30)
    ax.set_title(f"Prediction vs ground truth {add_title}")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
