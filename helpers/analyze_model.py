
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix

import helpers.run_model

def display_true_vs_pred(res_df, y_lim):
  ax = plt.subplot(111)
  sns.scatterplot(x='y_true', y='y_pred', data=res_df, s=8, ax=ax)
  ax.plot((0, y_lim), (0, y_lim), ':k')  
  ax.set_xlim(0, y_lim)
  ax.set_ylim(0, y_lim)
  ax.set_title('y_pred vs y_true')

def display_resid_distrib(res, y_lim, res_lim):
  plt.figure(figsize=(18,13))

  ax = plt.subplot(321)
  z = res.y_true.sort_values().iloc[:-100] # Skip extreme values
  sns.distplot(z, ax=ax)
  ax.set_xlim(*y_lim)

  ax = plt.subplot(322)
  sns.distplot(res.resid.sort_values().iloc[100:], ax=ax) # Skip the 100 lowest negative resid
  ax.set_xlim(*res_lim)

  ax = plt.subplot(323)
  z = res.y_pred.sort_values().iloc[:-100] # Skip extreme values
  sns.distplot(z, ax=ax)
  ax.set_xlim(*y_lim)

  ax = plt.subplot(324)
  sns.distplot(res.resid_standardized.sort_values().iloc[100:], ax=ax) # Skip the 100 lowest resid
  ax.set_xlim(-3, 3)

  ax = plt.subplot(325)
  stats.probplot(res.sort_values('resid')[100:].resid, plot=ax)
  ax.set_title("QQPlot - Residuals")
  
  ax = plt.subplot(326)
  sns.scatterplot(x='y_true', y='resid', data=res, s=5, ax=ax)
  ax.set_xlim(*y_lim)
  ax.set_ylim(*res_lim)
  ax.set_title('Residuals vs y_true')

  plt.tight_layout()

def get_confusion_matrix(classifier, x_test, y_true, normalize):
  # normalize='true' or 'pred' or 'all. Normalizes confusion matrix over the true (rows), predicted (columns)
  cm = confusion_matrix(y_true=y_true, y_pred=classifier.predict(x_test), normalize=normalize)
  cm_df = pd.DataFrame(cm, columns=list(range(cm.shape[0])))
  if not normalize:
    cm_df = cm_df.astype('int')
  cm_df.index.name = 'Actual'
  cm_df.columns.name = 'Predicted'
  return cm_df

def plot_confusion_matrix(classifier, x_test, y_true, normalize=None):
  cm_df = get_confusion_matrix(classifier, x_test, y_true, normalize=normalize)
  fig = plt.figure(figsize=(15, 10))
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  cmap = sns.cubehelix_palette(light=1, as_cmap=True)
  fmt='0d'
  if normalize:
    fmt='.2%'
  res = sns.heatmap(cm_df, annot=True,
                    #vmin=0.0, vmax=100.0,
                    fmt=fmt, cmap=cmap)
  res.invert_yaxis()
  #plt.yticks([0.5,1.5,2.5], [ 'Dog', 'Cat', 'Rabbit'],va='center')
  plt.title(f'Confusion Matrix (Normalize={normalize})')