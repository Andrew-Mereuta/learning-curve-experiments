import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_XY2(row):
    X = np.array(row.anchor_prediction)
    Y = np.array(row.score)  # make it numpy array
    return [X, Y]


def set_ylim2(row, margin=0.05):
    [X, Y] = get_XY2(row)
    Y = 1 - Y
    Y_diff = np.max(Y) - np.min(Y)
    plt.ylim([np.min(Y), np.max(Y)])
    # plt.ylim([np.min(Y) - Y_diff*margin,np.max(Y) + Y_diff*margin])


def plot_data2(row, label='test_anchors'):
    [X, Y] = get_XY2(row)
    Y = 1 - Y

    plt.plot(X, Y, '*r', label=label)
    set_ylim2(row)
    plt.xlabel('training set size')
    plt.ylabel('error rate')

    learner = row.learner
    openmlid = row.openmlid
    plt.title('%s dataset %d' % (learner, openmlid))


def plot_trn_data2(row, label='train anchors'):
    [X, Y] = get_XY2(row)
    Y = 1 - Y

    offset = np.argwhere(X == row.max_anchor_seen)[0][0]

    X_trn = X[:offset + 1]
    Y_trn = Y[:offset + 1]

    plt.plot(X_trn, Y_trn, 'ob', label=label)
    set_ylim2(row)
    plt.xlabel('training set size')
    plt.ylabel('error rate')

    learner = row.learner
    openmlid = row.openmlid
    plt.title('%s dataset %d' % (learner, openmlid))


def plot_prediction2(row, label=None):
    [X, Y] = get_XY2(row)
    Y = 1 - Y
    if label is None:
        plt.plot(X, 1 - row.prediction, ':', label=row.curve_model)
    else:
        plt.plot(X, 1 - row.prediction, ':', label=label)
    plt.legend()


def get_num_par(model_id):
    if model_id == 'last1':
        return 1
    if model_id in ['pow2', 'log2', 'exp2', 'lin2', 'ilog2']:
        return 2
    if model_id in ['pow3', 'exp3', 'vap3', 'expp3', 'expd3', 'logpower3']:
        return 3
    if model_id in ['mmf4', 'wbl4', 'exp4', 'pow4']:
        return 4


def get_fun_model_id(beta, model_id):
    num_par = get_num_par(model_id)
    fun = None

    # unpack parameters
    if num_par == 1:
        a = beta[0]
    if num_par == 2:
        a, b = beta[0], beta[1]
    if num_par == 3:
        a, b, c = beta[0], beta[1], beta[2]
    if num_par == 4:
        a, b, c, d = beta[0], beta[1], beta[2], beta[3]

    # define curve models
    if model_id == 'pow2':
        fun = lambda x: -a * x ** (-b)
    if model_id == 'pow3':
        fun = lambda x: a - b * x ** (-c)
    if model_id == 'log2':
        fun = lambda x: -a * np.log(x) + b
    if model_id == 'exp3':
        fun = lambda x: a * np.exp(-b * x) + c
    if model_id == 'exp2':
        fun = lambda x: a * np.exp(-b * x)
    if model_id == 'lin2':
        fun = lambda x: a * x + b
    if model_id == 'vap3':
        fun = lambda x: np.exp(a + b / x + c * np.log(x))
    if model_id == 'mmf4':
        fun = lambda x: (a * b + c * x ** d) / (b + x ** d)
    if model_id == 'wbl4':
        fun = lambda x: (c - b * np.exp(-a * (x ** d)))
    if model_id == 'exp4':
        fun = lambda x: c - np.exp(-a * (x ** d) + b)
    if model_id == 'expp3':
        # fun = lambda x: a * np.exp(-b*x) + c
        fun = lambda x: c - np.exp((x - b) ** a)
    if model_id == 'pow4':
        fun = lambda x: a - b * (x + d) ** (-c)  # has to closely match pow3
    if model_id == 'ilog2':
        fun = lambda x: b - (a / np.log(x))
    if model_id == 'expd3':
        fun = lambda x: c - (c - a) * np.exp(-b * x)
    if model_id == 'logpower3':
        fun = lambda x: a / (1 + (x / np.exp(b)) ** c)
    if model_id == 'last1':
        fun = lambda x: (a + x) - x  # casts the prediction to have the correct size
    return fun


def plot_prediction_smooth2(row, label=None):
    curve_model = row.curve_model

    [X, Y] = get_XY2(row)
    Y = 1 - Y

    fun = get_fun_model_id(row.beta, curve_model)

    X_plot = np.arange(np.min(X), np.max(X))
    Y_hat = 1 - fun(X_plot)
    if label is None:
        plt.plot(X_plot, Y_hat, '-', label=curve_model)
    else:
        plt.plot(X_plot, Y_hat, '-', label=label)
    plt.legend()


openmlid = 3
file_name = '/' + str(openmlid) + '_plot_summary.gz'
base_dir = '../lcdb-orig/openmlid_' + str(openmlid)
pca_dir = base_dir + '_pca_'
pca_percentages = [90, 70, 50]

default_summary = pd.read_pickle(base_dir + file_name)
max_anchor = -np.sort(-np.array(default_summary['max_anchor_seen']))[0]
summary = default_summary.query('max_anchor_seen == ' + str(max_anchor)).iloc[0, :]

summaries = {100: summary}

for pca_percentage in pca_percentages:
    directory = pca_dir + str(pca_percentage) + file_name
    pca_summary = pd.read_pickle(directory)
    max_anchor = -np.sort(-np.array(pca_summary['max_anchor_seen']))[0]
    summaries[pca_percentage] = pca_summary.query('max_anchor_seen == ' + str(max_anchor)).iloc[0, :]

test_label = 'test_anchors_'
train_label = 'train_anchors_'
curve_label = 'curve_label_'
for pca_percentage in summaries:
    s = summaries[pca_percentage]
    plot_data2(s, label=(test_label + str(pca_percentage)))
    plot_trn_data2(s, label=(train_label + str(pca_percentage)))
    plot_prediction2(s, label=(curve_label + str(pca_percentage) + '_' + s.curve_model))
    plot_prediction_smooth2(s, label=(curve_label + str(pca_percentage) + '_' + s.curve_model))

plt.ylim(0, 0.4)
plt.show()

np.set_printoptions(suppress=True)

for pca_percentage in summaries:
    s = summaries[pca_percentage]
    betas = np.array(s['beta'])
    print(f"{pca_percentage}: {betas}")

# print(summary)

