import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_data2(row):
    [X, Y] = get_XY2(row)
    Y = 1 - Y

    plt.plot(X, Y, '*r', label='test anchors')
    set_ylim2(row)
    plt.xlabel('training set size')
    plt.ylabel('error rate')

    learner = row.learner
    openmlid = row.openmlid
    plt.title('%s dataset %d' % (learner, openmlid))


def plot_trn_data2(row):
    [X, Y] = get_XY2(row)
    Y = 1 - Y

    offset = np.argwhere(X == row.max_anchor_seen)[0][0]

    X_trn = X[:offset + 1]
    Y_trn = Y[:offset + 1]

    plt.plot(X_trn, Y_trn, 'ob', label='train anchors')
    set_ylim2(row)
    plt.xlabel('training set size')
    plt.ylabel('error rate')

    learner = row.learner
    openmlid = row.openmlid
    plt.title('%s dataset %d' % (learner, openmlid))


def plot_prediction2(row):
    curve_model = row.curve_model
    [X, Y] = get_XY2(row)
    Y = 1 - Y

    plt.plot(X, 1 - row.prediction, ':', label=curve_model)
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


def plot_prediction_smooth2(row):
    curve_model = row.curve_model

    [X, Y] = get_XY2(row)
    Y = 1 - Y

    fun = get_fun_model_id(row.beta, curve_model)

    X_plot = np.arange(np.min(X), np.max(X))
    Y_hat = 1 - fun(X_plot)

    plt.plot(X_plot, Y_hat, '-', label=curve_model)
    plt.legend()


# Starts here
# plt.figure(figsize=(8, 8))

openmlid = 3
a_f = str(openmlid) + '_anchors_scores_example.gz'
e_f = str(openmlid) + '_extrapolations.gz'

dir = '../'

line = pd.read_pickle(dir + a_f)
rows = pd.read_pickle(dir + e_f)

anchor_prediction_array = np.tile(line['anchor_prediction'].values[0], (len(rows), 1))
rows['anchor_prediction'] = anchor_prediction_array.tolist()

score_array = np.tile(line['score'].values[0], (len(rows), 1))
rows['score'] = score_array.tolist()

row = rows.iloc[0, :]

# plots all the points on the curve (red stars)
plot_data2(row)

# plots the points used for training only (blue dots)
plot_trn_data2(row)

# plot the curve fit from the row of the dataframe (dotted)
plot_prediction2(row)

# plot the curve fit using the beta parameters in the dataframe (line)
# this plot is smoother since we can predict any x-value
# this curve should overlap with the curve plotted previously
plot_prediction_smooth2(row)

# show the plot
plt.show()

# show the information of the row
print(row)