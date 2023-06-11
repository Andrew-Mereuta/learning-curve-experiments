import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
import os


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


base_dir = '../lcdb-orig/'
pca_percentages = [0.5, 0.7, 0.9]
openmlid_dirs = os.listdir(base_dir)
postfix = '_plot_summary.gz'
summaries = {}
i = 0
np.set_printoptions(suppress=True)


for openmlid in openmlid_dirs:
    summaries[openmlid] = {}

    if i < 4:
        openmlid_dir = base_dir + openmlid + '/'
        learner_dirs = os.listdir(openmlid_dir)

        for learner_name in learner_dirs:
            summaries[openmlid][learner_name] = {}
            if i < 4:
                learner_dir = openmlid_dir + learner_name + '/'

                for pca_percentage in pca_percentages:
                    # i = i + 1
                    summary_directory = learner_dir + '_' + str(pca_percentage) + postfix
                    pca_summary = pd.read_pickle(summary_directory)

                    groups_by_curve = pca_summary.groupby(by='curve_model')
                    all_keys = groups_by_curve.groups.keys()

                    curves = []

                    for key in all_keys:
                        curve_group = groups_by_curve.get_group(key)
                        curve_group = curve_group.sort_values(by=['MSE tst'], ascending=True).iloc[0]
                        curves.append(curve_group)

                    summaries[openmlid][learner_name][pca_percentage] = pd.concat(curves, axis=1).T

functions_50 = {}
functions_70 = {}
functions_90 = {}

functions_mse_50 = {}
functions_mse_70 = {}
functions_mse_90 = {}

curve_types: set = set()

for openmlid, learner_data in summaries.items():
    for learner_name, pca_data in learner_data.items():
        for pca_key, data in pca_data.items():
            all_curves = data['curve_model']
            for curve in all_curves:
                curve_types.add(curve)
                if pca_key == 0.5:
                    curve_beta = functions_50.get(openmlid, {})
                    curve_mse = functions_mse_50.get(openmlid, {})

                    beta = curve_beta.get(curve, [])
                    mse = curve_mse.get(curve, [])

                    beta.append(data.query(f"curve_model == '{curve}'").iloc[0]['beta'])
                    mse.append(data.query(f"curve_model == '{curve}'").iloc[0]['MSE tst'])
                    # print(f"beta.size: {len(beta)};curve: {curve}; mse: {mse}")

                    curve_beta[curve] = beta
                    curve_mse[curve] = mse

                    functions_50[openmlid] = curve_beta
                    functions_mse_50[openmlid] = curve_mse
                elif pca_key == 0.7:
                    curve_beta = functions_70.get(openmlid, {})
                    curve_mse = functions_mse_70.get(openmlid, {})

                    beta = curve_beta.get(curve, [])
                    mse = curve_mse.get(curve, [])

                    beta.append(data.query(f"curve_model == '{curve}'").iloc[0]['beta'])
                    mse.append(data.query(f"curve_model == '{curve}'").iloc[0]['MSE tst'])

                    curve_beta[curve] = beta
                    curve_mse[curve] = mse

                    functions_70[openmlid] = curve_beta
                    functions_mse_70[openmlid] = curve_mse
                elif pca_key == 0.9:
                    curve_beta = functions_90.get(openmlid, {})
                    curve_mse = functions_mse_90.get(openmlid, {})

                    beta = curve_beta.get(curve, [])
                    mse = curve_mse.get(curve, [])

                    beta.append(data.query(f"curve_model == '{curve}'").iloc[0]['beta'])
                    mse.append(data.query(f"curve_model == '{curve}'").iloc[0]['MSE tst'])

                    curve_beta[curve] = beta
                    curve_mse[curve] = mse

                    functions_90[openmlid] = curve_beta
                    functions_mse_90[openmlid] = curve_mse


# curve = 'pow4'
# print(curve)
for openmlid in functions_50.keys():
    print(f"openmlid: {openmlid}")
    curve_beta50 = functions_50[openmlid]
    curve_beta70 = functions_70[openmlid]
    curve_beta90 = functions_90[openmlid]

    curve_mse50 = functions_mse_50[openmlid]
    curve_mse70 = functions_mse_70[openmlid]
    curve_mse90 = functions_mse_90[openmlid]

    for key, value in curve_mse50.items():
        average = sum(value) / len(value)
        curve_mse50[key] = average

    for key, value in curve_mse70.items():
        average = sum(value) / len(value)
        curve_mse70[key] = average

    for key, value in curve_mse90.items():
        average = sum(value) / len(value)
        curve_mse90[key] = average

    curve_mse50 = dict(sorted(curve_mse50.items(), key=lambda item: item[1]))
    curve_mse70 = dict(sorted(curve_mse70.items(), key=lambda item: item[1]))
    curve_mse90 = dict(sorted(curve_mse90.items(), key=lambda item: item[1]))

    for curve in curve_mse50.keys():
        print(f"\t{curve}")
        betas50 = np.mean(np.array(curve_beta50[curve]), axis=0)
        print(f"\t\t50: mse: {round(curve_mse50[curve], 4)}; {betas50}")

    for curve in curve_mse70.keys():
        print(f"\t{curve}")
        betas70 = np.mean(np.array(curve_beta70[curve]), axis=0)
        print(f"\t\t70: mse: {round(curve_mse70[curve], 4)}; {betas70}")

    for curve in curve_mse90.keys():
        print(f"\t{curve}")
        betas90 = np.mean(np.array(curve_beta90[curve]), axis=0)
        print(f"\t\t90: mse: {round(curve_mse90[curve], 4)}; {betas90}")

    # for curve in curve_types:
    #     print(f"\t{curve}")
    #     if curve in curve_beta50.keys():
    #         betas50 = np.around(np.mean(np.array(curve_beta50[curve]), axis=0), decimals=2)
    #         print(f"\t\t50: mse: {round(curve_mse50[curve],4)}; {betas50}")
    #     else:
    #         print(f"\t\t50: {None}")
    #
    #     if curve in curve_beta70.keys():
    #         betas70 = np.around(np.mean(np.array(curve_beta70[curve]), axis=0), decimals=2)
    #         print(f"\t\t70: mse: {round(curve_mse70[curve], 4)}; {betas70}")
    #     else:
    #         print(f"\t\t70: {None}")
    #
    #     if curve in curve_beta90.keys():
    #         betas90 = np.around(np.mean(np.array(curve_beta90[curve]), axis=0), decimals=2)
    #         print(f"\t\t90: mse: {round(curve_mse90[curve], 4)}; {betas90}")
    #     else:
    #         print(f"\t\t90: {None}")

#
#
# np.set_printoptions(suppress=True)
# print('PCA= 0.5')
# for openmlid in functions_50.keys():
#     curve_beta = functions_50[openmlid]
#     print(f"Openmlid: {openmlid}")
#     for curve in curve_beta.keys():
#         betas = np.array(np.mean(np.array(curve_beta[curve]), axis=0))
#         print(f"Curve: {curve};\nAvg beta: {betas}")
#     print('==================================================================================')
#
# print('==================================================================================')
# print('PCA= 0.7')
# for openmlid in functions_70.keys():
#     curve_beta = functions_70[openmlid]
#     print(f"Openmlid: {openmlid}")
#     for curve in curve_beta.keys():
#         betas = np.array(np.mean(np.array(curve_beta[curve]), axis=0))
#         print(f"Curve: {curve};\nAvg beta: {betas}")
#     print('==================================================================================')
#
# print('==================================================================================')
# print('PCA= 0.9')
# for openmlid in functions_90.keys():
#     curve_beta = functions_90[openmlid]
#     print(f"Openmlid: {openmlid}")
#     for curve in curve_beta.keys():
#         betas = np.array(np.mean(np.array(curve_beta[curve]), axis=0))
#         print(f"Curve: {curve};\nAvg beta: {betas}")
#     print('==================================================================================')
