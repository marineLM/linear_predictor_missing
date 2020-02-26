import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import numpy as np

plt.rcParams['pdf.fonttype'] = 42


def display_curves(scores_avg, bayes_rate, file_root, n_sizes, p_sizes,
                   do_legend=True):
    if len(p_sizes) == 1:
        x_axis = 'n'
    elif len(n_sizes) == 1:
        x_axis = 'p'

    idx = [scores_avg.train_test == 'train', scores_avg.train_test == 'test']
    # XXX: remove ", D1" for the MLP
    scores_avg.method = scores_avg.method.str.replace(', D1', '')

    for scoring in ['mse', 'r2']:
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))

        ax = axes
        for i in [0, 1]:
            if x_axis == 'n' and bayes_rate is not None:
                p = p_sizes[0]
                br = bayes_rate[bayes_rate.p == p]
                ax.axhline(br.iloc[0][scoring], label='Bayes rate', color='k')
            elif x_axis == 'p'and bayes_rate is not None:
                bayes_rate.sort_values(by='p', inplace=True)
                sns.lineplot(x=p_sizes, y=bayes_rate[scoring], ax=ax,
                             label='Bayes rate', color='k')

            if i == 0:
                # Train
                sns.lineplot(
                    data=scores_avg[idx[i]], x=x_axis, y=scoring,
                    hue='method', ax=ax, ci=None, legend=False,
                    style="train_test",
                    dashes=6 * [(1, 2)])
            else:
                sns.lineplot(
                    data=scores_avg[idx[i]], x=x_axis, y=scoring,
                    hue='method', ax=ax, estimator=np.median)

        if x_axis == 'n':
            ax.set(xscale="log")

        ax.set_xlabel('number of training samples')
        ax.set_ylabel(scoring.upper())
        # ax.set_title(titles[i])
        ax.grid(True)
        # if scoring == "mse":
        #     ax.set_ylim(top=20)
        handles, labels = ax.get_legend_handles_labels()
        if bayes_rate is not None:
            del handles[1], labels[1]
            del handles[1], labels[1]
        else:
            del handles[0], labels[0]
        if scoring == 'r2':
            ymin, ymax = plt.ylim()
            if 'mixture1' in file_root:
                ymin = 0.35
            elif 'mixture3' in file_root:
                ymin = 0.05
            else:
                ymin = 0.6
                plt.axhline(0, color='.8', lw=2)
            plt.ylim(ymin, 1.02)
            plt.axhline(1, color='.8', lw=2)
        # axes[i].legend(handles=handles, labels=labels)
        plt.xlim(scores_avg.n.min(), scores_avg.n.max())
        ax.get_legend().set_visible(False)
        lgd = fig.legend(handles, labels, loc=(.185, .15), ncol=2,
                         handletextpad=.3, handlelength=1.5)

        if do_legend:
            train_line = mlines.Line2D([], [], color='.5',
                                       linestyle=':', label='Train')
            test_line = mlines.Line2D([], [], color='.5',
                                      linestyle='-', label='Test')
            plt.legend(handles=[train_line, test_line], loc='upper right')

        plt.tight_layout()
        plt.savefig("../figures/{}_{}.pdf".format(
            file_root, scoring), bbox_inches='tight', edgecolor='none',
            facecolor='none')

        plt.close()


if __name__ == '__main__':

    for data_type in ['mixture1', 'mixture3', 'selfmasked_proba']:

        filename = 'allresultsPPCA_{}_10iter'.format(
            data_type)

        scores_avg = pd.read_csv(
            '../results/' + filename + '_sa.csv', index_col=0)
        # Load bayes_rate if it could be computed
        # (typically not for 'selfmasked' yet)
        try:
            file = open('../results/' + filename + '_br.csv', 'rb')
        except FileNotFoundError:
            bayes_rate = None
        else:
            bayes_rate = pd.read_csv(file, index_col=0)
            file.close()

        # Choose dimensions
        p_sizes = [10]
        # n_sizes = [25000]

        # Limit learning curves to n=1e5
        n_sizes = scores_avg.n.unique()
        n_sizes = [n for n in n_sizes if n <= 1e5]
        # p_sizes = scores_avg.p.unique()

        # Choose the methods for which the curves will be plotted
        if data_type == 'mixture1':
            methods = ['ConstantImputedLR', 'EMLR', 'ExpandedLR', 'MICELR',
                       'MLP W0.3, D1', 'MLP W1, D1', 'MLP W4, D1']
            # methods= ['ConstantImputedLR', 'EMLR', 'ExpandedLR',
            #           'MLP W0.003, D1','MLP W0.01, D1', 'MLP W0.1, D1']
        elif data_type == 'mixture3' or data_type == 'mixture3_dim10':
            methods = ['ConstantImputedLR', 'EMLR', 'ExpandedLR', 'MICELR',
                       'MLP W0.01, D1', 'MLP W0.1, D1', 'MLP W0.5, D1']
        else:
            methods = ['ConstantImputedLR', 'EMLR', 'ExpandedLR', 'MICELR',
                       'MLP W0.3, D1', 'MLP W1, D1', 'MLP W4, D1']

        # Note: for the curves to be plotted we need at least one of n_sizes or
        # p_sizes to be equal to 1.
        if len(p_sizes) == 1:
            p = p_sizes[0]
            idx = (scores_avg.p == p) & scores_avg.method.isin(methods)
        elif len(n_sizes) == 1:
            n = n_sizes[0]
            idx = (scores_avg.n == n) & scores_avg.method.isin(methods)
        else:
            print('One of n_sizes or p_sizes should be of length 1')

        scores_avg = scores_avg.loc[idx, :]

        if len(n_sizes) == 1:
            file_root = 'scaling_{}_n{}'.format(data_type, n_sizes[0])
        else:
            file_root = 'learning_curves_{}_p{}'.format(data_type, p_sizes[0])

        display_curves(scores_avg, bayes_rate, file_root, n_sizes,
                       p_sizes, do_legend=(data_type == 'mixture3'))
