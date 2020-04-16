import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Closer ticks
plt.rcParams['ytick.major.pad'] = -2.5
plt.rcParams['pdf.fonttype'] = 42


def display_boxplots(scores_avg, b2yes_rate, file_root,
                     display_names=True):
    # XXX: remove ", D1" for the MLP
    scores_avg.method = scores_avg.method.str.replace(', D1', '')
    for scoring in ['mse', 'r2']:
        plt.figure(figsize=(3.9, 2.5))
        sns.set_style("whitegrid")
        ax = sns.boxplot(data=scores_avg,
                         hue='train_test',
                         palette=dict(train=(1, 1, 0), test='b'),
                         y='method',
                         x=scoring,
                         orient='h',
                         showfliers=False)
        # ax.set(xscale="log")

        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        legend = ax.legend()
        legend.set_title(None)

        # Horizontal greyed spans
        for i, _ in enumerate(plt.yticks()[0]):
            if not i % 2:
                continue
            ax.axhspan(i - .5, i + .5, color='.92', zorder=-2)

        if bayes_rate is not None:
            br = bayes_rate.iloc[0][scoring]
            ax.axvline(br, color="r")

        plt.axis('tight')
        if scoring == 'r2':
            # For r2 score, drawn ticker lines at 0 and 1
            xmin, xmax = plt.xlim()
            if xmax >= 1:
                ax.axvline(1, color=".8", zorder=0, lw=3)
                xmax = 1.02
            if xmin <= 0:
                ax.axvline(0, color=".8", zorder=0, lw=3)
            # XXX: Hack
            if xmin <= -1:
                xmin = -1
            plt.xlim(xmin=xmin, xmax=xmax)

        if scoring == 'r2':
            plt.xlim(xmin=xmin, xmax=xmax)

        if display_names:
            ticks = plt.xticks()[0]
            plt.xticks(ticks[::2], ['%.2f' % t for t in ticks[::2]])

        plt.ylabel(None)
        plt.xlabel('R2')
        plt.tight_layout(pad=.01)
        if not display_names:
            pos, names = plt.yticks()
            names = [name.get_text() for name in names]
            short_names = [(name.split(' ')[-1] if name.startswith('MLP')
                            else '') for name in names]
            plt.yticks(pos, short_names)
            # legend.set_visible(False)
        plt.savefig("../figures/{}_{}.pdf".format(file_root, scoring),
                    edgecolor='none', facecolor='none')
        plt.close()


if __name__ == '__main__':

    for plot_idx, data_type in enumerate(['mixture1', 'mixture3',
                                          'selfmasked_proba']):

        filename = 'allresultsPPCA_{}_10iter'.format(
            data_type)

        scores_avg = pd.read_csv('../results/' + filename + '_sa.csv',
                                 index_col=0)
        # Load bayes_rate if it could be computed
        # typically not for 'selfmasked' yet)
        try:
            file = open('../results/' + filename + '_br.csv', 'rb')
        except FileNotFoundError:
            bayes_rate = None
        else:
            bayes_rate = pd.read_csv(file, index_col=0)
            file.close()

        # Choose a dimension and a number of samples for which to plot the
        # boxplots
        p = 10
        n = int(0.75*1e5)

        # Choose the methods for which the boxplots will be plotted
        if data_type == 'mixture1':
            methods = ['ConstantImputedLR', 'EMLR', 'ExpandedLR', 'MICELR',
                       'MLP W0.3, D1', 'MLP W1, D1', 'MLP W4, D1']
            # methods= ['ConstantImputedLR', 'EMLR', 'ExpandedLR',
            #           'MLP W0.003, D1','MLP W0.01, D1', 'MLP W0.1, D1']
        elif data_type == 'mixture3':
            methods = ['ConstantImputedLR', 'EMLR', 'ExpandedLR', 'MICELR',
                       'MLP W0.01, D1', 'MLP W0.1, D1', 'MLP W0.5, D1']
        else:
            methods = ['ConstantImputedLR', 'EMLR', 'ExpandedLR', 'MICELR',
                       'MLP W0.3, D1', 'MLP W1, D1', 'MLP W4, D1']

        # Filter the rows of the dataframe to keep only the rows with the
        # correct number of samples, diemsion, and methods
        idx = (scores_avg.p == p) & (scores_avg.n == n) & \
            scores_avg.method.isin(methods)

        # Extract the bayes rate for the correct dimension
        if bayes_rate is not None:
            bayes_rate = bayes_rate[bayes_rate.p == p]

        scores_avg = scores_avg.loc[idx, :]

        file_root = 'boxplots_{}_n{}_dim{}'.format(
            data_type, n, p)
        display_boxplots(scores_avg, bayes_rate, file_root,
                         display_names=(plot_idx == 0))
