import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import numpy as np

plt.rcParams['pdf.fonttype'] = 42


def display_scaling_n(scores_avg, bayes_rate, q, data_type, figname):
    # Take one p out of two
    # scores_avg = scores_avg[scores_avg.p % 4 == 0]

    p_sizes = scores_avg.p.unique()
    # XXX: remove ", D1" for the MLP
    scores_avg.method = scores_avg.method.str.replace(', D1', '')

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(.75*4, .75*3.7))
    titles = ['train', 'test']
    is_legend = [False, 'brief']
    for p in p_sizes:

        # rescale with Bayes rate
        if bayes_rate is not None:
            br = bayes_rate[bayes_rate.p == p]

            scores_avg.loc[scores_avg.p == p, 'r2'] = -(
                br.r2.iloc[0] - scores_avg.loc[scores_avg.p == p, 'r2'])

        idx = (scores_avg.p == p) & (scores_avg.method != 'ExpandedLR')
        if data_type == 'mixture1':
            scores_avg.loc[idx, 'n'] /= (q*p)*(2*p+2)+1
            # scores_avg.loc[idx, 'n'] /= (q*2**p)*(2*p+2)+1
        elif data_type == 'mixture3':
            scores_avg.loc[idx, 'n'] /= (q*2**p)*(2*p+2)+1
        else:
            scores_avg.loc[idx, 'n'] /= (q*p)*(2*p+2)+1

        idx = (scores_avg.p == p) & (scores_avg.method == 'ExpandedLR')
        scores_avg.loc[idx, 'n'] /= (p*2**(p-1) + 2**p)

    for i, ax in enumerate(axes):
        sns.lineplot(data=scores_avg[scores_avg.train_test == titles[i]],
                     x='n', y='r2', ax=ax,  hue='p', style='method',
                     legend='brief',
                     estimator=np.median,
                     ci=False,
                     palette=sns.color_palette('hls', len(p_sizes)))

        handles, labels = ax.get_legend_handles_labels()

        ax.set_xlabel('Nb of training samples per parameter')
        if data_type == 'selfmasked_proba':
            ax.set_ylabel('R2')
            ax.set_xlim(left=0, right=30)
            if i == 0:
                ax.set_ylim(bottom=0.7, top=1.)
            else:
                ax.set_ylim(bottom=0, top=1.1)
        else:
            ax.set_ylabel('R2 - R2_Bayes')
            ax.set_xlim(left=0, right=100)
            if i == 0:
                ax.set_ylim(bottom=-0.12, top=0.1)
            else:
                ax.set_ylim(bottom=-0.2, top=0.05)
        # ax.set_title(titles[i])
        plt.text(.01, (.99 if i == 1 else .01), titles[i],
                 va=('top' if i == 1 else 'bottom'), ha='left',
                 weight='bold', size=14,
                 # bbox=dict(facecolor='white', alpha=0.5),
                 transform=ax.transAxes)
        # ax.legend().set_title('p: dimension')
        ax.grid(True)
        if i == 0:
            # Add the method legend on the first ax
            legend = ax.legend(
                handles=handles[-2:],
                labels=labels[-2:], loc='lower right',
                handletextpad=.3, title="method",
                handlelength=1.5, borderaxespad=.25, ncol=2)
            # plt.gca().add_artist(legend)

    first_legend = plt.legend(handles=handles[1:-3], labels=labels[1:-3],
                              loc='best', ncol=4,
                              handlelength=1.5, borderaxespad=.25,
                              handletextpad=.3,
                              title='d: number of features')
    ax = plt.gca().add_artist(first_legend)

    # plt.tight_layout(pad=.01, h_pad=0)
    plt.subplots_adjust(left=.07, right=1.1, bottom=.02, top=1.05,
                        hspace=.14)
    plt.savefig("../figures/{}.pdf".format(figname), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    for data_type in ['mixture1', 'mixture3', 'selfmasked_proba']:

        filename = 'allresultsPPCA_{}_10iter'.format(
            data_type)

        scores_avg = pd.read_csv('../results/' + filename + '_sa.csv',
                                 index_col=0)
        # Load bayes_rate if it could be computed
        # (typically not for 'selfmasked' yet)
        try:
            file = open('../results/' + filename + '_br.csv', 'rb')
        except FileNotFoundError:
            bayes_rate = None
        else:
            bayes_rate = pd.read_csv(file, index_col=0)
            file.close()

        # Choose a value for parameter q
        if data_type == 'mixture1':
            q = 2
            # q = 0.1
        elif data_type == 'mixture3':
            q = 0.5
        else:
            q = 2

        meth = 'MLP W{}, D1'.format(q)

        # Filter the rows of the dataframe to keep only the rows with the
        # correct methods.
        methods = [meth, 'ExpandedLR']
        scores_avg = scores_avg.loc[scores_avg.method.isin(methods)]

        figname = 'MLP_scaling_n_{}_q{}'.format(
            data_type, q)
        display_scaling_n(scores_avg, bayes_rate, q, data_type, figname)
