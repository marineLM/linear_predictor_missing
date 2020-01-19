import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def display_scaling_q(scores_avg, bayes_rate, data_type, figname,
                      do_legend=True):

    p_sizes = scores_avg.p.unique()

    fig, axes = plt.subplots(1, 1, figsize=(4, 3.5))
    ax = axes
    # titles = ['train', 'test']
    titles = ['test']   # XXX: plot only test to simplify things
    for title in titles:
        # for p in p_sizes[::2]:
        for p in p_sizes:
            df_p = scores_avg.loc[scores_avg.p == p, :]

            # rescale with Bayes rate
            if bayes_rate is not None:
                br = bayes_rate[bayes_rate.p == p]
                df_p.r2 = br.r2.iloc[0] - df_p.r2

            if title == 'train':
                # Train
                sns.lineplot(data=df_p[df_p.train_test == title], x='q',
                             y='r2', ax=ax, label=None,
                             ci=None, legend=False,
                             style="train_test",
                             dashes=6 * [(1, 2)])
            else:
                sns.lineplot(data=df_p[df_p.train_test == title], x='q',
                             y='r2', ax=ax, label=p)

            if data_type == 'mixture1':
                ax.set_xlabel('$n_h/d$: nb of hidden units per dimension')
                ax.set_ylabel('Excess is test error: R2 - R2_Bayes')
            elif data_type == 'mixture3':
                ax.set_xlabel('$n_h/2^d$: nb of hidden units divided by $2^d$')
                ax.set_ylabel('Excess is test error: R2 - R2_Bayes')
            else:
                ax.set_xlabel('$n_h/d$: nb of hidden units per dimension')
                ax.set_ylabel('Test error: R2')

        if do_legend:
            legend = ax.legend(ncol=2)
            legend.set_title('d: input dimension')
        ax.grid(True)

    plt.tight_layout()
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

        # Choose a number of samples for which to plot the scaling in q
        if data_type == 'mixture3':
            n = int(0.75*500000)
        else:
            n = int(0.75*100000)

        # Filter the rows of the dataframe to keep only the rows with the
        # correct number of samples and methods.
        MLP_methods = [meth for meth in scores_avg.method.unique() if
                       'MLP' in meth]
        idx = (scores_avg.n == n) & (scores_avg.method.isin(MLP_methods))
        scores_avg = scores_avg.loc[idx, :]

        # Add a column with parameter q
        scores_avg['q'] = scores_avg['method'].apply(
            lambda s: float(s.split('W')[1].split(',')[0]))

        # Filter the rows for which the number of hidden units was strictly
        # less than 4
        if data_type == 'mixture3':
            idx = np.floor(scores_avg.q*2**scores_avg.p) > 3
            scores_avg = scores_avg[idx]
        elif data_type == 'selfmasked_proba' or data_type == 'mixture1':
            idx = np.floor(scores_avg.q*scores_avg.p) > 3
            scores_avg = scores_avg[idx]

        figname = 'MLP_scaling_q_{}_n{}'.format(
            data_type, n)
        display_scaling_q(scores_avg, bayes_rate, data_type, figname,
                          )
