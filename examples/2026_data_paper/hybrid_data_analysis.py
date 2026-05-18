import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('mosden.plotting')


def bar_plot_gen(df, nu, nu_err, hl, hl_err, dataset, save_mod=''):
    ax = sns.barplot(df, x=dataset, y=nu, palette='viridis')
    x_coords = [p.get_x() + p.get_width() / 2 for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=df[nu_err],
                fmt='none', capsize=3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    plt.xlabel('Dataset')
    plt.ylim((0, np.max(1.1*df[nu])))
    plt.tight_layout()
    plt.savefig(f'./{save_mod}_yield_bar.png')
    plt.close()


    ax = sns.barplot(df, x=dataset, y=hl, palette='viridis')
    x_coords = [p.get_x() + p.get_width() / 2 for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=df[hl_err],
                fmt='none', capsize=3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    plt.xlabel('Dataset')
    plt.ylim((0, np.max(1.1*df[hl])))
    plt.tight_layout()
    plt.savefig(f'./{save_mod}_hl_bar.png')
    plt.close()


def bar_plots(base_df: pd.DataFrame):
    cfy = r'$CFY$'
    tau = r'$\tau$'
    pn = r'$P_n$'
    nu = r'$\nu_d (I)$'
    nu_err = r'$\Delta \nu_d (I)$'
    hl = r'$\bar{\tau} (I)$ $[s]$'
    hl_err = r'$\Delta \bar{\tau} (I)$ $[s]$'

    df = base_df.loc[base_df[cfy] == base_df[tau]]
    df = df.loc[df[cfy] == df[pn]]
    iaea_df_row = base_df.loc[base_df[cfy] == 'JENDL-5'].loc[base_df[pn] == 'IAEA'].loc[base_df[tau] == 'IAEA']
    pure_df = pd.concat((df, iaea_df_row))
    
    bar_plot_gen(pure_df, nu, nu_err, hl, hl_err, pn, save_mod='pure')

    iaea_df = base_df.loc[base_df[pn] == 'IAEA'].loc[base_df[tau] == 'IAEA']
    df = iaea_df

    bar_plot_gen(iaea_df, nu, nu_err, hl, hl_err, cfy, save_mod='iaea')


    
    return None

def heatmap_plot(base_df: pd.DataFrame):
    x = r'$CFY$'
    y = r'$P_n$'
    z = r'$\nu_d (I)$'
    numin = np.min(base_df[z])
    numax = np.max(base_df[z])
    for tau in set(base_df[r'$\tau$']):
        df = base_df.loc[base_df[r'$\tau$'] == tau]
        X = df[x]
        Y = df[y]
        Z = df[z]
        df = pd.DataFrame.from_dict(np.array([X, Y, Z]).T)
        df.columns = [x, y, z]
        pivotted = df.pivot(index=x, columns=y, values=z)
        color = sns.color_palette("viridis", as_cmap=True)
        pivotted = pivotted.astype(float)
        ax = sns.heatmap(pivotted, cmap=color, vmin=numin, vmax=numax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
        ax.invert_yaxis()
        ax.collections[0].colorbar.set_label(z)
        plt.tight_layout()
        plt.savefig(f'./yield_surf_tau_{tau.replace('/','')}.png')
        plt.close()

    return None


def process_data(data_path: str):
    df = pd.read_csv(data_path)
    bar_plots(df)
    heatmap_plot(df)

    return None



if __name__ == '__main__':
    data_table_path = './data/images/data.csv'
    process_data(data_table_path) 