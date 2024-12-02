import pandas as pd
import seaborn as sns
import pub_ready_plots as prp
import matplotlib.pyplot as plt

def main(vars):

    # Load QM7 datasets
    df_2D500_QM7 = pd.read_csv(vars['QM7_2D500'])
    df_2D1K_QM7 = pd.read_csv(vars['QM7_2D1K'])
    df_2D10K_QM7 = pd.read_csv(vars['QM7_2D10K'])
    df_2D50K_QM7 = pd.read_csv(vars['QM7_2D50K'])
    df_3D500_QM7 = pd.read_csv(vars['QM7_3D500'])
    df_3D1K_QM7 = pd.read_csv(vars['QM7_3D1K'])
    df_3D10K_QM7 = pd.read_csv(vars['QM7_3D10K'])
    df_3D50K_QM7 = pd.read_csv(vars['QM7_3D50K'])

    # Load QM9 datasets
    df_2D500_QM9 = pd.read_csv(vars['QM9_2D500'])
    df_2D1K_QM9 = pd.read_csv(vars['QM9_2D1K'])
    df_2D10K_QM9 = pd.read_csv(vars['QM9_2D10K'])
    df_2D50K_QM9 = pd.read_csv(vars['QM9_2D50K'])
    df_3D500_QM9 = pd.read_csv(vars['QM9_3D500'])
    df_3D1K_QM9 = pd.read_csv(vars['QM9_3D1K'])
    df_3D10K_QM9 = pd.read_csv(vars['QM9_3D10K'])
    df_3D50K_QM9 = pd.read_csv(vars['QM9_3D50K'])

    sns.set_style("whitegrid")
    sns.set_palette("pastel")

    with prp.get_context(
        layout=prp.Layout.NEURIPS,  
        width_frac=1,              
        height_frac=0.25,          
        nrows=2,                   
        ncols=4,                    
    ) as (fig, axs):    
        
        # QM7 Plots (First Row)
        line1, = axs[0, 0].plot(df_2D500_QM7.index[:-1], df_2D500_QM7['Mean'].iloc[:-1], label=f'2D', color='blue')
        line2, = axs[0, 0].plot(df_3D500_QM7.index[:-1], df_3D500_QM7['Mean'].iloc[:-1], label=f'3D', color='orange')
        axs[0, 0].set_title(r"$N=500$", fontname='Times New Roman')
        axs[0, 0].set_ylabel(r"$\Delta \text{E} \:\: \text{kcal/mol}$", fontname='Times New Roman')
        axs[0, 0].set_ylim(-1200, -400)  # Set y-axis range
        axs[0, 0].tick_params(labelbottom=False)

        axs[0, 1].plot(df_2D1K_QM7.index[:-1], df_2D1K_QM7['Mean'].iloc[:-1], label=f'2D', color='blue')
        axs[0, 1].plot(df_3D1K_QM7.index[:-1], df_3D1K_QM7['Mean'].iloc[:-1], label=f'3D', color='orange')
        axs[0, 1].set_title(r"$N=1000$", fontname='Times New Roman')
        axs[0, 1].set_ylim(-1200, -400)  # Set y-axis range
        axs[0, 1].tick_params(labelleft=False, labelbottom=False)

        axs[0, 2].plot(df_2D10K_QM7.index[:-1], df_2D10K_QM7['Mean'].iloc[:-1], label=f'2D', color='blue')
        axs[0, 2].plot(df_3D10K_QM7.index[:-1], df_3D10K_QM7['Mean'].iloc[:-1], label=f'3D', color='orange')
        axs[0, 2].set_title(r"$N=10000$", fontname='Times New Roman')
        axs[0, 2].set_ylim(-1200, -400)  # Set y-axis range
        axs[0, 2].tick_params(labelleft=False, labelbottom=False)

        axs[0, 3].plot(df_2D50K_QM7.index[:-1], df_2D50K_QM7['Mean'].iloc[:-1], label=f'2D', color='blue')
        axs[0, 3].plot(df_3D50K_QM7.index[:-1], df_3D50K_QM7['Mean'].iloc[:-1], label=f'3D', color='orange')
        axs[0, 3].set_title(r"$N=50000$", fontname='Times New Roman')
        axs[0, 3].set_ylim(-1200, -400)  # Set y-axis range
        axs[0, 3].tick_params(labelleft=False, labelbottom=False)

        # QM9 Plots (Second Row)
        axs[1, 0].plot(df_2D500_QM9.index[:-1], df_2D500_QM9['Mean'].iloc[:-1], label='2D', color='blue')
        axs[1, 0].plot(df_3D500_QM9.index[:-1], df_3D500_QM9['Mean'].iloc[:-1], label='3D', color='orange')
        axs[1, 0].set_ylabel(r"$\Delta \text{E}_{\text{gap}} \:\: \text{eV}$", fontname='Times New Roman')
        # axs[1, 0].set_ylim(6, 16)
        axs[1, 0].set_xlabel("Step", fontname='Times New Roman') 

        axs[1, 1].plot(df_2D1K_QM9.index[:-1], df_2D1K_QM9['Mean'].iloc[:-1], label='2D', color='blue')
        axs[1, 1].plot(df_3D1K_QM9.index[:-1], df_3D1K_QM9['Mean'].iloc[:-1], label='3D', color='orange')
        axs[1, 1].tick_params(labelleft=False)
        # axs[1, 1].set_ylim(6, 16)
        axs[1, 1].set_xlabel("Step", fontname='Times New Roman') 

        axs[1, 2].plot(df_2D10K_QM9.index[:-1], df_2D10K_QM9['Mean'].iloc[:-1], label='2D', color='blue')
        axs[1, 2].plot(df_3D10K_QM9.index[:-1], df_3D10K_QM9['Mean'].iloc[:-1], label='3D', color='orange')
        axs[1, 2].tick_params(labelleft=False)
        # axs[1, 2].set_ylim(6, 16)
        axs[1, 2].set_xlabel("Step", fontname='Times New Roman') 

        axs[1, 3].plot(df_2D50K_QM9.index[:-1], df_2D50K_QM9['Mean'].iloc[:-1], label='2D', color='blue')
        axs[1, 3].plot(df_3D50K_QM9.index[:-1], df_3D50K_QM9['Mean'].iloc[:-1], label='3D', color='orange')
        axs[1, 3].tick_params(labelleft=False)
        # axs[1, 3].set_ylim(6, 16)
        axs[1, 3].set_xlabel("Step", fontname='Times New Roman') 

        axs[1, 3].legend([line1, line2], ['2D', '3D'],
                         fontsize='small', borderaxespad=0.5, borderpad=0.5,
                         labelspacing=0.3, handlelength=2, handletextpad=0.5)

    fig.savefig(f"output_plot_app.pdf")
    # plt.show()

if __name__ == "__main__":
    variables = {
        'QM7_2D500': '~/Downloads/QM7/Sample Complexities/2D500.csv',   
        'QM7_2D1K': '~/Downloads/QM7/Sample Complexities/2D1K.csv',       
        'QM7_2D10K': '~/Downloads/QM7/Sample Complexities/2D10K.csv',
        'QM7_2D50K': '~/Downloads/QM7/Sample Complexities/2D50K.csv',
        'QM7_3D500': '~/Downloads/QM7/Sample Complexities/3D500.csv',   
        'QM7_3D1K': '~/Downloads/QM7/Sample Complexities/3D1K.csv',       
        'QM7_3D10K': '~/Downloads/QM7/Sample Complexities/3D10K.csv',
        'QM7_3D50K': '~/Downloads/QM7/Sample Complexities/3D50K.csv',
        'QM9_2D500': '~/Downloads/QM9/Sample Complexities/2D500.csv',   
        'QM9_2D1K': '~/Downloads/QM9/Sample Complexities/2D1K.csv',       
        'QM9_2D10K': '~/Downloads/QM9/Sample Complexities/2D10K.csv',
        'QM9_2D50K': '~/Downloads/QM9/Sample Complexities/2D50K.csv',
        'QM9_3D500': '~/Downloads/QM9/Sample Complexities/3D500.csv',   
        'QM9_3D1K': '~/Downloads/QM9/Sample Complexities/3D1K.csv',       
        'QM9_3D10K': '~/Downloads/QM9/Sample Complexities/3D10K.csv',
        'QM9_3D50K': '~/Downloads/QM9/Sample Complexities/3D50K.csv',
    }
    main(variables)
    