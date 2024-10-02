import pandas as pd
import seaborn as sns
import pub_ready_plots as prp
import matplotlib.pyplot as plt

def main(vars):

    # Load QM7 datasets
    df_2DSingle_QM7 = pd.read_csv(vars['QM7_2DSingle'])
    df_2DMultiple_QM7 = pd.read_csv(vars['QM7_2DMultiple'])
    df_3DSingle_QM7 = pd.read_csv(vars['QM7_3DSingle'])
    df_3DMultiple_QM7 = pd.read_csv(vars['QM7_3DMultiple'])

    # Load QM9 datasets
    df_2DSingle_QM9 = pd.read_csv(vars['QM9_2DSingle'])
    df_2DMultiple_QM9 = pd.read_csv(vars['QM9_2DMultiple'])
    df_3DSingle_QM9 = pd.read_csv(vars['QM9_3DSingle'])
    df_3DMultiple_QM9 = pd.read_csv(vars['QM9_3DMultiple'])

    sns.set_style("whitegrid")
    sns.set_palette("pastel")

    with prp.get_context(
        layout=prp.Layout.NEURIPS,  
        width_frac=1,              
        height_frac=0.25,          
        nrows=2,                   
        ncols=2,                    
    ) as (fig, axs):    
        
        # QM7 Plots (First Row)
        line1, = axs[0, 0].plot(df_2DSingle_QM7.index[:-1], df_2DSingle_QM7['Mean'].iloc[:-1], label=f'2D', color='blue')
        line2, = axs[0, 0].plot(df_3DSingle_QM7.index[:-1], df_3DSingle_QM7['Mean'].iloc[:-1], label=f'3D', color='orange')
        axs[0, 0].set_title("Single Property Task", fontname='Times New Roman')
        axs[0, 0].set_ylabel(r"$\Delta \text{E} \:\: \text{kcal/mol}$", fontname='Times New Roman')
        axs[0, 0].set_ylim(-1200, -400)  # Set y-axis range
        axs[0, 0].tick_params(labelbottom=False)

        axs[0, 1].plot(df_2DMultiple_QM7.index[:-1], df_2DMultiple_QM7['Mean'].iloc[:-1], label=f'2D', color='blue')
        axs[0, 1].plot(df_3DMultiple_QM7.index[:-1], df_3DMultiple_QM7['Mean'].iloc[:-1], label=f'3D', color='orange')
        axs[0, 1].set_title("Transfer Learning Task", fontname='Times New Roman')
        axs[0, 1].set_ylim(-1200, -400)  # Set y-axis range
        axs[0, 1].tick_params(labelleft=False, labelbottom=False)

        # QM9 Plots (Second Row)
        axs[1, 0].plot(df_2DSingle_QM9.index[:-1], df_2DSingle_QM9['Mean'].iloc[:-1], label='2D', color='blue')
        axs[1, 0].plot(df_3DSingle_QM9.index[:-1], df_3DSingle_QM9['Mean'].iloc[:-1], label='3D', color='orange')
        axs[1, 0].set_ylabel(r"$\Delta \text{E}_{\text{gap}} \:\: \text{eV}$", fontname='Times New Roman')
        # axs[1, 0].set_ylim(6, 16)
        axs[1, 0].set_xlabel("Step", fontname='Times New Roman') 

        axs[1, 1].plot(df_2DMultiple_QM9.index[:-1], df_2DMultiple_QM9['Mean'].iloc[:-1], label='2D', color='blue')
        axs[1, 1].plot(df_3DMultiple_QM9.index[:-1], df_3DMultiple_QM9['Mean'].iloc[:-1], label='3D', color='orange')
        axs[1, 1].tick_params(labelleft=False)
        # axs[1, 1].set_ylim(6, 16)
        axs[1, 1].set_xlabel("Step", fontname='Times New Roman') 

        axs[1, 1].legend([line1, line2], ['2D', '3D'],
                         fontsize='small', borderaxespad=0.5, borderpad=0.5,
                         labelspacing=0.3, handlelength=2, handletextpad=0.5)

    fig.savefig(f"output_plot_type.pdf")
    # plt.show()

if __name__ == "__main__":
    variables = {
        'QM7_2DSingle': '~/Downloads/QM7/Type/2DSingle.csv',   
        'QM7_2DMultiple': '~/Downloads/QM7/Type/2DMultiple.csv',       
        'QM7_3DSingle': '~/Downloads/QM7/Type/3DSingle.csv',   
        'QM7_3DMultiple': '~/Downloads/QM7/Type/3DMultiple.csv',       
        'QM9_2DSingle': '~/Downloads/QM9/Type/2DSingle.csv',   
        'QM9_2DMultiple': '~/Downloads/QM9/Type/2DMultiple.csv',       
        'QM9_3DSingle': '~/Downloads/QM9/Type/3DSingle.csv',   
        'QM9_3DMultiple': '~/Downloads/QM9/Type/3DMultiple.csv',       
    }
    main(variables)
    