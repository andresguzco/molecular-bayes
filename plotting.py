import pandas as pd
import seaborn as sns
import pub_ready_plots as prp
import matplotlib.pyplot as plt

def main(vars):
    # Load QM7 Data
    df_GP_QM7 = pd.read_csv(vars['GP_QM7'])
    df_RS_QM7 = pd.read_csv(vars['RS_QM7'])
    df_3D_QM7 = pd.read_csv(vars['3D_QM7'])
    df_2D_QM7 = pd.read_csv(vars['2D_QM7'])
    df_3DGP_QM7 = pd.read_csv(vars['3DGP_QM7'])
    df_2DGP_QM7 = pd.read_csv(vars['2DGP_QM7'])
    df_3DLaplace_QM7 = pd.read_csv(vars['3DLaplace_QM7'])
    df_2DLaplace_QM7 = pd.read_csv(vars['2DLaplace_QM7'])

    # Load QM9 Data
    df_GP_QM9 = pd.read_csv(vars['GP_QM9'])
    df_RS_QM9 = pd.read_csv(vars['RS_QM9'])
    df_3D_QM9 = pd.read_csv(vars['3D_QM9'])
    df_2D_QM9 = pd.read_csv(vars['2D_QM9'])
    df_3DGP_QM9 = pd.read_csv(vars['3DGP_QM9'])
    df_2DGP_QM9 = pd.read_csv(vars['2DGP_QM9'])
    df_3DLaplace_QM9 = pd.read_csv(vars['3DLaplace_QM9'])
    df_2DLaplace_QM9 = pd.read_csv(vars['2DLaplace_QM9'])

    sns.set_style("whitegrid")
    sns.set_palette("pastel")

    with prp.get_context(
        layout=prp.Layout.NEURIPS,  
        width_frac=1,              
        height_frac=0.25,   
        nrows=2,                   
        ncols=3                                
    ) as (fig, axs):
        
        # First row (QM7 dataset)
        # Aggregated Performance QM7
        line1, = axs[0, 0].plot(df_GP_QM7.index[:-1], df_GP_QM7['Mean'].iloc[:-1], label=f'GP', color='red')
        line2, = axs[0, 0].plot(df_RS_QM7.index[:-1], df_RS_QM7['Mean'].iloc[:-1], label=f'RS', color='green')
        line3, = axs[0, 0].plot(df_3D_QM7.index[:-1], df_3D_QM7['Mean'].iloc[:-1], label=f'3D', color='blue')
        line4, = axs[0, 0].plot(df_2D_QM7.index[:-1], df_2D_QM7['Mean'].iloc[:-1], label=f'2D', color='orange')
        axs[0, 0].set_title("Aggregated Performance", fontname='Times New Roman')
        axs[0, 0].set_ylabel(r"$\Delta \text{E} \:\: \text{kcal/mol}$", fontname='Times New Roman')
        axs[0, 0].set_ylim(-1200, -400)  # Set y-axis range
        axs[0, 0].tick_params(labelbottom=False)


        # GP Regressor QM7
        axs[0, 1].plot(df_3DGP_QM7.index[:-1], df_3DGP_QM7['Mean'].iloc[:-1], label=f'3D', color='blue')
        axs[0, 1].plot(df_2DGP_QM7.index[:-1], df_2DGP_QM7['Mean'].iloc[:-1], label=f'2D', color='orange')
        axs[0, 1].set_title('GP', fontname='Times New Roman')
        axs[0, 1].set_ylim(-1200, -400)  # Set y-axis range
        axs[0, 1].tick_params(labelleft=False, labelbottom=False)


        # LLA Regressor QM7
        axs[0, 2].plot(df_3DLaplace_QM7.index[:-1], df_3DLaplace_QM7['Mean'].iloc[:-1], label=f'3D', color='blue')
        axs[0, 2].plot(df_2DLaplace_QM7.index[:-1], df_2DLaplace_QM7['Mean'].iloc[:-1], label=f'2D', color='orange')
        axs[0, 2].set_title('LLA', fontname='Times New Roman')
        axs[0, 2].set_ylim(-1200, -400)  # Set y-axis range
        axs[0, 2].tick_params(labelleft=False, labelbottom=False)


        # Second row (QM9 dataset)
        # Aggregated Performance QM9
        line5, = axs[1, 0].plot(df_GP_QM9.index[:-1], df_GP_QM9['Mean'].iloc[:-1], label=f'GP', color='red')
        line6, = axs[1, 0].plot(df_RS_QM9.index[:-1], df_RS_QM9['Mean'].iloc[:-1], label=f'RS', color='green')
        line7, = axs[1, 0].plot(df_3D_QM9.index[:-1], df_3D_QM9['Mean'].iloc[:-1], label=f'3D', color='blue')
        line8, = axs[1, 0].plot(df_2D_QM9.index[:-1], df_2D_QM9['Mean'].iloc[:-1], label=f'2D', color='orange')
        axs[1, 0].set_xlabel("Step", fontname='Times New Roman') 
        axs[1, 0].set_ylabel(r"$\Delta \text{E}_{\text{gap}} \:\: \text{eV}$", fontname='Times New Roman') 

        # GP Regressor QM9
        axs[1, 1].plot(df_3DGP_QM9.index[:-1], df_3DGP_QM9['Mean'].iloc[:-1], label=f'3D', color='blue')
        axs[1, 1].plot(df_2DGP_QM9.index[:-1], df_2DGP_QM9['Mean'].iloc[:-1], label=f'2D', color='orange')
        axs[1, 1].set_xlabel('Step', fontname='Times New Roman') 
        axs[1, 1].tick_params(labelleft=False)

        # LLA Regressor QM9
        axs[1, 2].plot(df_3DLaplace_QM9.index[:-1], df_3DLaplace_QM9['Mean'].iloc[:-1], label=f'3D', color='blue')
        axs[1, 2].plot(df_2DLaplace_QM9.index[:-1], df_2DLaplace_QM9['Mean'].iloc[:-1], label=f'2D', color='orange')
        axs[1, 2].set_xlabel('Step', fontname='Times New Roman') 
        axs[1, 2].tick_params(labelleft=False)

        # Adding legend to the first row
        axs[1, 2].legend([line1, line2, line3, line4], ['GP', 'RS', '3D', '2D'],
                      fontsize='small', borderaxespad=0.5, borderpad=0.5,
                      labelspacing=0.3, handlelength=2, handletextpad=0.5)

    fig.savefig(f"output_plot.pdf")
    # plt.show()

if __name__ == "__main__":
    variables = {
        # QM7 data paths
        'RS_QM7': '~/Downloads/QM7/General/RS.csv',   
        'GP_QM7': '~/Downloads/QM7/General/GP.csv',       
        '3D_QM7': '~/Downloads/QM7/General/3D.csv',
        '2D_QM7': '~/Downloads/QM7/General/2D.csv',
        '3DGP_QM7': '~/Downloads/QM7/General/3DGP.csv',
        '2DGP_QM7': '~/Downloads/QM7/General/2DGP.csv',
        '3DLaplace_QM7': '~/Downloads/QM7/General/3DLaplace.csv',
        '2DLaplace_QM7': '~/Downloads/QM7/General/2DLaplace.csv',

        # QM9 data paths
        'RS_QM9': '~/Downloads/QM9/General/RS.csv',   
        'GP_QM9': '~/Downloads/QM9/General/GP.csv',       
        '3D_QM9': '~/Downloads/QM9/General/3D.csv',
        '2D_QM9': '~/Downloads/QM9/General/2D.csv',
        '3DGP_QM9': '~/Downloads/QM9/General/3DGP.csv',
        '2DGP_QM9': '~/Downloads/QM9/General/2DGP.csv',
        '3DLaplace_QM9': '~/Downloads/QM9/General/3DLaplace.csv',
        '2DLaplace_QM9': '~/Downloads/QM9/General/2DLaplace.csv',
    }
    main(variables)