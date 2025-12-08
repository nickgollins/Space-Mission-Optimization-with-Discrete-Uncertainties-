import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gurobipy import GRB, Model
import pandas as pd
from tqdm import tqdm
import sys, os, json
import heapq

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from lib.campaign_planning.campaign_planner import CampaignPlanner, CampaignPlannerParameters
from lib.campaign_planning.lunar_logistics import LunarScheduling, LunarLogMissionParameters

def calc_pareto_probs(df:pd.DataFrame) -> list[float]:
    pareto_prob_list = []
    for index, design_entry in tqdm(df.iterrows(), total=len(df), desc="Processing pareto probabilities"):
        pareto_prob = 0
        if pd.notna(design_entry['expected_value_error']) and pd.notna(design_entry['expected_prob_infeasibility_error']) and pd.notna(design_entry['expected_prob_infeasibility']):
            for other_index, other_entry in df.iterrows():
                if index != other_index:
                    local_dom_prob = 0
                    # if pd.notna(other_entry['expected_value_error']) and pd.notna(other_entry['expected_value_error']):
                        # print(design_entry['expected_value'], design_entry['expected_value_error'], other_entry['expected_value'], other_entry['expected_value_error'])
                    expected_value_std_error = design_entry['expected_value_error']
                    m_obj = (other_entry['expected_value'] - design_entry['expected_value'])/design_entry['expected_value_error']
                    
                    s_obj = other_entry['expected_value_error']/expected_value_std_error
                    
                    # Probability that objective 1 of other point is less than objective 1 of design point
                    local_dom_prob += np.log(1-(1/(1 + np.exp(-2.5*m_obj/np.sqrt(2+2*s_obj**2)))))
                    # print(local_dom_prob)
                    expected_prob_infeas = design_entry['expected_prob_infeasibility']
                    expected_prob_infeas_std_error = design_entry['expected_prob_infeasibility_error']
                    m_infeas_prob = (other_entry['expected_prob_infeasibility'] - expected_prob_infeas)/np.sqrt(design_entry['expected_prob_infeasibility_error'])
                    s_infeas_prob = other_entry['expected_prob_infeasibility_error']/expected_prob_infeas_std_error
                   
                    # Probability that objective 2 of other point is less than objective 2 of design point
                    local_dom_prob += np.log(1-(1/(1 + np.exp(-2.5*m_infeas_prob/np.sqrt(2+2*s_infeas_prob**2)))))
                    pareto_prob += np.log(1-np.exp(local_dom_prob))
                    # print(m_infeas_prob, s_infeas_prob, local_dom_prob)
            assert np.exp(pareto_prob) >= 0 and np.exp(pareto_prob) <= 1, f"Probability of design point {design_entry['point']} being on the pareto front {np.exp(pareto_prob)} is not between 0 and 1, = {np.exp(pareto_prob)}"
            pareto_prob_list.append(pareto_prob)
            
            # print(design_entry['point'], design_entry['expected_prob_infeasibility'], np.exp(log_prob))
        else:
            pareto_prob_list.append(np.nan)
    return pareto_prob_list

def plot_network_flow(flow_data:np.array, 
                      node_labels:list[str], 
                      time_labels:list[str], 
                      title:str,
                      save_path=str
                    ):
    grid_cols = len(node_labels)-1
    grid_rows = len(time_labels)-1
    line_color = 'black'
    line_width = 1

    # Set the font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    
    fig, ax = plt.subplots(figsize=(4, 6))
    
    # Set the limits of the plot
    ax.set_xlim(0, grid_cols)
    ax.set_ylim(0, grid_rows)

    # Draw vertical lines
    for grid_col in np.linspace(-0.5, grid_cols+0.5, grid_cols+2, endpoint=True):
        ax.plot([grid_col, grid_col], [-0.5, grid_rows+0.5], color=line_color, linewidth=line_width)

    # Draw horizontal lines
    for grid_row in np.linspace(-0.5, grid_rows+0.5, grid_rows+2, endpoint=True):
        ax.plot([-0.5, grid_cols+0.5], [grid_row, grid_row], color=line_color, linewidth=line_width)

    # Remove axis lines (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Remove axis labels and ticks
    ax.set_xticks(range(-1, grid_cols+2))
    ax.set_yticks(range(-1, grid_rows+2))

    # Set custom tick labels, hiding the first and last
    ax.set_xticklabels([''] + [str(i) for i in range(grid_cols+1)] + [''])
    ax.set_yticklabels([''] + time_labels + [''])

    # Remove the small tick lines but keep the labels
    ax.tick_params(axis='both', which='both', length=0)

    # Move x-axis labels to the top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    ax.invert_yaxis()

    ax.set_xlabel('Node ID')
    ax.set_ylabel('Time Period')
    # Set aspect of the plot to be equal
    ax.set_aspect('equal')
    # ax.set_title(title)

    # Create a colormap
    norm = mcolors.Normalize(vmin=flow_data.min(), vmax=flow_data.max())
    cmap = cm.BuPu

    for (i,j,t) in np.ndindex(flow_data.shape):
        if flow_data[i,j,t] > 0.5:
            color = cmap(norm(flow_data[i,j,t]))
            start = (i, t)
            if j == i: end = (j, t+1)
            else: end = (j, t)
            ax.annotate('', xy=end, xytext=start, 
                        arrowprops=dict(edgecolor='black', arrowstyle='->', linewidth=1.7))
            ax.annotate('', xy=end, xytext=start, 
                        arrowprops=dict(edgecolor=color, arrowstyle='->', linewidth=1.5))

    # Add a colorbar to the new axis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = ax.inset_axes([1.04, 0.025, 0.05, 0.95])
    cbar = plt.colorbar(sm, cax=cax, shrink=0.8)
    cbar.set_label('Network Flow Quantity')

    # # Adjust the colorbar to match the height of the main plot
    # pos = cax.get_position()
    # cax.set_position([pos.x1 + 0.01, pos.y0+100, 0.02, pos.height-1])
    
    if np.all(flow_data == np.round(flow_data)) and flow_data.max() < 10:
        tick_values = np.arange(np.floor(flow_data.min()), np.ceil(flow_data.max()) + 1, 1)
        cbar.set_ticks(tick_values)

    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.show()

def plot_pareto_errors(num_points:int,
                       df:pd.DataFrame,
                       pareto_prob_list:list[float],
                       save_path=str
                    ) -> None:
    '''Calculate the probability of a design point being on the pareto front
    
    Args:
        :param design_entry: Design point to calculate the probability of being on the pareto front
    Returns:
        :return: Probability of the design point being on the pareto front
    '''
        

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.grid(True)

    # # Plot the data
    for i, design in enumerate(df.iterrows()):
        if pareto_prob_list[i] in heapq.nlargest(num_points, pareto_prob_list):
            ax.scatter(design[1]['expected_value'], 
                        design[1]['expected_prob_infeasibility'],
                        alpha=0.9,
                        c='rebeccapurple',
                        edgecolor='black',
                        )
            


            ax.errorbar(design[1]['expected_value'], 
                        design[1]['expected_prob_infeasibility'], 
                        xerr=design[1]['expected_value_error'],
                        yerr=design[1]['expected_prob_infeasibility_error'],
                        color='rebeccapurple',
                        alpha=0.3,
                        capsize=5,
                        fmt='.',
                        capthick=0.5,
                        markersize=5)

    ax.set_xlabel('Network Flow Model Expected Value (Total Campaign Launch Mass, kg)')
    ax.set_ylabel('Expected Probability of Infeasibility')
    ax.set_xlim([3.2E5,3.6E5])
    ax.set_ylim([0,1])
 
    # Set global font properties
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # Save the plot as a PDF
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()
    return 

def plot_pareto_prob(df:pd.DataFrame,
                     pareto_prob_list:list[float],
                     save_path=str
                    ) -> None:
    '''Calculate the probability of a design point being on the pareto front
    
    Args:
        :param design_entry: Design point to calculate the probability of being on the pareto front
    Returns:
        :return: Probability of the design point being on the pareto front
    '''

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = mcolors.Normalize(vmin=-50, vmax=max(pareto_prob_list))
    cmap = cm.Purples
    ax.grid(True)


    plt.scatter(df['expected_value'], 
                df['expected_prob_infeasibility'], 
                s=20,
                color=cmap(norm(pareto_prob_list)),
                edgecolor='black',
                linewidths=0.25
    )

    for i, design in enumerate(df.iterrows()):
        if pareto_prob_list[i] == max(pareto_prob_list):
            plt.scatter(design[1]['expected_value'],
                        design[1]['expected_prob_infeasibility'],
                        edgecolors='red',
                        facecolors='none',
                        s=60,
                        marker='o'
                        )
            
    ax.set_xlabel('Network Flow Model Expected Value (Total Campaign Launch Mass, kg)')
    ax.set_ylabel('Expected Probability of Infeasibility')
    ax.set_xlim([3.2E5,4E5])
    ax.set_ylim([0,1])
    # Create a ScalarMappable object for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Add the colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Order of Magnitude of Probability of Dominance')

    # Set global font properties
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # Save the plot as a PDF
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()
    return 

def plot_num_evals(df:pd.DataFrame,
                   save_path:str
                  ) -> None:
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = mcolors.LogNorm(vmin=1E-3, vmax=max(df['sum_total_prob']))
    cmap = cm.Purples
 
    ax.grid(True)



    plt.scatter(df['expected_value'],
                df['expected_prob_infeasibility'],
                s=20,
                color=cmap(norm(df['sum_total_prob'])),
                edgecolor='black',
                linewidths=0.25
                )
                
    ax.set_xlabel('Network Flow Model Expected Value (Total Campaign Launch Mass, kg)')
    ax.set_ylabel('Expected Probability of Infeasibility')
    ax.set_xlim([3.2E5,4E5])
    ax.set_ylim([0,1])

    # Create a ScalarMappable object for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Add the colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Fraction of Distribution Evaluated')

    # Set global font properties
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # Save the plot as a PDF
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()
    return

def plot_exp_vs_base(df:pd.DataFrame,
                   save_path:str
                  ) -> None:
    # Create a figure and axis
    fig, ax = plt.subplots()
    norm = mcolors.Normalize(vmin=0, vmax=max(df['expected_prob_infeasibility']))
    cmap = cm.Purples
    ax.grid(True)

    ax.scatter(df['base_objective'],
                df['expected_value'],
                s=norm(df['expected_prob_infeasibility'])*20,
                color=cmap(norm(df['expected_prob_infeasibility']))
                )
    
    ax.plot(np.linspace(3.2E5,4E5,100),
            np.linspace(3.2E5,4E5,100),
            color='grey',
            linewidth=1,
            linestyle='--'
            )   
    # ax.set_xlabel('Fraction of Distribution Evaluated')
    # ax.set_ylabel('Expected Probability of Infeasibility')
    ax.set_xlim([3.2E5,4E5])
    ax.set_ylim([3.2E5,4E5])

    # Create a ScalarMappable object for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Add the colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Expected Probability of Infeasibility')

    # Set global font properties
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # Save the plot as a PDF
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()
    return

def plot_networks(df:pd.DataFrame,
                 pareto_prob_list:list[float], 
                 problem:CampaignPlanner,
                 scheduling_model:LunarScheduling,
                 save_path:str):
    """"
    Plot the network flow of the solution with the greatest pareto probability in the results dataframe
    """
    
    best_index= [i for i, x in enumerate(pareto_prob_list) if x == max(pareto_prob_list)]
    for index, design_entry in enumerate(df.iterrows()):
        if index == best_index[0]:
            best_point = design_entry[1]['point']
            break
    best_point = best_point.replace("(", "").replace(")", "").split(", ")

    model, xI_index, xF_index = problem.pretty(best_point, output='output/case_studies/Artemis_MOO_sol.xlsx')
    data_dict = {}

    for n in scheduling_model.N:
        data_dict[f"Vehicle {n}"] = np.zeros((scheduling_model.number_nodes,
                                            scheduling_model.number_nodes,
                                            xI_index[-1][-1]+1))

    data_dict[1] = np.zeros((scheduling_model.number_nodes,
                            scheduling_model.number_nodes,
                            xI_index[-1][-1]+1))

    data_dict[5] = np.zeros((scheduling_model.number_nodes,
                                    scheduling_model.number_nodes,
                                    xI_index[-1][-1]+1))

    data_dict[4] = np.zeros((scheduling_model.number_nodes,
                                    scheduling_model.number_nodes,
                                    xI_index[-1][-1]+1))

    data_dict[3] = np.zeros((scheduling_model.number_nodes,
                                    scheduling_model.number_nodes,
                                    xI_index[-1][-1]+1))

    data_dict[2] = np.zeros((scheduling_model.number_nodes,
                                    scheduling_model.number_nodes,
                                    xI_index[-1][-1]+1))

    data_dict[7] = np.zeros((scheduling_model.number_nodes,
                                    scheduling_model.number_nodes,
                                    xI_index[-1][-1]+1))

    data_dict[6] = np.zeros((scheduling_model.number_nodes,
                                    scheduling_model.number_nodes,
                                    xI_index[-1][-1]+1))


    for ind in xI_index:
        if ind[-2] == 0:
            if ind[3] == 0:
                data_dict[f"Vehicle {ind[0]}"][ind[1], ind[2], ind[-1]] += model.getVarByName("xI[%d,%d,%d,%d,%d,%d]" % (ind)).X
            elif ind[3] == 1:
                data_dict[ind[3]][ind[1], ind[2], ind[-1]] += model.getVarByName("xI[%d,%d,%d,%d,%d,%d]" % (ind)).X

    for ind in xF_index:        
        if ind[-2] == 0:
            if (ind[1], ind[2]) != (0,0): 
                data_dict[ind[3]+2][ind[1], ind[2], ind[-1]] += model.getVarByName("xF[%d,%d,%d,%d,%d,%d]" % (ind)).X

    keys_list = list(data_dict.keys())

    sorted_x = np.unique(sorted(best_point))
    ylabels = []

    for i in range(len(sorted_x)):
        ylabels += [str(sorted_x[i])] + ['']

    title_dict={}
    for n in range(scheduling_model.number_vehicles):
        title_dict[f"Vehicle {n}"] = scheduling_model.spacecraft.name[n]

    title_dict[1] = "Crew"
    title_dict[2] = "Infrastructure"
    title_dict[3] = "Maintenance Supplies"
    title_dict[4] = "Crew Consumables"
    title_dict[5] = "Inert Payloads"
    title_dict[6] = "Oxidizer"
    title_dict[7] = "Fuel"

    for key in keys_list:
        plot_network_flow(data_dict[key], 
                          node_labels=[0,1,2,3,4], 
                          time_labels=ylabels,  
                          title=title_dict[key],
                          save_path=f"figures/{key}_flow.pdf")
        
def plot_decision_prob(problem:CampaignPlanner, 
                       df:pd.DataFrame, 
                       prob_list:list[float],
                       save_path:str
                    ) -> None:
    v_line_x = []
    v_line_y = []

    bounds = problem.get_bounds()
    for i, bound in enumerate(bounds[0]):
        v_line_x.append((bounds[0][i], bounds[0][i]))
        v_line_y.append((i, i+1))
        v_line_x.append((bounds[1][i]+1, bounds[1][i]+1))
        v_line_y.append((i, i+1))
        
    grid_data = np.zeros((19, 76))
    for i, design in enumerate(df.iterrows()):
        point = [int(x) for x in design[1]['point'].replace("(", "").replace(")", "").split(", ")]
        for j, x in enumerate(point):
            grid_data[j, x] += np.exp(prob_list[i])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the array onto the grid
    cax = ax.imshow(grid_data, 
                    norm=mcolors.LogNorm(vmin=10E-6, vmax=grid_data.max()),
                    cmap='Purples', 
                    interpolation='nearest',  
                    extent=[0, grid_data.shape[1], grid_data.shape[0], 0]
                )

    # Create a divider for the existing axes instance
    divider = make_axes_locatable(ax)

    # Append a new axis for the colorbar
    cax_colorbar = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and pad as needed

    # Add a colorbar to show the color scale
    cbar = fig.colorbar(cax, cax=cax_colorbar)
    cbar.set_label('Sum Prob. of Appearing \n in Dominant Solution', labelpad=20, loc='center')
    cbar.ax.yaxis.set_label_coords(3, 0.4)
    # Draw grid lines
    ax.set_xticks(np.arange(0, grid_data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(0, grid_data.shape[0],1), minor=True)

    # Draw specific vertical lines with limited height
    for i, line in enumerate(v_line_x):
        if i % 2 == 0:
            ax.plot(line, v_line_y[i], color='green', linewidth=2)
        else:
            ax.plot(line, v_line_y[i], color='r', linewidth=2)
    for x in range(19):
        ax.plot((0,77), (x,x), color='grey', linestyle='--', linewidth=0.5)

    ax.set_xlim(0,77)
    ax.set_ylim(0,19)
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Payload Num.')

    # Set global font properties
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

def plot_MIQCP(mem_file:str, 
               save_path:str
            ) -> None:
    import json

    mem_list = json.load(open(mem_file))

    fig, ax = plt.subplots(figsize=(10, 3))


    ax.bar(range(1, len(mem_list)+1),
            mem_list, 
            color='rebeccapurple', 
            label='Model Memory', 
            alpha=0.8, 
            edgecolor='black'
        )
    
    ax.plot((19, 19), (0, max(mem_list)), color='red', linestyle='--', linewidth=1)
    ax.set_xlim(0, 20)
    ax.set_xticks(range(1, 21))
    ax.set_xlabel('Number of Payloads in Campaign')
    ax.set_ylabel('Model Memory (GB)')
    ax.set_ylim(0, max(mem_list))

    ax.annotate('Full Artemis case study size', xy=(19, max(mem_list)/2), xytext=(18.75, max(mem_list)/2),
                rotation=90, horizontalalignment='center', verticalalignment='center')


    # Set global font properties
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12


    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

def plot_MIQCP_pareto(MIQCP_results_file:str,
                      save_path:str
                    ) -> None:
    array = np.loadtxt(MIQCP_results_file, delimiter=',')
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.grid(True)
    
    ax.scatter(array[:, 0], array[:, 1], color='indigo', alpha=1, edgecolor='black')
    ax.set_xlabel('Network Flow Model Expected Value (Total Campaign Launch Mass, kg)')
    ax.set_ylabel('Expected Probability of Infeasibility')

    # Set global font properties
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    plt.savefig(save_path, format='pdf', bbox_inches='tight')

    return

def plot_veh_usage_prob(problem:LunarScheduling,
                        vehicle_usage_file:str,
                        save_path:str,
                    ) -> None:
    with open(vehicle_usage_file, 'r') as f:
        num_vehs = np.array([json.loads(line)[1] for line in f])
    with open(vehicle_usage_file, 'r') as f:    
        probs = np.array([json.loads(line)[-1] for line in f])

    num_vehs[num_vehs < 1E-6] = 0  # Set very small values to zero

    d = {}

    for i, row in enumerate(num_vehs):
        key = tuple(int(j) for j in row)  # Convert row to tuple
    #  print(probs[i], key)
        if key not in d:
            d[key] = [probs[i]]
        else:
            d[key].append(probs[i])

    grid = np.zeros((max([max(key) for key in d.keys()])+2, len(list(d.keys())[0])))

    for key, value in d.items():
        prob = np.log(np.sum(np.exp(value)))
        for i, j in enumerate(key):
            if prob > -50:
                print(i,j, prob)
            if grid[j,i] == 0:
                grid[j,i] = prob
            else:
                grid[j,i] = np.log(np.exp(prob) + np.exp(grid[j,i]))

    grid[grid == 0 ] = np.nan  # Set zero values to NaN for better visualization
    fig, ax = plt.subplots()

    norm = mcolors.Normalize(vmin=-10, vmax=0)
    cmap = cm.Purples
    cmap.set_bad(color='white')  # Set color for NaN values

    cax = ax.imshow(grid, 
            cmap=cmap, 
            aspect='auto', 
            extent=(0, grid.shape[1], 0, grid.shape[0]),
            origin='lower',
            interpolation='nearest', 
            norm=norm
    )



    # Draw grid lines
    ax.set_xticks(np.arange(0, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(0, grid.shape[0],1), minor=True)
    ax.grid(True, which='both')

    x_labels = [problem.vehicle_data[i][0] for i in range(1, len(problem.vehicle_data))]
    ax.set_xticklabels('', rotation=90, ha='center')
    ax.set_yticklabels('    ', rotation=0, ha='center')
    ax.set_ylabel('Number of Vehicles Used')

    # Manually add x labels at shifted positions
    for i, label in enumerate(x_labels):
        ax.text(i + 0.5, -0.2, label, rotation=90, ha='center', va='top', transform=ax.transData)

    for i, label in enumerate(range(grid.shape[0])):
        ax.text(-0.3, i + 0.4, label, transform=ax.transData)

    cbar = fig.colorbar(cax, ax=ax, )
    cbar.set_label('Order of Magnitude of Probability\nof Appearing in Dominant Solution', labelpad=20, loc='center')

    # Set global font properties
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

    return




