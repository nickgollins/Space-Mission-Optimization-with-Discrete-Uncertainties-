import numpy as np
import matplotlib.pyplot as plt
import matplotlib


SYNODIC_PERIOD = 29.530588853
DAYS_PER_SUNANGLE = SYNODIC_PERIOD / 360


def load_LETs(filepath:str="./data/NICK_LET_perilune_100km_target_idx1_1757_polar8595.txt",
              spacecraft=None,
              plotting:bool=False) -> list[list[int|float]]:
    
    """
    Load trajectory data from file, find the best suited trajectory for each spacecraft
    by calculating propellant savings, and at the best trajectory to the logistics network
    for that spacecraft.

    Inputs:
        - filepath: str, path to the trajectory data file
        - spacecraft: vehicleModel, spacecraft data
        - plotting: bool, whether to plot the data
    """

    # Load trajectory data
    data_matrix = np.loadtxt(filepath, delimiter=",")
    ID = data_matrix[:,0]                           # Trajectory ID
    tof = data_matrix[:,1]/86400/DAYS_PER_SUNANGLE  # Time of flight in sun angle degrees
    dep_sun_angle = data_matrix[:,2]*180/np.pi     # degrees
    arr_sun_angle = data_matrix[:,3]*180/np.pi     # degrees
    dV_LOI = data_matrix[:,7]                       # km/s
    C3 = data_matrix[:,4]                           # km^2/s^2

    num_spacecraft = len(spacecraft.name)

    # Calculate impact increased LTI cost
    mu = 3.986E5   # km^3/s^2
    g0 = 9.8  # m/s^2
    R_Earth = 6371  # km
    r_LEO = 185 + R_Earth # km
    r_Moon = 384472  # km
    v_LEO = np.sqrt(mu/r_LEO)  # km/s
    v_LTI = (C3+2*mu/r_LEO)**0.5 #  km/s
    dV_LTI = v_LTI - v_LEO
    # print(dV_LTI)

    # Initialize arrays
    m_prop_Ddirect = [0 for i in range(num_spacecraft)]
    m_prop_LOIdirect = [0 for i in range(num_spacecraft)]
    m_prop_LTIdirect = [0 for i in range(num_spacecraft)]
    m_prop_totaldirect = [0 for i in range(num_spacecraft)]
    m_propD = [[0 for t in tof] for i in range(num_spacecraft)]
    m_propLOI = [[0 for t in tof] for i in range(num_spacecraft)]
    m_propLTI = [[0 for t in tof] for i in range(num_spacecraft)]
    m_prop_total = [[0 for t in tof] for i in range(num_spacecraft)]
    m_prop_totaldirect = [0 for i in range(num_spacecraft)]
    delta_m_prop = [[] for i in range(num_spacecraft)]
    best_traj = [0 for i in range(num_spacecraft)]
    Z_LET = [0 for i in range(num_spacecraft)]
    LET_network = [[] for i in range(num_spacecraft)]

    # Same descent dV for all spacecraft / trajectories plans
    dV_D = 1.87  # km/s

    # Direct transfer dVs
    dV_LOIdirect = 0.893  # km/s
    dV_LTIdirect = 3.152  # km/s
    tof_direct = 3/DAYS_PER_SUNANGLE  # Assume 3 day direct transfer

    months, degrees = divmod(tof+dep_sun_angle, 360)
    LLO_time = tof_direct/DAYS_PER_SUNANGLE # LLO loitering time in sun angle so that we can assume direct transfer launches at 0 sun angle

    loiter_time = []
    for deg in degrees:
        if deg <= LLO_time:
            loiter_time.append(LLO_time - deg)
        else:
            loiter_time.append(LLO_time+ 360 - deg)
    loiter_time = np.array(loiter_time)

    # Process impact of LETs for each spacecraft
    for n in range(num_spacecraft):
        # Calculate overall change in propellant mass requirement
        for transfer, t in enumerate(loiter_time):  # Descent propellant
            # Prop requirements for LET ConOps
            m_propD[n][transfer] = (spacecraft.dry_mass[n] + spacecraft.payload_cap[n]) * (np.exp(dV_D*1000 / (spacecraft.Isp[n] * g0)) - 1)
            # Adjust for boiloff, with descent tof = loiter_time
            m_propD[n][transfer] = m_propD[n][transfer]*(spacecraft.oxy_ratio[n]/(1-spacecraft.oxy_boil_off_rate[n])**t
                                 + (1-spacecraft.oxy_ratio[n])/(1-spacecraft.fuel_boil_off_rate[n])**t)

        for transfer, t in enumerate(tof):  # LET propellant
            m_propLOI[n][transfer] = (spacecraft.dry_mass[n] + spacecraft.payload_cap[n] + m_propD[n][transfer]) * (np.exp(
                dV_LOI[transfer] * 1000 / (spacecraft.Isp[n] * g0)) - 1)
            # Adjust for boiloff, with LET tof
            m_propLOI[n][transfer] = (m_propLOI[n][transfer]+m_propD[n][transfer])*(spacecraft.oxy_ratio[n]/(1-spacecraft.oxy_boil_off_rate[n])**t
                                     + (1-spacecraft.oxy_ratio[n])/(1-spacecraft.fuel_boil_off_rate[n])**t) - m_propD[n][transfer]

            m_propLTI[n][transfer] = (spacecraft.dry_mass[n] + spacecraft.payload_cap[n] + m_propD[n][transfer]+ m_propLOI[n][transfer]) * (np.exp(dV_LTI[transfer] * 1000 / (spacecraft.launcher_Isp[n] * g0)) - 1)
            m_prop_total[n][transfer] = m_propD[n][transfer] + m_propLOI[n][transfer] + m_propLTI[n][transfer]

        # Prop requirements for direct ConOps
        m_prop_Ddirect[n] = (spacecraft.dry_mass[n] + spacecraft.payload_cap[n]) * (np.exp(dV_D*1000 / (spacecraft.Isp[n] * g0)) - 1)
        m_prop_LOIdirect[n] = (spacecraft.dry_mass[n] + spacecraft.payload_cap[n] + m_prop_Ddirect[n]) * (np.exp(dV_LOIdirect * 1000 / (spacecraft.Isp[n] * g0)) - 1)
        # Adjust for boiloff, with direct transfer tof = 3 days
        m_prop_LOIdirect[n] = m_prop_LOIdirect[n] * (spacecraft.oxy_ratio[n] / (1 - spacecraft.oxy_boil_off_rate[n]) ** tof_direct
                        + (1 - spacecraft.oxy_ratio[n]) / (1 - spacecraft.fuel_boil_off_rate[n]) ** tof_direct)

        m_prop_LTIdirect[n] = (spacecraft.dry_mass[n] + spacecraft.payload_cap[n] + m_prop_Ddirect[n] + m_prop_LOIdirect[n]) * (np.exp(dV_LTIdirect * 1000 / (spacecraft.launcher_Isp[n] * g0)) - 1)
        m_prop_totaldirect[n] = m_prop_Ddirect[n] + m_prop_LOIdirect[n] + m_prop_LTIdirect[n]


        delta_m_prop[n] = [(m - m_prop_totaldirect[n]) for m in m_prop_total[n]]

        best_traj[n] = delta_m_prop[n].index(min(delta_m_prop[n]))
        # Mass fraction for LET injection relative to direct LTI
        Z_LET[n] = np.exp((dV_LTI[best_traj[n]] - dV_LTIdirect) * 1000 / (spacecraft.launcher_Isp[n] * g0))

        # Add LEO -> WSB arc
        ## Discrete tof are *2 because of outbound/inbound alternating arcs
        LET_network[n].append(
            [[0, 5], [Z_LET[n], Z_LET[n]*100], [Z_LET[n], Z_LET[n], Z_LET[n], Z_LET[n], Z_LET[n], Z_LET[n]], 1, [0, tof[best_traj[n]], (dep_sun_angle[best_traj[n]]+tof[best_traj[n]])//360*2, 0]]
        )

        ## High cost version to test effect of the looser scheduling but with the LETs
        # LET_network[n].append(
        #     [[0, 3], [1E10, Z_LET[n] * 100], [Z_LET[n], Z_LET[n], Z_LET[n], Z_LET[n], Z_LET[n], Z_LET[n]], 1,
        #      [0, tof[best_traj[n]], (dep_sun_angle[best_traj[n]] + tof[best_traj[n]]) // 360 * 2, 0]]
        # )

        # WSB -> LLO arc
        if loiter_time[best_traj[n]] <= tof_direct:  # If loiter time is shorter than direct transfer time, discrete tof = 0
            LET_network[n].append(
                [[5, 3], [0, 0], [0, 0, 0, 0, 0, 0], 1, [0, loiter_time[best_traj[n]], 0, dV_LOI[best_traj[n]]]]
            )
        else:  # If loiter time is longer than direct transfer time, discrete tof = 1
            LET_network[n].append(
                [[5, 3], [0, 0], [0, 0, 0, 0, 0, 0], 1, [0, loiter_time[best_traj[n]], 1*2, dV_LOI[best_traj[n]]]]
            )
        # Fill in not-allowable arcs
        for not_allowed in [[1, 5], [2,5], [3, 5], [4, 5], [5, 5], [5, 0], [5,1], [5,2], [5, 3]]:
            LET_network[n].append([not_allowed, [0, 0], [0, 0, 0, 0, 0, 0], 0, [0, 0, 0, 0]])

    # Plotting
    if plotting==True:
        # Set the font name to Times New Roman
        # font_name = "Times New Roman"
        #
        # font_path = "C:/Windows/Fonts/"+font_name+".ttf"
        #
        # # Check if Times New Roman font is available, otherwise use a default font
        # if font_manager.get_font(font_path):
        #     plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
        # else:
        #     plt.rcParams["font.family"] = plt.rcParams["font.sans-serif"]

        # Set the default font to Times New Roman
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams.update({'font.size': 14})
        # Set the math font to Times New Roman
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
        matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

        plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=12)  # fontsize of the tick labels

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        im0 = ax.scatter(
            tof,
            C3,
            c=dV_LOI,
            cmap="viridis",
            edgecolors="black"
        )
        fig.colorbar(im0, label=r"$\Delta V_{LOI}$, km/s")
        ax.set(xlabel="Time of flight, $^o$ sun angle", ylabel="C3, km$^2$/s$^2$")
        ax.grid(True, alpha=0.5)
        fig.tight_layout()

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        im0 = ax.scatter(
            dep_sun_angle,
            arr_sun_angle,
            c=C3,
            cmap="viridis",
            edgecolors="black"
        )
        fig.colorbar(im0, label="C3, km$^2$/s$^2$")
        ax.set(xlabel="Departure time, $^o$ sun angle", ylabel="Arrival time, $^o$ sun angle")
        ax.grid(True, alpha=0.5)
        fig.tight_layout()

        for n in range(num_spacecraft):
            print("Best trajectory for spacecraft", n, ":", ID[best_traj[n]])
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            im0 = ax.scatter(
                (loiter_time + tof),
                dV_LTI + dV_LOI,
                c=delta_m_prop[n],
                cmap="viridis",
                edgecolors="black"
            )
            im1 = ax.scatter(
                (loiter_time[best_traj[n]] + tof[best_traj[n]]),
                dV_LTI[best_traj[n]] + dV_LOI[best_traj[n]],
                marker="o",
                facecolors='none', edgecolors='r',
                s=100
            )
            fig.colorbar(im0, label="Delta Propellant Mass, kg")
            ax.set(xlabel="Time of flight + loitering time in LLO, $^o$ sun angle", ylabel=r"$\Delta V_{LTI}+\Delta V_{LOI}$, km/s",
                   title=spacecraft.name[n] + " - Propellant type: " + spacecraft.prop_type[n])
            ax.grid(True, alpha=0.5)
            fig.tight_layout()
        plt.show()

    return LET_network



