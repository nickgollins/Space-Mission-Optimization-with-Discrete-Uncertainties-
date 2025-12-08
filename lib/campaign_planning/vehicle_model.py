from .LET import DAYS_PER_SUNANGLE

class VehicleModel(object):
    def __init__(self):
        self.name = []
        self.prop_cap = []
        self.payload_cap = []
        self.dry_mass = []
        self.Alpha = []  # structural fraction
        self.Isp = []  # specific impulse
        self.launch_frequency = []
        self.available_from = []
        self.launch_cost = []
        self.domain = []
        self.mixture_ratio = []
        self.oxy_ratio = []
        self.oxy_boil_off_rate = []
        self.fuel_boil_off_rate = []
        self.number_non_stack_vehicles = 0
        self.prop_type = []
        self.launcher_Isp = []

        return

    def define_stack(self, stack) -> None:  
        '''
          Extract stack definitions. Args are vehicles included in the stack. Active element must go first.
        '''
        
        self.name.append("Stack {}".format(stack))
        self.payload_cap.append(self.payload_cap[int(stack[0])] - sum(self.dry_mass[i] for i in stack[1:len(stack)]))
        self.prop_cap.append(sum(self.prop_cap[i] for i in stack))
        self.dry_mass.append(sum(self.dry_mass[i] for i in stack))
        self.Isp.append(self.Isp[int(stack[0])])
        self.launcher_Isp.append(self.launcher_Isp[int(stack[0])])
        self.domain.append(self.domain[int(stack[0])])
        self.mixture_ratio.append(self.mixture_ratio[int(stack[0])])
        self.oxy_ratio.append(self.oxy_ratio[int(stack[0])])
        self.oxy_boil_off_rate.append(self.oxy_boil_off_rate[int(stack[0])])
        self.fuel_boil_off_rate.append(self.fuel_boil_off_rate[int(stack[0])])
        # print(self.name[-1], self.payload_cap[-1], self.prop_cap[-1], self.dry_mass[-1], self.Isp[-1], self.domain[-1])
        return

    def get_stacks(self, stack_data) -> None:
        for stack in stack_data:
            self.define_stack(stack)
        return


def get_vehicle_data(vehicle_data_file) -> VehicleModel:
    '''
        Initialise vehicle object
    '''
    spacecraft = VehicleModel()
    i = 0
    for row in vehicle_data_file:
        if i > 0:  # Row 0 is headers
            assert row[10] in ["Storable", "CH4/LOX", "LH2/LOX"], f"Invalid propellant type {row[10]} in vehicle {row[0]} data."
            spacecraft.name.append(row[0])
            spacecraft.payload_cap.append(float(row[1]))
            spacecraft.prop_cap.append(float(row[2]))
            spacecraft.dry_mass.append(float(row[3]))
            spacecraft.Isp.append(float(row[4]))
            if row[10] == "Storable": 
                spacecraft.prop_type.append("Storable")
                spacecraft.mixture_ratio.append(2.61)  # N2O4/UDMH  http://www.astronautix.com/n/n2o4udmh.html
                spacecraft.oxy_boil_off_rate.append(0*DAYS_PER_SUNANGLE)  # Storable prop has no boil-off
                spacecraft.fuel_boil_off_rate.append(0*DAYS_PER_SUNANGLE)
            elif row[10] == "CH4/LOX": 
                spacecraft.prop_type.append("Methalox")
                spacecraft.mixture_ratio.append(3.6)  # LOX/CH4  https://www.faa.gov/space/stakeholder_engagement/spacex_starship/media/Appendix_G_Exhaust_Plume_Calculations.pdf
                spacecraft.oxy_boil_off_rate.append(0.025/100*DAYS_PER_SUNANGLE)  # Oxygen % boil-off per degree sun angle, from Chai and Whilhite 2014
                spacecraft.fuel_boil_off_rate.append(0.08/100*DAYS_PER_SUNANGLE)  # Methane % boil-off per day
            elif row[10] == "LH2/LOX": 
                spacecraft.prop_type.append("LH2/LOX")
                spacecraft.mixture_ratio.append(6)  # https://www.nasa.gov/returntoflight/system/system_SSME.html#:~:text=Each%20Space%20Shuttle%20Main%20Engine,of%20213%2C188%20(470%2C000%20pounds).
                spacecraft.oxy_boil_off_rate.append(0.025/100*DAYS_PER_SUNANGLE)   # Oxygen % boil-off per day, from Chai and Whilhite 2014
                spacecraft.fuel_boil_off_rate.append(0.1/100*DAYS_PER_SUNANGLE)   # Hydrogen % boil-off per day, from Chai and Whilhite 2014
            spacecraft.oxy_ratio.append(spacecraft.mixture_ratio[-1]/(1+spacecraft.mixture_ratio[-1]))
            spacecraft.launch_frequency.append(int(row[5]))
            spacecraft.available_from.append(int(row[6]))
            spacecraft.launch_cost.append(float(row[7]))
            domain_list = []
            for arc in row[8].split('], ['):
                if arc != '':
                    arc = arc.replace('[','')
                    arc = arc.replace(']','')
                    domain_list.append([int(k) for k in arc.split(',')])
            spacecraft.domain.append(domain_list)
            spacecraft.launcher_Isp.append(float(row[9]))
            # print('Sorted row ', i, ' of vehicle data.')
        i += 1
    spacecraft.number_non_stack_vehicles = i-1
    return spacecraft



