class ExampleProblem:
    def __init__(self):
        return

    def fitness(self, x):
        obj = x[0]**3 + x[1]**2 + x[0]*x[1]
        return [obj] + [c for c in self.get_constrs(x)]
    
    def get_bounds(self):
        return ([-5]*2,[5]*2)
    
    def get_nec(self):
        return 0
    
    def get_nic(self):
        return 2
    
    def get_nix(self):
        return 2
    
    def get_constrs(self, x):
        ci1 = - x[0] + 1
        ci2 = - x[1] + 1
        return ci1, ci2

    def test_feasible(self, x):
        constr_check = all([i <= 0 for i in self.get_constrs(x)])
        bound_check = all([x[i] >= self.get_bounds()[0][i] and x[i] <= self.get_bounds()[1][i] for i in range(len(x))])
        return constr_check and bound_check
    
