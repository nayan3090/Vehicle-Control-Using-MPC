import numpy as np
from sim.sim1d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['FULL_RECALCULATE'] = False

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 20
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference = [50, 0, 0]

    def plant_model(self, prev_state, dt, pedal, steering):
        # Update states
        x_t = prev_state[0]
        v_t = prev_state[3] # m/s
        
        # New states
        x_t_1 = x_t + v_t*dt
        v_t_1 = v_t + pedal*dt - v_t/25

        return [x_t_1, 0, 0, v_t_1]

    def cost_function(self, u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0

        for i in range(0,self.horizon):
            state = self.plant_model(state, self.dt, u[i*2], u[i*2+1]) 
            
            # Cost function tuned to obtain optimized control action
            cost += abs(ref[0] - state[0])

            # Adding a penalty for not following highway speed limit
            if state[3]>(10/3.6):
                cost += cost*10000
        return cost

sim_run(options, ModelPredictiveControl)