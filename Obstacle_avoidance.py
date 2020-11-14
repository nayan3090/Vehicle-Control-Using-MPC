import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = True

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 15
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference1 = [10, 0, 0]
        self.reference2 = None

        self.x_obs = 5
        self.y_obs = 0.1

    def plant_model(self,prev_state, dt, pedal, steering):
        # Update states
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]
        pedal_t = prev_state[4]
        steering_t = prev_state[5]

        # Introducing acceleration response lag
        pedal_lag = (pedal - pedal_t)/3
        a_t = pedal_t + pedal_lag

        # Introducing steering response lag
        steering_lag = (steering - steering_t)/6
        phi_t = steering_t + steering_lag


        # New states
        x_t_1 = x_t + v_t*np.cos(psi_t)*dt
        y_t_1 = y_t + v_t*np.sin(psi_t)*dt
        psi_t_1 = psi_t + v_t*(np.tan(phi_t)/2.5)*dt
        v_t_1 = v_t + a_t*dt - v_t/25

        return [x_t_1, y_t_1, psi_t_1, v_t_1, a_t, phi_t]

    def cost_function(self,u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0

        for k in range(0, self.horizon):
            state = self.plant_model(state, self.dt, u[k*2], u[k*2+1])
            # Cost function tuned to obtain optimized control action
            cost += abs(ref[0] - state[0])**2 + abs(ref[1] - state[1])**2 + abs(ref[2] - state[2])**2 + self.obstacle_distance(state[0], state[1])

        return cost

    # Obstacle avoidance   
    def obstacle_distance(self, x, y):
        distance = ((self.x_obs - x)**2 + (self.y_obs - y)**2)**0.5
        
        if (distance > 2):
            return 15
        else:
            return 1/distance*30

sim_run(options, ModelPredictiveControl)
