# IMPORTANT NOTE: THIS CODE IS WRITTEN FOR NACA 0012. IF YOU WANT TO USE IT FOR OTHER AIRFOILS, MODIFY DESIGN VARIABLES OF CPF PARAMETERIZATION #
import os.path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO # PPO algorithm is the default DRL algorithm. You can use other DRL algorithms available in stable_baselines3 package
import subprocess
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
import time
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import neuralfoil

#========== USER SETTINGS ==========#
#-----------------------------------#

# Working directory of the optimizer. The code writes and reads files in the working directory!!
## These directories are just sample. Modify them according to your system
### NOTE: Check existence of the directories!!!
HALL_OF_FAME = r"C:\CPFxDRL\training\Hall_of_Fame"

# Directory for saving optimization logs including final evaluated airfoil and monitoring csv files of cores during training phase
## This directory is just sample. Change it according to your system
LOG_DIR = r"C:\CPFxDRL\training\logs"

# Number of total working CPUs. If you want to use multi-processing set it larger than 1
## If you are working in Windows, it runs the code automatically on the most powerful cpu cores
### If you don't want to use multi-processing, just set this variable equal to 1
NUM_CPU = 12

# Number of total_steps
TOTAL_TIME_STEPS = 301056

#-----------------------------------#
#===================================#

# Main environment of airfoil optimization
class AirfoilCPFEnv(gym.Env):
    
    # Initial information of the optimization settings
    def __init__(self, env_rank=0):
        super(AirfoilCPFEnv, self).__init__()

        self.hall_of_fame = HALL_OF_FAME
        os.makedirs(self.hall_of_fame, exist_ok=True)

        # Number of design variables including a0, b4, c4, d0, x_Mu, y_Mu, x_Ml, y_Ml
        self.total_vars = 6

        self.baseline_cpf_vars = np.array([
            0.3003177, 0.0600172, #x_Mu, y_Mu
            0.3003177, #x_Ml
            0.030, 0.100, -.030 #Initial value for a0, b4, d0 (it is not required to calculate them for the actual NACA 0012, just give them a value to start)
            ], dtype=np.float32)

        # Baseline coordinates of critical points. Necessary for Affine transformation
        self.base_x_Mu = 0.3003177
        self.base_y_Mu = 0.0600172
        self.base_x_Ml = 0.3003177
        self.base_y_Ml = -.0600172

        # Adjust lower and upper bounds for your own airfoil
        ### NOTE: don't set lower bounds of critical point coordinates (either x and y) equal to 0, this may cause divide by zero errors.
        ### To capture all feasible airfoils, set minimum and maximum allowed bounds of x-coordinate to 0.00001 and 0.99999
        #### For better and faster convergence, limit bounds of x-coordinates of the critical points 
        ##### These default numbers are calculated for the NACA 0012 (refer to 'CPFxDRL: A Vast and Self-Intersection-Free Airfoil Optimization Framework' paper written by Ghorbani & Taghavi Zenouz)
        self.lower_bounds = np.array([0.15, 0.04, 0.15, 0.016, 0.001, -.052], dtype=np.float32)
        self.upper_bounds = np.array([0.85, 0.08, 0.85, 0.052, 0.458, -.016], dtype=np.float32)

        # Settings of action space of DRL algorithm. It is recommended not to change this section!!
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.total_vars,),
            dtype=np.float32
            )

        # Settings of observation space of DRL algorithm. It is recommended not to change this section!!
        self.observation_space = spaces.Box(
            low=self.lower_bounds,
            high=self.upper_bounds,
            shape=(self.total_vars,),
            dtype=np.float32
            )

        # Settings of points distribution on airfoil surface. It is cosine-based distribution (you can change it to other distributions)
        # 1. n_points is number of points on each upper and lower surfaces
        # 2. theta is mapped angle
        # 3. x is final x-coordinate of the points
        self.n_points = 100
        self.theta = np.linspace(0, np.pi, self.n_points)
        self.x = 0.5 * (1 - np.cos(self.theta))

        # Number of steps in each episode
        # It is recommended to set current_step to 0.
        # max_steps is maximum number of steps for each episode (you are free to change it depending on your system and problem)
        self.current_step = 0
        self.max_steps = 50

        # Default goal of optimization is to maximize lift-to-drag ratio. You can change it to your own optimization goal
        ## NOTE: Set the worst value as the initial value (like negative infinity or zero for the lift-to-drag ratio)
        self.best_reward = 0.0

    # Reset all information to their initial values at the end of episode
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
        self.current_vars = self.baseline_cpf_vars.copy()
        self.current_step = 0
        observation = self.current_vars.copy()
        info = {}
        return observation, info

    # Solving dependent coefficients of each segment based on assigned values of independent coefficients
    def _solve_dependent_coefficients(self, design_vars):
        x_Mu, y_Mu, x_Ml, a0, b4, d0 = design_vars
        y_Ml = y_Mu - 0.12

        # C2 continuity at L.E.
        m2 = (self.base_x_Mu) / (x_Mu)
        n2 = (y_Mu) / (self.base_y_Mu)
        m3 = (self.base_x_Ml) / (x_Ml)
        n3 = (y_Ml) / (self.base_y_Ml)
        c4 = -1 * ((n2 * np.sqrt(m2)) / (n3 * np.sqrt(m3))) * b4

        # Segment 1
        a1 = -11.00000 * a0 + 0.59083
        a2 = 43.33333 * a0 - 2.08033
        a3 = -70.37037 * a0 + 3.06481
        a4 = 37.03701 * a0 - 1.57531

        # Segment 2
        # suppress b0 because it is zero (in case of the NACA 0012)
        b1 = -3.421459 * b4 + 0.531773
        b2 = 7.595197 * b4 - 1.545054
        b3 = -7.587163 * b4 + 1.464461

        # Segment 3
        # suppress c0 because it is zero (in case of the NACA 0012)
        c1 = -3.421459 * c4 - 0.531773
        c2 = 7.595197 * c4 + 1.545054
        c3 = -7.587163 * c4 - 1.464461

        # Segment 4
        d1 = -11.00000 * d0 - 0.59083
        d2 = 43.33333 * d0 + 2.08033
        d3 = -70.37037 * d0 - 3.06481
        d4 = 37.03701 * d0 + 1.57531

        # Save calculated coefficients
        weights_s1 = [a0, a1, a2, a3, a4]
        weights_s2 = [b1, b2, b3, b4]
        weights_s3 = [c1, c2, c3, c4]
        weights_s4 = [d0, d1, d2, d3, d4]

        return weights_s1, weights_s2, weights_s3, weights_s4

    # Calculation of y-coordinate of the points
    def _generate_coordinates(self, design_vars):
        x_Mu, y_Mu, x_Ml = design_vars[:3]
        y_Ml = y_Mu - 0.12
        weights_s1, weights_s2, weights_s3, weights_s4 = self._solve_dependent_coefficients(design_vars)

        y_upper = np.zeros(self.n_points)
        y_lower = np.zeros(self.n_points)

        ## Baseline segment functions (before Affine mapping) (f_i(x))
        # Segment 1
        def f1(x):
            return (weights_s1[0] + (weights_s1[1] * x) + (weights_s1[2] * (x**2)) + (weights_s1[3] * (x**3)) + (weights_s1[4] * (x**4)))
        # Segment 2
        def f2(x):
            # note that this region starts from b1 : weights_s2[0] means b1 not b0 (this is designated to make the code faster. you can change it)
            x = np.clip(x, 0, None)
            return ((weights_s2[0] * x) + (weights_s2[1] * (x**2)) + (weights_s2[2] * (x**3)) + (weights_s2[3] * (x**0.5)))
        # Segment 3
        def f3(x):
            # note that this region starts from c1 : weights_s3[0] means c1 not c0 (this is designated to make the code faster. you can change it)
            x = np.clip(x, 0, None)
            return ((weights_s3[0] * x) + (weights_s3[1] * (x**2)) + (weights_s3[2] * (x**3)) + (weights_s3[3] * (x**0.5)))
        # Segment 4
        def f4(x):
            return (weights_s4[0] + (weights_s4[1] * x) + (weights_s4[2] * (x**2)) + (weights_s4[3] * (x**3)) + (weights_s4[4] * (x**4)))

        ## Final segment functions (after Affine mapping) (g_i(x))
        # Segment 1
        mask_up_aft = (self.x >= x_Mu)
        x_up_aft = self.x[mask_up_aft]
        m1 = (1.0 - self.base_x_Mu) / (1.0 - x_Mu)
        n1 = (y_Mu) / (self.base_y_Mu)
        def g1(x):
            return y_Mu + n1 * (f1(self.base_x_Mu + m1 * (x - x_Mu)) - self.base_y_Mu)
        y_upper[mask_up_aft] = g1(x_up_aft)

        # Segment 2
        mask_up_nose = (self.x < x_Mu)
        x_up_nose = self.x[mask_up_nose]
        m2 = (self.base_x_Mu) / (x_Mu)
        n2 = (y_Mu) / (self.base_y_Mu)
        def g2(x):
            return y_Mu + n2 * (f2(self.base_x_Mu + m2 * (x - x_Mu)) - self.base_y_Mu)
        y_upper[mask_up_nose] = g2(x_up_nose)

        # Segment 3
        mask_lo_nose = (self.x <= x_Ml)
        x_lo_nose = self.x[mask_lo_nose]
        m3 = (self.base_x_Ml) / (x_Ml)
        n3 = (y_Ml) / (self.base_y_Ml)
        def g3(x):
            return y_Ml + n3 * (f3(self.base_x_Ml + m3 * (x - x_Ml)) - self.base_y_Ml)
        y_lower[mask_lo_nose] = g3(x_lo_nose)

        # Segment 4
        mask_lo_aft = (self.x > x_Ml)
        x_lo_aft = self.x[mask_lo_aft]
        m4 = (1.0 - self.base_x_Ml) / (1.0 - x_Ml)
        n4 = (y_Ml) / (self.base_y_Ml)
        def g4(x):
            return y_Ml + n4 * (f4(self.base_x_Ml + m4 * (x - x_Ml)) - self.base_y_Ml)
        y_lower[mask_lo_aft] = g4(x_lo_aft)

        # To ensure leading edge and trailing edge have 0 y-coordinate. If your airfoil isn't limited to these conditions change these lines
        # Some airfoils such as Clark Y doesn't have trailing edges with y=0
        y_upper[0], y_lower[0] = 0.0, 0.0 # This line ensures that y_LE equals to zero
        y_upper[-1], y_lower[-1] = 0.0, 0.0 # This line ensures that y_TE equals to zero

        return y_upper, y_lower

    # Running flow-field solver around generated airfoils.
    # The solver is NeuralFoil.
    def _run_solver(self, y_upper, y_lower, Re=100000):
        # Prepare points in standard selig input format
        x_up = self.x[::-1]
        y_up = y_upper[::-1]
        x_low = self.x[1:]
        y_low = y_lower[1:]

        x_selig = np.concatenate([x_up, x_low])
        y_selig = np.concatenate([y_up, y_low])

        # Target variables
        ld_ratio_av = -10.0
        cd_value_av = 0.0
        cm_abs_max = 0.0
        cm_delta = 0.0

        # Design range of angles-of-attack
        alphas = np.linspace(0.0, 5.0, 51)

        # converting from selig format to NeuralFoil input format
        airfoil_coordinates = np.column_stack((x_selig, y_selig))

        # Solving flow-field
        aero_data = neuralfoil.get_aero_from_coordinates(
            coordinates=airfoil_coordinates,
            alpha=alphas,
            Re=Re,
            model_size="xxlarge"
            )

        # Extracting aerodynamic properties
        CL = aero_data['CL']
        CD = aero_data['CD']
        CM = aero_data['CM']

        # Calculation of target
        ld_ratio = CL / CD
        cd_value = CD

        ld_ratio_av = np.mean(ld_ratio)
        cd_value_av = np.mean(cd_value)
        cm_abs_max = np.max([np.abs(np.max(CM)), np.abs(np.min(CM))])
        cm_delta = np.abs(np.max(CM) - np.min(CM))

        return ld_ratio_av, cd_value_av, cm_abs_max, cm_delta

    # Action happens at each step
    def step(self, action):
        self.current_step += 1

        var_range = self.upper_bounds - self.lower_bounds
        step_size = var_range * 0.05
        self.current_vars += action * step_size
        self.current_vars = np.clip(self.current_vars, self.lower_bounds, self.upper_bounds)

        x_Mu, y_Mu, x_Ml, a0, b4, d0 = self.current_vars[0], self.current_vars[1], self.current_vars[2], self.current_vars[3], self.current_vars[4], self.current_vars[5]
        y_Ml = y_Mu - 0.12

        # C2 continuity at L.E.
        m2 = (self.base_x_Mu) / (x_Mu)
        n2 = (y_Mu) / (self.base_y_Mu)
        m3 = (self.base_x_Ml) / (x_Ml)
        n3 = (y_Ml) / (self.base_y_Ml)
        c4 = -1 * ((n2 * np.sqrt(m2)) / (n3 * np.sqrt(m3))) * b4

        y_upper, y_lower = self._generate_coordinates(self.current_vars)

        ld_ratio = 0.0
        cd = 0.0
        cm_delta = 0.0
        cm_abs_max = 0.0
        reward = ld_ratio

        terminated = False

        # Preventing self-intersecting geometries
        ## Although this phenomenon never happens with CPF, this condition is added just to ensure the code will not crash
        ### If you are certain to your parameterization(No human-generated faults), you can delete this condition
        if np.any(y_upper < y_lower):
            reward = -50.0
            terminated = True
            ld_ratio = -50.0
            cd = 0.0
            cm = 0.0
        else:
            ld_ratio, cd, cm_abs_max, cm_delta = self._run_solver(y_upper, y_lower)
            terminated = False

            # Given that current_vars[6] (c4) is calculated using other parameters to guarantee curvature continuity at L.E, it must be checked
            if cd <= 0.002 or ld_ratio == -10.0 or c4 < -.458 or c4 > 0.0: # 0.002 is approximation of a flat plate drag (Nothing has less drag compared to flat plate)
                reward = -10.0
            else:
                w1 = 250.0
                w2 = 250.0
                reward = ld_ratio - (w1 * cd) - (w2 * (cm_abs_max + cm_delta))

        truncated = self.current_step >= self.max_steps
        observation = self.current_vars.copy()

        info = {"L_over_D": ld_ratio, "Cd": cd, "|Cm|": cm_abs_max, "delta_Cm": cm_delta}

        if self.current_step % 10 == 0 and cd > 0.0:
            print(f"Step: {self.current_step} | L/D: {ld_ratio:.2f} | Cd: {cd:.5f} | Cm_delta: {cm_delta:.6f} | Cm_abs_max: {cm_abs_max:.6f}| reward: {reward:.5f}")


        #Hall of Fame#
        # Given that the DRL agent might generate better airfoils than final extracted geometry during the training phase, the best airfoils during training are saved in Hall_of_Fame
        if reward > self.best_reward and cd > 0.002:
            self.best_reward = reward
            pid = os.getpid()

            best_vars_path = os.path.join(self.hall_of_fame, f"Best_Vars_{pid}.txt")
            with open(best_vars_path, "w") as f:
                f.write(f"Record L/D: {ld_ratio:.2f} | Cd: {cd:.5f} | Cm_delta: {cm_delta:.6f} | Cm_abs_max: {cm_abs_max:.6f}\n")
                f.write(f"x_Mu, y_Mu, x_Ml, a0, b4, d0: \n")
                f.write(np.array2string(self.current_vars, separator=', '))

        return observation, reward, terminated, truncated, info

#------ Callback Management ------#
class GlobalProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, check_freq=100, verbose=0):
        super(GlobalProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            percent_complete = (self.num_timesteps / self.total_timesteps) * 100
            print(f"Total Steps: {self.num_timesteps} / {self.total_timesteps} ({percent_complete:.1f}%)")
        return True


#------ Stable Baselines 3 ------#
# Start running DRL model
def make_env(log_dir, rank):
    def _init():
        env = AirfoilCPFEnv(env_rank=rank)
        env = Monitor(env, os.path.join(log_dir, f"{rank}"))
        return env
    return _init

if __name__ == "__main__":

    # Author's information
    print(r"#=================================================#")
    print(r"#---         CPFxDRL Airfoil Optimizer         ---#")
    print(r"#---                Created by:                ---#")
    print(r"#---           MohammadHadi Ghorbani           ---#")
    print(r"#---              BSc. Student at              ---#")
    print(r"#---     School of Mechanical Engineering      ---#")
    print(r"#--- Iran University of Science and Technology ---#")
    print(r"#=================================================#")

    num_cpu = NUM_CPU
    
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    # Creating environment
    print(f"--- Creating Vectorized Environment with {num_cpu} Cores ---")
    env = SubprocVecEnv([make_env(log_dir, i) for i in range(num_cpu)])

    model = PPO("MlpPolicy", env, n_steps=512, verbose=1, learning_rate=0.0003, seed=0)

    # Starting training phase
    print("--- Start Learning (Parallel) ---")

    total_time_steps = TOTAL_TIME_STEPS

    t0_train = time.time()
    progress_callback = GlobalProgressCallback(total_timesteps=total_time_steps, check_freq=120)
    model.learn(total_timesteps=total_time_steps, callback=progress_callback)
    t1_train = time.time()

    train_time = t1_train - t0_train

    model.save(os.path.join(log_dir, "CPFxDRL"))
    print("--- Learning Completed ---")

    # Starting evaluation phase. This phase includes just 1 episode
    print("\n--- Start Extracting Optimum Airfoil ---")
    eval_env = AirfoilCPFEnv()

    obs, info = eval_env.reset()
    terminated = False
    truncated = False

    t0_eval = time.time()
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
    t1_eval = time.time()

    eval_time = t1_eval - t0_eval

    optimized_vars = eval_env.current_vars
    final_ld = info["L_over_D"]
    final_cd = info["Cd"]
    final_abs_max_cm = info["|Cm|"]
    final_del_cm = info["delta_Cm"]

    #Aerodynamic results
    print("\n========== Final Optimization Results ==========")
    print(f"Final L/D: {final_ld:.2f}")
    print(f"Final Cd : {final_cd:.5f}")
    print(f"Final max_abs_cm : {final_abs_max_cm:.6f}")
    print(f"Final delta_Cm : {final_del_cm:.6f}")
    print("Optimized CPF variables (x_Mu, y_Mu, x_Ml, a0, b4, c4, d0):")
    print(optimized_vars)
    print("================================================")

    #Computational results
    evals_per_sec = total_time_steps / train_time
    core_hours = (train_time * num_cpu) / 3600.0

    print("\n========== Computational Cost Report ==========")
    print(f"Total Geometries Evaluated : {total_time_steps}");
    print(f"Training Time              : {train_time:.2f} seconds")
    print(f"Optimization Extraction    : {eval_time:.4f} seconds")
    print(f"Processing Speed           : {evals_per_sec:.2f} Timesteps/Sec")
    print(f"Total CPU Workload         : {core_hours:.3f} Core-Hours")
    print("===============================================")

    #write results on .txt file#
    report_text = f"""
    ========== Final Optimization Results ==========
    Final L/D: {final_ld:.2f}
    Final Cd : {final_cd:.5f}
    Final max_abs_cm : {final_abs_max_cm:.6f}
    Final delta_Cm : {final_del_cm:.6f}
    Optimized CPF variables (x_Mu, y_Mu, x_Ml, a0, b4, c4, d0):
    {optimized_vars}
    ================================================

    ========== Computational Cost Report ===========
    Total Geometries Evaluated : {total_time_steps}
    Training Time              : {train_time:.2f} seconds
    Optimization Extraction    : {eval_time:.4f} seconds
    Processing Speed           : {evals_per_sec:.2f} Timesteps/Sec
    Total CPU Workload         : {core_hours:.3f} Core-Hours
    ================================================
    """
    
    results_file_path = os.path.join(log_dir, "results.txt")
    with open(results_file_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    print("Results saved")

