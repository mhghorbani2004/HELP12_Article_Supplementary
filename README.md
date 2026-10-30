**HELP 12: A High-Efficacy Low-Pitching-Moment Airfoil Series Synthesized by the CPFxDRL-SuperFast Multipoint Optimization Architecture**



**Authors:** MohammadHadi Ghorbani, Reza Taghavi Zenouz



This repository contains the official source codes, and trained models for the paper "HELP 12: A High-Efficacy Low-Pitching-Moment Airfoil Series Synthesized by the CPFxDRL-SuperFast Multipoint Optimization Architecture"



**Overview:**

CPFxDRL-SuperFast: Airfoil Optimization Framework

This repository contains the source code and data for the paper "HELP 12: A High-Efficacy Low-Pitching-Moment Airfoil Series Synthesized by the CPFxDRL-SuperFast Multipoint Optimization Architecture". It introduces a novel airfoil series which balances the trade-off between lift-to-drag ratio and moment coefficient



**Repository Structure:**

The code is organized into distinct folders to ensure full reproducibility of the tables and results presented in the paper:

'Python' folder contains scripts of optimizer codes which are utilized to optimize airfoils in the paper

'C#' folder contains scripts of airfoil regenerator (using airfoil data files) and CST weights calculator codes

'XFOIL\_Commands' folder contains macros used in XFOIL in automated optimization loops

'Trained\_Models' folder contains trained models using CPFxDRL-Superfast framework during the research

'Coordinate\_Distributions of Generated\_Airfoils' folder contains coordinate distributions of final HELP 12 family airfoils



**Perquisites:**

Python 3.x: 'stable-baselines3', 'numpy', 'gym', 'NeuralFoil'

.NET Framework / Visual Studio: To compile and run the C# codes

XFOIL: Required for low-fidelity aerodynamic evaluation



**Pretrained Models:**

To save computational time and guarantee reproducibility, we have included the final trained PPO models in the 'Trained\_Models' directory. You can load these '.zip' models using the 'stable-baselines3' library to generate the optimized aerodynamic shapes instantly without re-running the extensive training episodes.



**Citation:**

If you find this code, data, or framework used in your research, please consider citing our paper:



title={HELP 12: A High-Efficacy Low-Pitching-Moment Airfoil Series Synthesized by the CPFxDRL-SuperFast Multipoint Optimization Architecture},



author={Ghorbani, MohammadHadi and Taghavi Zenouz, Reza},



year={2026}



