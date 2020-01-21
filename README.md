# Normative Multitasking Model

This repository contains code for my undergraduate thesis work (adv. Jonathan D. Cohen, Yael Niv) on normative modelling of multitasking capacity limitations. A subset of this work was initially presented at the Annual Meeting of the Cognitive Science Society 2018 under the title "Efficiency of learning vs. processing: Towards a normative theory of multitasking".

Key:

.*_regression_agent.py contain code for the task/agent simulations. 
In logistic_regression_agent.py, the agent infers a sigmoid for the probability of success function, whereas in
linear_regression_agent.py, they infer a line.

path.py, explore_parameters.py collect data by running the above simulations with a given set of parameters. 

.*.cmd run the data collection scripts on the cluster.

plot_parameters.ipynb, ceq_plot.m plot the figures for the paper.
