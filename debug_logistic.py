import readline
import numpy as np

import math
import sys

from random import randint
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector

######### DATA PROCESSING FUNCTIONS

# given some number of tasks, return the reward for those tasks
def get_reward(num_tasks, t, is_tensor, true_params):
	true_V, true_P, true_b0, true_b1 = true_params
	R = []
	T = []
	if is_tensor:
		reward = 0
		for i in range(num_tasks):
			if np.random.uniform(0, 1) < lin_logistic(t, true_b1, true_b0):
				reward += true_V

		if not reward == 0:
			R = [reward]
			T = [0]
	else:
		for i in range(num_tasks):
			if np.random.uniform(0, 1) < lin_logistic(t, true_b1, true_b0):
				R.append(true_V - i * true_P)
				T.append(i)

	return (R, T)

# expected discounted reward over the experiment time window
# gamma is discount factor
def expected_discounted_reward(b0, b1, V, P, gamma, t_0, T, max_tasks, task_dist):
	DISCOUNT_SCALE_FACTOR = 0.025
	s = 0
	for t in range(t_0, T + 1): 
		discount_factor = gamma ** ((t - t_0) * DISCOUNT_SCALE_FACTOR)
		s += (discount_factor * expected_trial_reward(b0, b1, V, P, t, max_tasks, task_dist))
	
	return s

# expected reward on a particular trial
def expected_trial_reward(b0, b1, V, P, t, max_tasks, task_dist):
	s = 0
	for i in np.arange(1, max_tasks + 1):
		prob_task = task_dist[i - 1]
		prob_success = lin_logistic(t, b1, b0)
		num_terms = i / 2
		sum_term = (2 * V) - ((i - 1) * P)

		s += prob_task * prob_success * num_terms * sum_term

	return s

######### VARIOUS UTILITY FUNCTIONS

# logistic form 1 / 1 + exp(-(k * (x - x0)))
def logistic(x, x0, k):
	return expit(k * (x - x0))

# logistic form 1 / 1 + exp(-(b_1 * x + b_0))
def lin_logistic(x, b_1, b_0):
	return expit(b_1 * x + b_0)

# translate k(x - m) to b0 + x * b_1
def km_to_b0b1(k, m):
	return (-m * k, k)

# translate b0 + x * b_1 to k(x - m) 
def b0b1_to_km(b0, b1):
	return (b1, -b0 / b1)

# using softmax, pick an action
def pick_action(basis_val, tensor_val, params, algorithm="softmax"): 
	if algorithm == "softmax":
		tau = params
		vals = np.array([basis_val, tensor_val]) # hack so i can copy paste from RL agent code
		denom = np.sum(np.exp(vals / tau))
		dist = np.exp(vals / tau) / denom

		choice = np.random.choice(2, p=dist)
	else:
		eps = params
		if np.random.uniform(0, 1) < eps:
			choice = np.random.choice(2)
		elif basis_val > tensor_val:
			choice = 0
		else:
			choice = 1

	return choice

def plot_reg(reg, x=None, data=None):
	T = np.arange(0, 200).reshape(-1, 1)
	try: 
		plt.scatter([x], data)
		probs = reg.predict_proba(T)
		print(probs[:, 1].shape)
		plt.plot(T, probs[:, 1])
	except:
		probs = reg.predict_proba(T)
		plt.plot(T, probs[:, 1])

#### main script

# experiment parameters
max_tasks = 4                     # max number of concurrent tasks
num_trials = 1000                 # number of experiment trials
true_V = 1                        # reward for correctness
true_P = float(sys.argv[1])       # punishment for time delay
prop_mt = float(sys.argv[2])      # proportion of multitasking trials
gamma = float(sys.argv[3])        # reward discount factor
epsilon = 0.1                     # parameter for epsilon greedy algorithm

# training parameters
basis_true_k = 0.01                                # logistic training gradient
basis_true_m = num_trials * float(sys.argv[4])     # logistic training midpoint
basis_true_b0, basis_true_b1 = km_to_b0b1(basis_true_k, basis_true_m) # translate gradient/midpoint to linear parameters

tensor_true_k = 0.01                               # logistic training gradient
tensor_true_m = num_trials * float(sys.argv[5])    # logistic training midpoint
tensor_true_b0, tensor_true_b1 = km_to_b0b1(tensor_true_k, tensor_true_m) # translate gradient/midpoint to linear parameters

true_tasks = np.zeros((max_tasks,)) # true task distribution
for i in range(max_tasks):
	# if i == 0:
	# 	true_tasks[i] = 1 - prop_mt
	# else:
	# 	true_tasks[i] = prop_mt / (max_tasks - 1)
	if i == 2:
		true_tasks[i] = 1
	else:
		true_tasks[i] = 0

basis_true_params = (true_V, true_P, basis_true_b0, basis_true_b1)
tensor_true_params = (true_V, 0, tensor_true_b0, tensor_true_b1)

## various prior parameters

# Dirichlet parameter
prior_alpha = np.ones((max_tasks,)) # uniform

# Logistic regression parameters, basis
prior_midpoint = num_trials / 2
prior_gradient = 0.1
basis_logistic_prior_means = np.array(km_to_b0b1(prior_gradient, prior_midpoint))
basis_logistic_prior_Hessian = np.ones((2,)) * 0.001 # very uncertain

# Logistic regression parameters, tensor
prior_midpoint = num_trials / 2
prior_gradient = 0.1
tensor_logistic_prior_means = np.array(km_to_b0b1(prior_gradient, prior_midpoint))
tensor_logistic_prior_Hessian = np.ones((2,)) * 0.001 # very uncertain

# rpy2 setup
base = importr('base')
utils = importr('utils')
arm = importr('arm')

# simulation setup
R = [true_V, true_V] # list of accumulated rewards
T = [0, 0] # list of timesteps at which rewards were accumulated
b_X = [num_trials - 1, 0]  # list of times
b_Y = [1, 0]  # list of successes at those times
t_X = [num_trials - 1, 0]  # list of times
t_Y = [1, 0]  # list of successes at those times
task_counts = np.zeros((max_tasks,)) # list of tasks sampled
task_counts[0] = 4

# initial representation reward values
basis_val = 0
tensor_val = 0

basis_pick_count = 2
tensor_pick_count = 2

# simulation loop
for t in np.arange(2, num_trials):
	# pick number of tasks
	num_tasks = np.random.choice(np.arange(1, max_tasks + 1), p=true_tasks)

	# pick representation according to softmax over values
	is_tensor = (pick_action(basis_val, tensor_val, params=epsilon, algorithm="e-greedy") == 1) # pick_action returns 1 for tensor

	# get reward for all tasks
	if is_tensor:
		trial_R, trial_T = get_reward(num_tasks, tensor_pick_count, is_tensor, tensor_true_params)
		
		# add data to success history
		for i in range(len(trial_T)):
			t_X.append(tensor_pick_count)
			t_Y.append(1)

		for i in range(num_tasks - len(trial_T)):
			t_X.append(tensor_pick_count)
			t_Y.append(0)

		tensor_pick_count += 1

	else:
		trial_R, trial_T = get_reward(num_tasks, basis_pick_count, is_tensor, basis_true_params)

		# add data to success history
		for i in range(len(trial_T)):
			b_X.append(basis_pick_count)
			b_Y.append(1)

		for i in range(num_tasks - len(trial_T)):
			b_X.append(basis_pick_count)
			b_Y.append(0)

		basis_pick_count += 1

	# add data to reward history
	R += trial_R
	T += trial_T

	# infer V, P based off of reward history
	# infer_V, infer_P = sample_vp(np.array(R).reshape(-1, 1), np.array(T).reshape(-1, 1), vp_prior_params)
	infer_V = true_V
	infer_P = true_P

	# infer logistic training curve parameters based off of success	
	b_X_rpy2 = FloatVector(b_X).r_repr()
	b_Y_rpy2 = FloatVector(b_Y).r_repr()
	robjects.r('''
		b_data = data.frame(%s, cbind(%s, 1 - %s))
		b_reg = bayesglm((cbind(%s, 1-%s)) ~ %s, b_data, family="binomial", prior.mean = %f, prior.mean.for.intercept = %f)
		''' % (b_X_rpy2, b_Y_rpy2, b_Y_rpy2, b_Y_rpy2, b_Y_rpy2, b_X_rpy2, basis_true_b1, basis_true_b0))

	b_reg = robjects.r['b_reg']

	t_X_rpy2 = FloatVector(t_X).r_repr()
	t_Y_rpy2 = FloatVector(t_Y).r_repr()
	robjects.r('''
		t_data = data.frame(%s, cbind(%s, 1 - %s))
		t_reg = bayesglm((cbind(%s, 1-%s)) ~ %s, t_data, family="binomial", prior.mean = %f, prior.mean.for.intercept = %f)
		''' % (t_X_rpy2, t_Y_rpy2, t_Y_rpy2, t_Y_rpy2, t_Y_rpy2, t_X_rpy2, tensor_true_b1, tensor_true_b0))

	t_reg = robjects.r['t_reg']

	# infer task distribution based off of num_tasks
	infer_task_dist = true_tasks

	# recompute expected basis discounted reward values
	basis_val = float(expected_discounted_reward(b_reg.rx2('coefficients')[0], b_reg.rx2('coefficients')[1], infer_V, infer_P, gamma, basis_pick_count, num_trials - tensor_pick_count, max_tasks, infer_task_dist))
	# basis_val = float(expected_discounted_reward(basis_true_b0, basis_true_b1, infer_V, infer_P, gamma, basis_pick_count, num_trials - tensor_pick_count, max_tasks, infer_task_dist))
	

	# recompute expected tensor reward value -- infer_P = 0 since you're parallelizing the execution
	tensor_val = float(expected_discounted_reward(t_reg.rx2('coefficients')[0], t_reg.rx2('coefficients')[1], infer_V, 0, gamma, tensor_pick_count, num_trials - basis_pick_count, max_tasks, infer_task_dist))
	# tensor_val = float(expected_discounted_reward(tensor_true_b0, tensor_true_b1, infer_V, 0, gamma, tensor_pick_count, num_trials - basis_pick_count, max_tasks, infer_task_dist))

	# print('basis_val = %f' % basis_val)
	# print('tensor_val = %f' % tensor_val)

	# sys.exit()
	# woo, logging.
	# print("Iteration %d:\n" % t)
	# print("V = %f\nP = %f\n" % (infer_V, infer_P))

	# b_k, b_m = b0b1_to_km(b_reg.rx2('coefficients')[0], b_reg.rx2('coefficients')[1])
	# t_k, t_m = b0b1_to_km(t_reg.rx2('coefficients')[0], t_reg.rx2('coefficients')[1])

	# print("Tensor pick count: %d\nTensor value: %f\nTensor logistic params: (%f, %f)\nTruth: (%f, %f)\n" % (tensor_pick_count, tensor_val, t_reg.rx2('coefficients')[0], t_reg.rx2('coefficients')[1], tensor_true_b0, tensor_true_b1))
	# print("Basis pick count: %d\nBasis value: %f\nBasis logistic params: (%f, %f)\nTruth: (%f, %f)\n" % (basis_pick_count, basis_val, b_reg.rx2('coefficients')[0], b_reg.rx2('coefficients')[1], basis_true_b0, basis_true_b1))

print("%d,%d" % (tensor_pick_count, basis_pick_count))
