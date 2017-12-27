import readline
import numpy as np

import math
import sys

from random import randint
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

######### V, P SAMPLING FUNCTIONS

# sample v and p values from the posterior
def sample_vp(R, T, prior_params):
	burn_in = 100
	sample_iterations = 500

	ndim = 2
	nwalkers = 50
	pos = [np.random.randn(ndim) for i in range(nwalkers)]

	sampler = mc.EnsembleSampler(nwalkers, ndim, lnprob, args=(R, T, prior_params))
	sampler.run_mcmc(pos, sample_iterations)
	samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))

	return samples[-1, :]

######### PROPMT SAMPLING FUNCTIONS

# sample a multinomial task distribution for the number of tasks
def sample_multinomial(counts, prior_alpha):
	multinom = np.random.dirichlet(counts + prior_alpha)
	return multinom

######### P(SUCCESS) SAMPLING FUNCTIONS

# run weighted least squares on predictors X for Y
def infer_wls_params(X, Y, has_intercept = True, eta=0.2):
	weights = []
	for i in range(len(X)):
		weights.insert(0, eta * ((1 - eta) ** i))

	if has_intercept:
		X = sm.add_constant(X, has_constant='add')

	wls_model = sm.WLS(Y, X, weights = weights)
	results = wls_model.fit()

	if has_intercept:
		return results.params[0], results.params[1]
	else:
		return 0, results.params[0]

def infer_ols_params(X, Y, has_intercept = True, eta=None):
	if has_intercept:
		X = sm.add_constant(X, has_constant='add')

	ols_model = sm.OLS(Y, X)
	results = ols_model.fit()

	if has_intercept:
		return results.params[0], results.params[1]
	else:
		return 0, results.params[0]


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
def expected_discounted_reward(grad, intercept, V, P, gamma, t_0, T, max_tasks, task_dist):
	DISCOUNT_SCALE_FACTOR = 0.025
	s = 0
	for t in range(t_0, T + 1): 
		discount_factor = gamma ** ((t - t_0) * DISCOUNT_SCALE_FACTOR)
		s += (discount_factor * expected_trial_reward(grad, intercept, V, P, t, max_tasks, task_dist))
	
	return s

# expected reward on a particular trial
def expected_trial_reward(grad, intercept, V, P, t, max_tasks, task_dist):
	s = 0
	for i in np.arange(1, max_tasks + 1):
		prob_task = task_dist[i - 1]
		prob_success = lin(grad, t, intercept)
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

def lin(grad, t, intercept=0):
	val = intercept + (grad * t)

	if val >= 1:
		val = 1
	if val <= 0:
		val = 0

	return val

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
		if np.random.uniform(0, 1) < eps or tensor_val == basis_val:
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
eta = 0.5                         # recency weighting parameter

# training parameters 
basis_true_k = 0.02                                # logistic training gradient
basis_true_m = num_trials * float(sys.argv[4])     # logistic training midpoint
basis_true_b0, basis_true_b1 = km_to_b0b1(basis_true_k, basis_true_m) # translate gradient/midpoint to linear parameters

tensor_true_k = 0.02                               # logistic training gradient
tensor_true_m = num_trials * float(sys.argv[5])    # logistic training midpoint
tensor_true_b0, tensor_true_b1 = km_to_b0b1(tensor_true_k, tensor_true_m) # translate gradient/midpoint to linear parameters

true_tasks = np.zeros((max_tasks,)) # true task distribution
for i in range(max_tasks):
	if i == 0:
		true_tasks[i] = 1 - prop_mt
	else:
		true_tasks[i] = prop_mt / (max_tasks - 1)

basis_true_params = (true_V, true_P, basis_true_b0, basis_true_b1)
tensor_true_params = (true_V, 0, tensor_true_b0, tensor_true_b1)

## various prior parameters

# Dirichlet parameter
prior_alpha = np.ones((max_tasks,)) # uniform

# Reward inference parameters
v_prior_mean = true_V
v_prior_var = 0.1

p_prior_mean = true_P
p_prior_var = 0.001

vp_prior_params = (v_prior_mean, v_prior_var, p_prior_mean, p_prior_var)

# simulation setup
R = []    # list of accumulated rewards
T = []    # list of timesteps at which rewards were accumulated
b_X = []  # list of times
b_Y = []  # list of successes at those times
t_X = []  # list of times
t_Y = []  # list of successes at those times
task_counts = np.zeros((max_tasks,)) # list of tasks sampled

# initial representation reward values
basis_val = 0
tensor_val = 0

basis_pick_count = 0
tensor_pick_count = 0

# simulation loop
choices = []

for t in np.arange(0, num_trials):
	# pick number of tasks
	num_tasks = np.random.choice(np.arange(1, max_tasks + 1), p=true_tasks)
	task_counts[num_tasks - 1] += 1

	# pick representation according to softmax over values
	is_tensor = (pick_action(basis_val, tensor_val, params=epsilon, algorithm="e-greedy") == 1) # pick_action returns 1 for tensor
	choices.append(is_tensor)

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
	if b_X:
		basis_intercept, basis_grad = infer_ols_params(b_X, b_Y, has_intercept = False, eta = eta)
	if t_X:
		tensor_intercept, tensor_grad = infer_ols_params(t_X, t_Y, has_intercept = False, eta = eta)

	# infer task distribution based off of num_tasks
	infer_task_dist = sample_multinomial(task_counts, prior_alpha)

	# recompute expected basis discounted reward values
	if b_X:
		basis_val = float(expected_discounted_reward(basis_grad, basis_intercept, infer_V, infer_P, gamma, basis_pick_count, num_trials - tensor_pick_count, max_tasks, infer_task_dist))
	
	# recompute expected tensor reward value -- infer_P = 0 since you're parallelizing the execution
	if t_X:
		tensor_val = float(expected_discounted_reward(tensor_grad, tensor_intercept, infer_V, 0, gamma, tensor_pick_count, num_trials - basis_pick_count, max_tasks, infer_task_dist))

	# woo, logging.
	# print("Iteration %d:\n" % t)
	# print("V = %f\nP = %f\n" % (infer_V, infer_P))

	# if t_X:
	# 	print("Tensor pick count: %d\nTensor value: %f\nTensor linear params: (%f, %f)\nTruth: (%f, %f)\n" % (tensor_pick_count, tensor_val, tensor_intercept, tensor_grad, tensor_true_b0, tensor_true_b1))
	# if b_X:
	# 	print("Basis pick count: %d\nBasis value: %f\nBasis linear params: (%f, %f)\nTruth: (%f, %f)\n" % (basis_pick_count, basis_val, basis_intercept, basis_grad, basis_true_b0, basis_true_b1))

print("%d,%d" % (tensor_pick_count, basis_pick_count))
print(choices)
