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

# it's convention to provide the bayesian terms as a logarithm, apparently...
# computes the log-prior on mu_V, mu_T given optional prior parameters
def lnprior(theta, prior_params):
	v_prior_mean, v_prior_var, p_prior_mean, p_prior_var = prior_params
	v, p = theta
	vp_t1 = math.log(1 / math.sqrt(2 * math.pi * v_prior_var))
	vp_t2 = (v - v_prior_mean)**2
	vp_t2 /= (2 * v_prior_var)
	vp = vp_t1 - vp_t2

	pp_t1 = math.log(1 / math.sqrt(2 * math.pi * p_prior_var)) # if i were smart, i'd define a log-Gauss function...
	pp_t2 = (p - p_prior_mean)**2
	pp_t2 /= (2 * p_prior_var)
	pp = pp_t1 - pp_t2

	return vp + pp

# see comment on lnprior...
# computes the log-likelihood for given mu_V, mu_T given data (R, T)
def lnlike(theta, R, T):
	v, p = theta
	s = 0
	for i in range(R.shape[0]):
		like_var = 1 + (T[i, 0] ** 2) # if V ~ N(mu_V, 1), P ~ N(mu_P, 1), then V - tP is distributed
									  # according to N(mu_V - tmu_P, 1 + t^2)
		like_mean = v - (T[i, 0] * p)

		const_factor = math.log(1 / math.sqrt(2 * math.pi * like_var))
		exp_factor = (R[i, 0] - like_mean) ** 2
		exp_factor /= (2 * like_var)

		s += const_factor - exp_factor

	return s

# computes log of prior-likelihood product
def lnprob(theta, R, T, prior_params):
	lp = lnprior(theta, prior_params)
	ll = lnlike(theta, R, T)
	return lp + ll

######### PROPMT SAMPLING FUNCTIONS

# sample a multinomial task distribution for the number of tasks
def sample_multinomial(counts, prior_alpha):
	multinom = np.random.dirichlet(counts + prior_alpha)
	return multinom

######### P(SUCCESS) SAMPLING FUNCTIONS

# given a list of prior means and inverse prior covariances, return parameters for logistic regression
# on the data (X, Y)
def sample_b0b1(X, Y, prior_means, prior_Hessians):
	w_fit_MAP, H_posterior = bl.fit_bayes_logistic(Y, X, prior_means, prior_Hessians)
	return w_fit_MAP

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
	for t in np.arange(t_0, T + 1): 
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
		elif basis_val == tensor_val:
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
delta = 0                         # generalization parameter

# training parameters
basis_true_k = 0.01                                # logistic training gradient
basis_true_m = num_trials * float(sys.argv[4])     # logistic training midpoint
basis_true_b0, basis_true_b1 = km_to_b0b1(basis_true_k, basis_true_m) # translate gradient/midpoint to linear parameters

tensor_true_k = 0.01                               # logistic training gradient
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

# Reward inference parameters
v_prior_mean = true_V
v_prior_var = 0.1

p_prior_mean = true_P
p_prior_var = 0.001

vp_prior_params = (v_prior_mean, v_prior_var, p_prior_mean, p_prior_var)

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
basis_generalization_count = 0
tensor_pick_count = 2
tensor_generalization_count = 0

# simulation loop
choices = []

for t in np.arange(2, num_trials):
	# pick number of tasks
	num_tasks = np.random.choice(np.arange(1, max_tasks + 1), p=true_tasks)
	task_counts[num_tasks - 1] += 1

	# pick representation according to softmax over values
	is_tensor = (pick_action(basis_val, tensor_val, params=epsilon, algorithm="e-greedy") == 1) # pick_action returns 1 for tensor
	choices.append(is_tensor)
	# get reward for all tasks
	if is_tensor:
		trial_R, trial_T = get_reward(num_tasks, tensor_pick_count + tensor_generalization_count, is_tensor, tensor_true_params)
		
		# add data to success history
		for i in range(len(trial_T)):
			t_X.append(tensor_pick_count + tensor_generalization_count)
			t_Y.append(1)

		for i in range(num_tasks - len(trial_T)):
			t_X.append(tensor_pick_count + tensor_generalization_count)
			t_Y.append(0)

		tensor_pick_count += 1
		basis_pick_count += delta

	else:
		trial_R, trial_T = get_reward(num_tasks, basis_pick_count + basis_generalization_count, is_tensor, basis_true_params)

		# add data to success history
		for i in range(len(trial_T)):
			b_X.append(basis_pick_count + basis_generalization_count)
			b_Y.append(1)

		for i in range(num_tasks - len(trial_T)):
			b_X.append(basis_pick_count + basis_generalization_count)
			b_Y.append(0)

		basis_pick_count += 1
		tensor_pick_count += delta

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
	infer_task_dist = sample_multinomial(task_counts, prior_alpha)

	# recompute expected basis discounted reward values
	basis_val = float(expected_discounted_reward(b_reg.rx2('coefficients')[0], b_reg.rx2('coefficients')[1], infer_V, infer_P, gamma, basis_pick_count + basis_generalization_count, num_trials - tensor_pick_count + basis_generalization_count, max_tasks, infer_task_dist))
	basis_val = float(expected_discounted_reward(basis_true_b0, basis_true_b1, infer_V, infer_P, gamma, basis_pick_count + basis_generalization_count, num_trials - tensor_pick_count + basis_generalization_count, max_tasks, infer_task_dist))
	
	# recompute expected tensor reward value -- infer_P = 0 since you're parallelizing the execution
	tensor_val = float(expected_discounted_reward(t_reg.rx2('coefficients')[0], t_reg.rx2('coefficients')[1], infer_V, 0, gamma, tensor_pick_count + tensor_generalization_count, num_trials - basis_pick_count + tensor_generalization_count, max_tasks, infer_task_dist))
	tensor_val = float(expected_discounted_reward(tensor_true_b0, tensor_true_b1, infer_V, 0, gamma, tensor_pick_count + tensor_generalization_count, num_trials - basis_pick_count + tensor_generalization_count, max_tasks, infer_task_dist))

print("%d,%d" % (tensor_pick_count, basis_pick_count))
print(choices)
