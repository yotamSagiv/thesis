from sklearn.linear_model import LogisticRegression
from scipy.special import hyp2f1
from scipy.special import expit
import numpy as np
import matplotlib.pyplot as plt
import math
import bayes_logistic as bl
from random import randint

############ Helper functions. ############

# logistic form 1 / 1 + exp(k * (x - x0))
def logistic(x, x0, k):
	return expit(k * (x - x0))

# logistic form 1 / 1 + exp(b_1 * x + b_0)
def lin_logistic(x, b_1, b_0):
	return expit(b_1 * x + b_0)

# generate data
def generate_samples(num_samples, basis_params, tensor_params):
	basis_classes = []
	tensor_classes = []
	for t in range(num_samples):
		r = np.random.uniform(0, 1)
		if r < logistic(t % 40, basis_params[0], basis_params[1]): # logistic(t, basis_midpoint, basis_steepness)
			basis_classes.append(1)
		else:
			basis_classes.append(0)

		r = np.random.uniform(0, 1)
		if r < logistic(t % 40, tensor_params[0], tensor_params[1]): # logistic(t, tensor_midpoint, tensor_steepness)
			tensor_classes.append(1)
		else:
			tensor_classes.append(0)

	return (np.array(basis_classes), np.array(tensor_classes))

# approximate the integral with rectangle sums
def expected_discounted_reward(R, b_1, b_0, gamma, t_0, T, dt):
	s = 0
	for t in np.arange(t_0, T + dt, dt):
		discounted_rect_area = dt * lin_logistic(t, b_1, b_0) * (gamma ** (t - t_0))
		s += (discounted_rect_area * R)

	return s

# using softmax, pick an action
def pick_action(basis_val, tensor_val, tau=1): # softmax
	vals = np.array([basis_val, tensor_val]) # hack so i can copy paste from RL agent code
	denom = np.sum(np.exp(vals / tau))
	dist = np.exp(vals / tau) / denom

	choice = np.random.choice(2, p=dist)

	return choice

# plot data
def p(x=None, data=None):
	T = np.arange(-40, 40).reshape(-1, 1)
	try: 
		plt.scatter([x], data)
		probs = np.zeros(T.shape)
		for i in range(T.shape[0]):
			probs[i, 0] = logistic(T[i, 0], 20, 1)
		plt.plot(T, probs)
	except:
		probs = np.zeros(T.shape)
		for i in range(T.shape[0]):
			probs[i, 0] = logistic(T[i, 0], 20, 1)
		plt.plot(T, probs)

# plot inferred training curve
def p_reg(reg, x=None, data=None):
	T = np.arange(-40, 40).reshape(-1, 1)
	try: 
		plt.scatter([x], data)
		probs = reg.predict_proba(T)
		print(probs[:, 1].shape)
		plt.plot(T, probs[:, 1])
	except:
		probs = reg.predict_proba(T)
		plt.plot(T, probs[:, 1])

# assume unit variance, sample a mean from the posterior
def infer_mean(data, var, prior_mean=0, prior_var=10):
	sig_1 = 1 / ((prior_var ** -1) + (len(data) * (var ** -1)))
	s = 0
	for i in range(len(data)):
		s += data[i] * (var ** -1)

	mu_1 = sig_1 * (s + (prior_mean * (prior_var ** -1)))

	return np.random.normal(mu_1, math.sqrt(sig_1))

def km_to_b0b1(k, m):
	return (-m * k, k)

def b0b1_to_km(b0, b1):
	return (b1, -b0 / b1)
############ Main script. ############

# pick (true) training curves
basis_midpoint = 20
basis_steepness = 1
basis_max = 1

tensor_midpoint = 150
tensor_steepness = 1
tensor_max = 1

# simulation parameters
num_trials = 800
Rt = 5
Rb = 1
tau = 1
gamma = 0.95
prop_mt = 0.2
num_mt_tasks = 2

# prior data
basis_means = np.array(km_to_b0b1(basis_steepness, basis_midpoint))
tensor_means = np.array(km_to_b0b1(tensor_steepness, tensor_midpoint))

basis_hess = np.array((1, 1)) * 0.001
tensor_hess = np.diag(np.array((1, 1))) * 0.001

# simulation data 
X = np.arange(0, num_trials).reshape(-1, 1)
b_Y, t_Y = generate_samples(num_trials, [basis_midpoint, basis_steepness], [tensor_midpoint, tensor_steepness])

X = np.vstack((np.array([num_trials]), X))
b_Y = np.append(1, b_Y)
t_Y = np.append(1, t_Y)

b_reg = LogisticRegression(C=1e5)
t_reg = LogisticRegression(C=1e5)

# simulation init
b_pick_counter = 2
t_pick_counter = 2
basis_val = 0
tensor_val = 0

Rb_rewards = [Rb]
Rt_rewards = [Rt]

inferred_Rb = 0
inferred_Rt = 0

# run simulation
b_actual = 0
t_actual = 0
for i in np.arange(4, num_trials):
	a = pick_action(basis_val, tensor_val)
	print("Iteration %d: (%f, %f, %d, %f, %f)" % (i, basis_val, tensor_val, a, inferred_Rb, inferred_Rt))
	if a == 0: # basis
		b_pick_counter += 1
		b_actual += 1
		if b_Y[b_pick_counter - 1] == 1: # if we succeed
			Rb_rewards.append(Rb)

	else:
		t_pick_counter += 1
		t_actual += 1
		if t_Y[t_pick_counter - 1] == 1:
			Rt_rewards.append(Rt)
		
	inferred_Rt = max(infer_mean(Rt_rewards, var=0.5), 0)
	inferred_Rb = max(infer_mean(Rb_rewards, var=0.5), 0)

	b_reg.fit(X[:b_pick_counter, :], b_Y[:b_pick_counter])
	t_reg.fit(X[:t_pick_counter, :], t_Y[:t_pick_counter])

	basis_val = float(expected_discounted_reward(inferred_Rb, b_reg.coef_, b_reg.intercept_, gamma, b_pick_counter, num_trials - t_pick_counter, 0.01))
	tensor_val = float(expected_discounted_reward(inferred_Rt, t_reg.coef_, t_reg.intercept_, gamma, t_pick_counter, num_trials - b_pick_counter, 0.01))

print(b_actual)
print(t_actual)
