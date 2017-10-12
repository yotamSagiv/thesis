import numpy as np
import random

class Network(object):

	def __init__(self, num_indim, num_infeat, num_outdim, num_outfeat):
		self.b_O = np.random.randn(num_outdim * num_outfeat, 1)
		
		self.w_B_HO = np.random.randn(num_outdim * num_outfeat, num_indim * num_infeat)
		self.w_B_TO = np.random.randn(num_outdim * num_outfeat, num_indim + num_outdim)
		
		self.w_T_HO = np.random.randn(num_outdim * num_outfeat, num_outdim * num_indim * num_infeat)
		self.w_T_TO = np.random.randn(num_outdim * num_outfeat, num_indim * num_outdim)
		
	def feedforward(self, input, task, is_basis):
		if is_basis:
			return sigmoid(np.dot(self.w_B_HO, input) + np.dot(self.w_B_TO, task) + self.b_O)
		else:
			return sigmoid(np.dot(self.w_T_HO, input) + np.dot(self.w_T_TO, task) + self.b_O)
			
	def train(self, training_data, iterations, batch_size, eta, is_basis=True, is_verbose=True):
		n = len(training_data)
		for j in range(iterations):
			random.shuffle(training_data)
			batches = [
				training_data[k:k+batch_size]
				for k in range(0, n, batch_size)
			]
			
			for batch in batches:
				self.update_batch(batch, eta, is_basis)
			
			if is_verbose:
				print(j)
	
	def update_batch(self, batch, eta, is_basis):
		dB = np.zeros(self.b_O.shape)
		
		if is_basis:
			d_B_HO = np.zeros(self.w_B_HO.shape)
			d_B_TO = np.zeros(self.w_B_TO.shape)
		else:
			d_T_HO = np.zeros(self.w_T_HO.shape)
			d_T_TO = np.zeros(self.w_T_TO.shape)
		
		for x, y, z in batch:
			db, dw_H, dw_T = self.backprop(x, y, z, is_basis)
			dB += db
			
			if is_basis:
				d_B_HO += dw_H
				d_B_TO += dw_T
			else:
				d_T_HO += dw_H
				d_T_TO += dw_T
		
		if is_basis:
			self.w_B_HO -= ((eta / len(batch)) * d_B_HO)
			self.w_B_TO -= ((eta / len(batch)) * d_B_TO)
		else:
			self.w_T_HO -= ((eta / len(batch)) * d_T_HO)
			self.w_T_TO -= ((eta / len(batch)) * d_T_TO)
			
		self.b_O -= ((eta / len(batch)) * dB)
	
	def backprop(self, input, label, task, is_basis):
		dB = np.zeros(self.b_O.shape)
		
		if is_basis:
			d_B_HO = np.zeros(self.w_B_HO.shape)
			d_B_TO = np.zeros(self.w_B_TO.shape)
		else:
			d_T_HO = np.zeros(self.w_T_HO.shape)
			d_T_TO = np.zeros(self.w_T_TO.shape)
		
		# feedforward
		if is_basis:
			z_out = np.dot(self.w_B_HO, input) + np.dot(self.w_B_TO, task) + self.b_O
		else:
			z_out = np.dot(self.w_T_HO, input) + np.dot(self.w_T_TO, task) + self.b_O
		
		a_out = sigmoid(z_out)
		
		# backpropagate error
		delta = self.cost_derivative(a_out, label) * sigmoid_prime(z_out)
		dB = delta
		
		if is_basis:
			d_B_HO = np.dot(delta, input.T)
			d_B_TO = np.dot(delta, task.T)
			return (dB, d_B_HO, d_B_TO)
			
		else:
			d_T_HO = np.dot(delta, input.T)
			d_T_TO = np.dot(delta, task.T)
			return (dB, d_T_HO, d_T_TO)
		
		
	def cost_derivative(self, output, label):
		return (output - label)
	
	def evaluate(self, test_data, is_basis=True):
		mse = 0
		np.set_printoptions(suppress=True)
		for input, label, task in test_data:
			output = self.feedforward(input, task, is_basis)
			if mse == 0:
				print(input.reshape(-1, 4))
				print(label.reshape(-1, 4))
				print(output.reshape(-1, 4))
			mse += np.sum((label - output)**2)
			
		mse /= len(test_data)
		mse /= test_data[0][1].shape[0]
		
		return mse		
		
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))

def generate_basis_stimuli(num_stimuli, num_indim, num_infeat, num_outdim, num_outfeat):
	stimuli = []
	for t in range(num_stimuli):
		stim = np.zeros((num_indim * num_infeat, 1))
		label = np.zeros((num_outdim * num_outfeat, 1))
		task = np.zeros((num_outdim + num_indim, 1))
		
		indim = np.random.choice(num_indim)
		outdim = np.random.choice(num_outdim)
		
		task[indim, 0] = 1
		task[num_indim + outdim, 0] = 1
		
		infeat = np.random.choice(num_infeat)
		stim[indim * num_indim + infeat, 0] = 1
		label[outdim * num_outdim + infeat, 0] = 1
		
		stimuli.append((stim, label, task))
	
	return stimuli

def generate_tensor_stimuli(num_stimuli, num_indim, num_infeat, num_outdim, num_outfeat):
	stimuli = []
	for t in range(num_stimuli):
		stim = np.zeros((num_outdim * num_indim * num_infeat, 1))
		label = np.zeros((num_outdim * num_outfeat, 1))
		task = np.zeros((num_outdim * num_indim, 1))
		
		indim = np.random.choice(num_indim)
		outdim = np.random.choice(num_outdim)
		
		task[indim * num_indim + outdim, 0] = 1
		
		infeat = np.random.choice(num_infeat)
		stim[indim * num_indim + outdim + infeat, 0] = 1
		
		label[outdim * num_outdim + infeat, 0] = 1
		
		stimuli.append((stim, label, task))
	
	return stimuli
		

####

num_indim = 4
num_infeat = 4
num_outdim = 4
num_outfeat = 4
num_iterations = 50000
num_training_stimuli = 200
num_test_stimuli = 100
eta = 0.1
batch_size = 20
iv = False

n = Network(num_indim, num_infeat, num_outdim, num_outfeat)

# # tensor
# data = generate_tensor_stimuli(num_training_stimuli, num_indim, num_infeat, num_outdim, num_outfeat)
# n.train(data, num_iterations, batch_size, eta, False, is_verbose=True)
# 
# test_tensor_data = generate_tensor_stimuli(num_test_stimuli, num_indim, num_infeat, num_outdim, num_outfeat)
# print(n.evaluate(test_tensor_data, False))

# basis
data = generate_basis_stimuli(num_training_stimuli, num_indim, num_infeat, num_outdim, num_outfeat)
n.train(data, num_iterations, batch_size, eta, is_verbose=True)

test_basis_data = generate_basis_stimuli(num_test_stimuli, num_indim, num_infeat, num_outdim, num_outfeat)
print(n.b_O)
print(n.w_B_HO)
print(n.w_B_TO)
# print(n.evaluate(test_basis_data))
