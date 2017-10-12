import numpy as np

class NeuralNetwork:
	def __init__(self, num_out_dim, num_out_feat, num_in_dim, num_in_feat):
		num_out_nodes = num_out_dim * num_out_feat
		# i think that output bias doesn't matter quite as much, but i'll make it anyway
		self.b_O = np.zeros((num_out_nodes, 1)) - 1

		# basis set
		b_in_nodes = num_in_dim * num_in_feat
		self.w_B_IO = np.random.uniform(-0.3, 0.3, (num_out_nodes, b_in_nodes)) # input

		# tensor 
		t_in_nodes = num_in_dim * num_out_dim * num_in_feat
		self.w_T_IO = np.random.uniform(-0.3, 0.3, (num_out_nodes, t_in_nodes)) # input

	def run_basis(self, inp):
		out_act = self.sigmoid(np.dot(self.w_B_IO, inp) - self.b_O)
		return out_act

	def run_tensor(self, inp):
		out_act = self.sigmoid((self.w_T_IO * inp) - self.b_O)
		return out_act

	def train_basis(self, training_set, label_set, num_iterations, learning_rate):
		for t in range(num_iterations):
			print(t)
			for i in range(training_set.shape[0]):
				# feedforward
				inp = training_set[i, :].reshape(-1, 1)
				out = self.run_basis(inp)

				# calculate output delta
				label = label_set[i, :].reshape(-1, 1)
				d_out = self.cost_derivative(out, label) * self.sigmoid_prime(np.dot(self.w_B_IO, inp) - self.b_O)

				self.w_B_IO -= (learning_rate * d_out * inp.T)
				self.b_O -= (learning_rate * d_out)

	def train_tensor(self, training_set, label_set, num_iterations, learning_rate):
		for t in range(num_iterations):
			print(t)
			for i in range(training_set.shape[0]):
				# feedforward
				inp = training_set[i, :].reshape(-1, 1)
				out = self.run_tensor(inp)

				# calculate output delta
				label = label_set[i, :].reshape(-1, 1)
				d_out = self.cost_derivative(out, label) * self.sigmoid_prime(np.dot(self.w_T_IO, inp) - self.b_O)


				self.w_T_IO -= (learning_rate * d_out * inp.T)


	def cost_derivative(self, output, label):
		return output - label
		
	def sigmoid(self, x):
		return 1.0 / (1.0 + np.exp(-x))

	def sigmoid_prime(self, x):
		return self.sigmoid(x) * (1 - self.sigmoid(x))

##### Helper functions

def generate_basis_stimuli(num_out_dim, num_out_feat, num_in_dim, num_in_feat, num_stimuli):
	stimuli = np.zeros((num_stimuli, num_in_dim * num_in_feat))

	for i in range(num_stimuli):
		# stimulus
		in_dim = np.random.choice(num_in_dim)
		in_feat = np.random.choice(num_in_feat)

		stimuli[i, in_dim * num_in_dim + in_feat] = 1

	return (stimuli, stimuli.copy())

def cost(output, label):
	return np.sum((output - label)**2) / output.shape[0]

nn = NeuralNetwork(4, 4, 4, 4)
num_training_stimuli = 100
num_testing_stimuli = 50
num_iterations = 10000


# train basis set
(train_stimuli, train_labels) = generate_basis_stimuli(4, 4, 4, 4, num_training_stimuli)
nn.train_basis(train_stimuli, train_labels, num_iterations, 0.2)


# test basis set
total_cost = 0
(test_stimuli, test_labels) = generate_basis_stimuli(4, 4, 4, 4, num_testing_stimuli)
for i in range(50):
	out = nn.run_basis(test_stimuli[i, :].reshape(-1, 1))
	total_cost += cost(out, test_labels[i, :].reshape(-1, 1))

print(total_cost / 50)











