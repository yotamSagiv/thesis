import numpy as np
import tensorflow as tf
import tfihnet
import sys

# agent class
class RL_agent: 

	# init with number of bandits
	def __init__(self, num_bandits):
		self.num_bandits = num_bandits
		self.vals = np.zeros((1, num_bandits))

	def pick_action(self, tau): # softmax
		denom = np.sum(np.exp(self.vals / tau))
		dist = np.exp(self.vals / tau) / denom

		choice = np.random.choice(self.num_bandits, p=dist[0, :])

		return choice

	def update_value(self, action, reward, learning_rate):
		pe = reward - self.vals[0, action]
		self.vals[0, action] += (learning_rate * pe)

###### helper functions

# reward tensor
def reward(action, label, verbose=False):
	k = np.count_nonzero(label)
	a = action.argsort()[0, -k:]
	l = label.argsort()[0, -k:]
	if np.array_equal(np.sort(a), np.sort(l)):
		return 1
	else:
		
		if verbose:
			print(k)
			print(action)
			print(a)
			print(label)
			print(l)

		return 0

# generate stimuli with basis label
def generate_basis_stimuli(num_stimuli, num_indim, num_infeat, num_outdim, num_outfeat):
	stims = []
	tasks = []
	labels = []
	for t in range(num_stimuli):
		stim = np.zeros((num_indim * num_infeat, 1))
		label = np.zeros((num_indim * num_infeat, 1))
		task = np.zeros((num_outdim + num_indim, 1))
		
		indim = np.random.choice(num_indim)
		outdim = np.random.choice(num_outdim)
		
		task[indim, 0] = 1
		task[num_indim + outdim, 0] = 1
		
		# pick a feature for each dimension
		for i in range(num_indim):
			infeat = np.random.choice(num_infeat)
			if i == indim:
				task_infeat = infeat

			stim[i * num_indim + infeat, 0] = 1

		label[num_indim * indim + task_infeat, 0] = 1
		
		stims.append(stim.T)
		tasks.append(task.T)
		labels.append(label.T)
	
	return np.vstack(stims), np.vstack(tasks), np.vstack(labels)

# generate stimuli with tensor label
def generate_tensor_stimuli(num_stimuli, num_indim, num_infeat, num_outdim, num_outfeat):
	stims = []
	tasks = []
	labels = []
	for t in range(num_stimuli):
		stim = np.zeros((num_indim * num_infeat, 1))
		label = np.zeros((num_outdim * num_indim * num_infeat, 1))
		task = np.zeros((num_outdim * num_indim, 1))
		
		indim = np.random.choice(num_indim)
		outdim = np.random.choice(num_outdim)
		
		task_index = num_indim * indim + outdim
		task[task_index, 0] = 1
		
		# pick a feature for each dimension
		for i in range(num_indim):
			infeat = np.random.choice(num_infeat)
			if i == indim:
				task_infeat = infeat

			stim[i * num_indim + infeat, 0] = 1


		label[task_index * num_infeat + task_infeat, 0] = 1
		
		stims.append(stim.T)
		tasks.append(task.T)
		labels.append(label.T)
	
	return np.vstack(stims), np.vstack(tasks), np.vstack(labels)

# generate multitasking stimuli with basis label
def generate_basis_mt_stimuli(num_stimuli, prop_mt, num_indim, num_infeat, num_outdim, num_outfeat):
	stims = []
	tasks = []
	labels = []
	for t in range(num_stimuli):
		stim = np.zeros((num_indim * num_infeat, 1))
		label = np.zeros((num_indim * num_infeat, 1))
		task = np.zeros((num_outdim + num_indim, 1))
		
		mt = (np.random.uniform(0, 1) < prop_mt)

		if mt:
			indims = np.random.choice(num_indim, size=(2, ), replace=False)
			outdims = np.random.choice(num_outdim, size=(2, ), replace=False)

			task[indims[0], 0] = 1
			task[indims[1], 0] = 1
			task[num_indim + outdims[0], 0] = 1
			task[num_indim + outdims[1], 0] = 1

		else:
			indim = np.random.choice(num_indim)
			outdim = np.random.choice(num_outdim)
			task[indim, 0] = 1
			task[num_indim + outdim, 0] = 1
		
		# pick a feature for each dimension
		for i in range(num_indim):
			infeat = np.random.choice(num_infeat)
			if mt and indims[0] == i:
				task_infeat1 = infeat
				label[num_indim * indims[0] + task_infeat1, 0] = 1
			if mt and indims[1] == i:
				task_infeat2 = infeat
				label[num_indim * indims[1] + task_infeat2, 0] = 1
			if not mt and i == indim:
				task_infeat = infeat
				label[num_indim * indim + task_infeat, 0] = 1

			stim[i * num_indim + infeat, 0] = 1

		stims.append(stim.T)
		tasks.append(task.T)
		labels.append(label.T)
	
	return np.vstack(stims), np.vstack(tasks), np.vstack(labels)

# generate multitasking stimuli with tensor label
def generate_mt_stimuli(num_stimuli, is_tensor, prop_mt, num_indim, num_infeat, num_outdim, num_outfeat):
	stims = []
	tasks = []
	labels = []
	for t in range(num_stimuli):
		stim = np.zeros((num_indim * num_infeat, 1))
		label = np.zeros((num_outdim * num_outfeat, 1))
		if is_tensor:
			task = np.zeros((num_outdim * num_indim, 1))
		else:
			task = np.zeros((num_outdim + num_indim, 1))
		
		mt = (np.random.uniform(0, 1) < prop_mt)

		if mt:
			indims = np.random.choice(num_indim, size=(2, ), replace=False)
			outdims = np.random.choice(num_outdim, size=(2, ), replace=False)

			if is_tensor: # tensor task label
				task_index1 = num_indim * indims[0] + outdims[0]
				task_index2 = num_indim * indims[1] + outdims[1]
				task[task_index1, 0] = 1
				task[task_index2, 0] = 1
			else: # basis task label
				task[indims[0], 0] = 1
				task[indims[1], 0] = 1
				task[num_indim + outdims[0], 0] = 1
				task[num_indim + outdims[1], 0] = 1

		else:
			indim = np.random.choice(num_indim)
			outdim = np.random.choice(num_outdim)

			if is_tensor:
				task_index = num_indim * indim + outdim
				task[task_index, 0] = 1
			else:
				task[indim, 0] = 1
				task[num_indim + outdim] = 1
		
		# pick a feature for each dimension
		for i in range(num_indim):
			infeat = np.random.choice(num_infeat)
			if mt and indims[0] == i:
				task_infeat1 = infeat
				label[outdims[0] * num_outfeat + task_infeat1, 0] = 1
			if mt and indims[1] == i:
				task_infeat2 = infeat
				label[outdims[1] * num_outfeat + task_infeat2, 0] = 1
			if not mt and i == indim:
				task_infeat = infeat
				label[outdim * num_outfeat + task_infeat, 0] = 1

			stim[i * num_indim + infeat, 0] = 1

		stims.append(stim.T)
		tasks.append(task.T)
		labels.append(label.T)
	
	return np.vstack(stims), np.vstack(tasks), np.vstack(labels)

# task parameters
num_indim = int(sys.argv[1])
num_infeat = int(sys.argv[1])
num_outdim = int(sys.argv[1])
num_outfeat = int(sys.argv[1])
prop_mt = float(sys.argv[2])

# RL parameters
SOFTMAX_TAU = 0.25
RL_LEARNING_RATE = 0.01
num_trials = 100000

# NN parameters
NET_LEARNING_RATE = 0.05
num_subtrain_trials = 10
num_train_examples = 20

# create our agent
agent = RL_agent(2)

# choice tracking
basis_counter = 0
tensor_counter = 0

# correctness tracking
basis_correct = 0
tensor_correct = 0

# main agent training loop
with tf.Graph().as_default():
	with tf.name_scope('basis_net'):
		b_i, b_t, b_h, w_B_IH, w_B_TH, w_B_HO, w_B_TO, b_h_b, b_o_b, y_basis, y_b_ = tfihnet.basis_net(num_indim, num_infeat, num_outdim, num_outfeat)
	with tf.name_scope('tensor_net'):
		t_i, t_t, t_h, w_T_IH, w_T_TH, w_T_HO, w_T_TO, t_h_b, t_o_b, y_tensor, y_t_ = tfihnet.tensor_net(num_indim, num_infeat, num_outdim, num_outfeat)
	
	with tf.name_scope('basis_loss'):
		basis_loss = tfihnet.loss(y_basis, y_b_)
		b_l_s = tf.summary.scalar('basis_loss', basis_loss)
	with tf.name_scope('tensor_loss'):
		tensor_loss = tfihnet.loss(y_tensor, y_t_)
		t_l_s = tf.summary.scalar('tensor_loss', tensor_loss)

	with tf.name_scope('basis_trainer'):
		train_basis = tfihnet.train(basis_loss)
	with tf.name_scope('tensor_trainer'):
		train_tensor = tfihnet.train(tensor_loss)

	init = tf.global_variables_initializer()
	sess = tf.Session()

	writer = tf.summary.FileWriter('./logs/', graph=sess.graph)

	sess.run(init)
	for i in range(num_trials):
		print("step %d: [%s]" % (i, np.array_str(agent.vals, precision=3)))

		action = agent.pick_action(SOFTMAX_TAU)
		# action = 0
		# action = 1
		if action == 0: # minimal basis set
			basis_counter += 1
			for t in range(num_subtrain_trials):
				batch_is, batch_ts, batch_ys = generate_mt_stimuli(num_train_examples, False, prop_mt, num_indim, num_infeat, num_outdim, num_outfeat)
				feed_dict = {
					b_i: batch_is, 
					b_t: batch_ts, 
					y_b_: batch_ys
				}
				_, l, summary = sess.run([train_basis, basis_loss, b_l_s], feed_dict=feed_dict)

				writer.add_summary(summary, i)

			test_i, test_t, test_label = generate_mt_stimuli(1, False, prop_mt, num_indim, num_infeat, num_outdim, num_outfeat)
			feed_dict = {
				b_i: test_i, 
				b_t: test_t, 
				y_b_: test_label
			}
			output = sess.run(y_basis, feed_dict=feed_dict)
			r = reward(output, test_label)
			basis_correct += r
		else: # tensor product
			tensor_counter += 1
			for t in range(num_subtrain_trials):
				batch_is, batch_ts, batch_ys = generate_mt_stimuli(num_train_examples, True, prop_mt, num_indim, num_infeat, num_outdim, num_outfeat)
				feed_dict= {
					t_i: batch_is, 
					t_t: batch_ts, 
					y_t_: batch_ys
				}
				_, l, summary = sess.run([train_tensor, tensor_loss, t_l_s], feed_dict=feed_dict)

				writer.add_summary(summary, i)

			test_i, test_t, test_label = generate_mt_stimuli(1, True, prop_mt, num_indim, num_infeat, num_outdim, num_outfeat)
			feed_dict = {
				t_i: test_i, 
				t_t: test_t, 
				y_t_: test_label
			}
			output = sess.run(y_tensor, feed_dict=feed_dict)
			r = reward(output, test_label)
			tensor_correct += r

		agent.update_value(action, r, RL_LEARNING_RATE)

	print(agent.vals)
	print("basis choice proportion: %f" % (basis_counter / num_trials))
	print("tensor choice proportion: %f" % (tensor_counter / num_trials))
	print("basis reward proportion: %f" % (basis_correct / basis_counter))
	print("tensor reward proportion: %f" % (tensor_correct / tensor_counter))

	


