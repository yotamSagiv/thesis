import tensorflow as tf
import numpy as np
import sys

def basis_net(num_indim, num_infeat, num_outdim, num_outfeat):
	# input layers
	with tf.name_scope('basis_input_layer'):
		b_i = tf.placeholder(tf.float32, [None, num_indim * num_infeat]) 
	with tf.name_scope('basis_task_layer'):
		b_t = tf.placeholder(tf.float32, [None, num_indim + num_outdim]) 

	# weights
	with tf.name_scope('basis_IH_weights'):
		w_B_IH = tf.Variable(tf.zeros([num_indim * num_infeat, num_indim * num_infeat]))
	with tf.name_scope('basis_HO_weights'):
		seed_inf = np.zeros([num_infeat, num_outfeat], dtype=np.float32)
		np.fill_diagonal(seed_inf, 1)
		seed_sup = seed_inf.copy()
		for i in range(num_indim - 1):
			seed_sup = np.vstack((seed_sup, seed_inf))

		w = seed_sup.copy()
		for i in range(num_outdim - 1):
			w = np.hstack((w, seed_sup))

		w_B_HO = tf.Variable(w)
	with tf.name_scope('basis_TH_weights'):
		w = np.zeros([num_indim + num_outdim, num_indim * num_infeat], dtype=np.float32)

		feat_counter = 0
		j = 0
		for i in range(num_indim * num_infeat):
			w[j, i] = 1
			feat_counter += 1
			if feat_counter == num_infeat:
				j += 1
				feat_counter = 0

		w_B_TH = tf.Variable(w)
	with tf.name_scope('basis_TO_weights'):
		w = np.zeros([num_indim + num_outdim, num_outdim * num_outfeat], dtype=np.float32)

		feat_counter = 0
		j = num_indim
		for i in range(num_indim * num_infeat):
			w[j, i] = 1
			feat_counter += 1
			if feat_counter == num_infeat:
				j += 1
				feat_counter = 0

		w_B_TO = tf.Variable(w)

	# bias
	with tf.name_scope('basis_hidden_bias'):
		b_h_b = tf.Variable(tf.zeros([num_indim * num_infeat]))
	with tf.name_scope('basis_output_bias'):
		b_o_b = tf.Variable(tf.zeros([num_outdim * num_outfeat])) # lol bob

	# hidden
	with tf.name_scope('basis_hidden'):
		b_h = tf.nn.sigmoid(tf.matmul(b_i, w_B_IH) + tf.matmul(b_t, w_B_TH) + b_h_b)

	# output
	with tf.name_scope('basis_output'):
		y = tf.nn.sigmoid(tf.matmul(b_h, w_B_HO) + tf.matmul(b_t, w_B_TO) + b_o_b)

	# label
	with tf.name_scope('basis_label'):
		y_ = tf.placeholder(tf.float32, [None, num_indim * num_infeat])

	return b_i, b_t, b_h, w_B_IH, w_B_TH, w_B_HO, w_B_TO, b_h_b, b_o_b, y, y_

def tensor_net(num_indim, num_infeat, num_outdim, num_outfeat):
	# input layers
	with tf.name_scope('tensor_input_layer'):
		t_i = tf.placeholder(tf.float32, [None, num_indim * num_infeat])
	with tf.name_scope('tensor_task_layer'):
		t_t = tf.placeholder(tf.float32, [None, num_indim * num_outdim]) 

	# weights
	with tf.name_scope('tensor_IH_weights'):
		w_T_IH = tf.Variable(tf.zeros([num_indim * num_infeat, num_indim * num_outdim * num_infeat]))
	with tf.name_scope('tensor_TH_weights'):
		w = np.zeros([num_indim * num_outdim, num_indim * num_outdim * num_infeat], dtype=np.float32)

		j = 0
		mod = 0
		for i in range(num_indim * num_outdim * num_infeat):
			w[j, i] = 1
			mod += 1
			if mod == num_infeat:
				j += 1
				mod = 0

		w_T_TH = tf.Variable(w)
	with tf.name_scope('tensor_HO_weights'):
		seed = np.zeros((num_outdim * num_infeat, num_outdim * num_outfeat), dtype=np.float32)
		np.fill_diagonal(seed, 1)
		w = seed.copy()
		for i in range(num_indim - 1):
			w = np.vstack((w, seed))

		w_T_HO = tf.Variable(w)
	with tf.name_scope('tensor_TO_weights'):
		w_T_TO = tf.Variable(tf.zeros([num_indim * num_outdim, num_outdim * num_outfeat]))

	# biases
	with tf.name_scope('tensor_hidden_bias'):
		t_h_b = tf.Variable(tf.zeros([num_indim * num_outdim * num_infeat]))

	with tf.name_scope('tensor_output_bias'):
		t_o_b = tf.Variable(tf.zeros([num_outdim * num_outfeat]))

	# hidden activation
	with tf.name_scope('tensor_hidden'):
		t_h = tf.nn.sigmoid(tf.matmul(t_i, w_T_IH) + tf.matmul(t_t, w_T_TH) + t_h_b)

	# output
	with tf.name_scope('tensor_output'):
		y = tf.nn.sigmoid(tf.matmul(t_h, w_T_HO) + tf.matmul(t_t, w_T_TO) + t_o_b)

	# label
	with tf.name_scope('tensor_label'):
		y_ = tf.placeholder(tf.float32, [None, num_outdim * num_outfeat])

	return t_i, t_t, t_h, w_T_IH, w_T_TH, w_T_HO, w_T_TO, t_h_b, t_o_b, y, y_

def loss(y, y_):
	return tf.reduce_mean(tf.squared_difference(y, y_))

def train(loss):
	global_step = tf.Variable(0, name='global_step', trainable=False)
	return tf.train.GradientDescentOptimizer(0.3).minimize(loss, global_step=global_step)


######## script



# num_indim = 3
# num_infeat = 3
# num_outdim = 3
# num_outfeat = 3
# is_basis = (sys.argv[1] == "basis")

# if is_basis: 
# 	# layers
# 	b_i = tf.placeholder(tf.float32, [None, num_indim * num_infeat]) 
# 	b_t = tf.placeholder(tf.float32, [None, num_indim + num_outdim]) 

# 	# weights
# 	w_B_IH = tf.Variable(tf.zeros([num_indim * num_infeat, num_indim * num_infeat]))
# 	w_B_TH = tf.Variable(tf.zeros([num_indim + num_outdim, num_indim * num_infeat]))

# 	# bias
# 	b_b = tf.Variable(tf.zeros([num_indim * num_infeat]))

# 	# output
# 	y = tf.nn.sigmoid(tf.matmul(b_i, w_B_IH) + tf.matmul(b_t, w_B_TH) + b_b)

# 	# label
# 	y_ = tf.placeholder(tf.float32, [None, num_indim * num_infeat])

# else: 
# 	# layers
# 	t_i = tf.placeholder(tf.float32, [None, num_indim * num_infeat])
# 	t_t = tf.placeholder(tf.float32, [None, num_indim * num_outdim]) 

# 	# weights
# 	w_T_IH = tf.Variable(tf.zeros([num_indim * num_infeat, num_indim * num_outdim * num_infeat]))
# 	w_T_TH = tf.Variable(tf.zeros([num_indim * num_outdim, num_indim * num_outdim * num_infeat]))

# 	# bias
# 	t_b = tf.Variable(tf.zeros([num_indim * num_outdim * num_infeat]))

# 	# output
# 	y = tf.nn.sigmoid(tf.matmul(t_i, w_T_IH) + tf.matmul(t_t, w_T_TH) + t_b)

# 	# label
# 	y_ = tf.placeholder(tf.float32, [None, num_indim * num_outdim * num_infeat])


# # use quadratic loss
# loss = tf.reduce_mean(tf.squared_difference(y, y_))

# # set up summarizer, global step
# tf.summary.scalar('loss', loss)
# global_step = tf.Variable(0, name='global_step', trainable=False)

# # training op
# train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss, global_step=global_step)

# # accuracy measure
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# # build summary Tensor
# summary = tf.summary.merge_all()

# # TensorFlow setup/run
# sess = tf.InteractiveSession()

# # instantiate summary writer
# summary_writer = tf.summary.FileWriter('./tfnetlogs/honet', sess.graph)

# # init vars
# tf.global_variables_initializer().run()


# if is_basis:
# 	print('basis data')
# else:
# 	print('tensor data')

# # training loop
# for t in range(10000):
# 	if is_basis:
# 		batch_is, batch_ts, batch_ys = generate_basis_stimuli(300, num_indim, num_infeat, num_outdim, num_outfeat)
# 		feed_dict= {
# 			b_i: batch_is, 
# 			b_t: batch_ts, 
# 			y_: batch_ys
# 		}
# 		_, l = sess.run([train_step, loss], feed_dict=feed_dict)
		
# 		if t % 100 == 0:
# 			print('step %d: loss = %.8f' % (t, l))
# 			summary_str = sess.run(summary, feed_dict=feed_dict)
# 			summary_writer.add_summary(summary_str, t)
# 			summary_writer.flush()
# 	else:
# 		batch_is, batch_ts, batch_ys = generate_tensor_stimuli(300, num_indim, num_infeat, num_outdim, num_outfeat)
# 		feed_dict= {
# 			t_i: batch_is, 
# 			t_t: batch_ts, 
# 			y_: batch_ys
# 		}
# 		_, l = sess.run([train_step, loss], feed_dict=feed_dict)
		
# 		if t % 100 == 0:
# 			print('step %d: loss = %.8f' % (t, l))
# 			summary_str = sess.run(summary, feed_dict=feed_dict)
# 			summary_writer.add_summary(summary_str, t)
# 			summary_writer.flush()

# if is_basis:
# 	test_i, test_t, test_y = generate_basis_stimuli(100, num_indim, num_infeat, num_outdim, num_outfeat)
# 	print('test accuracy: %.2f\n' % sess.run(accuracy, feed_dict={b_i:test_i, b_t:test_t, y_:test_y}))
# else:
# 	test_i, test_t, test_y = generate_tensor_stimuli(100, num_indim, num_infeat, num_outdim, num_outfeat)
# 	print('test accuracy: %.2f\n' % sess.run(accuracy, feed_dict={t_i:test_i, t_t:test_t, y_:test_y}))



















