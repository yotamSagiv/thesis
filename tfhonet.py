import tensorflow as tf
import numpy as np

def generate_basis_stimuli(num_stimuli, num_indim, num_infeat, num_outdim, num_outfeat):
	stims = []
	tasks = []
	labels = []
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
		
		stims.append(stim.T)
		tasks.append(task.T)
		labels.append(label.T)
	
	return np.vstack(stims), np.vstack(tasks), np.vstack(labels)

def generate_tensor_stimuli(num_stimuli, num_indim, num_infeat, num_outdim, num_outfeat):
	stims = []
	tasks = []
	labels = []
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
		
		stims.append(stim.T)
		tasks.append(task.T)
		labels.append(label.T)
	
	return np.vstack(stims), np.vstack(tasks), np.vstack(labels)

num_indim = 3
num_infeat = 3
num_outdim = 3
num_outfeat = 3

# basis layers
b_h = tf.placeholder(tf.float32, [None, num_indim * num_infeat]) 
b_t = tf.placeholder(tf.float32, [None, num_indim + num_outdim]) 

# tensor layers
t_h = tf.placeholder(tf.float32, [None, num_outdim * num_indim * num_infeat]) 
t_t = tf.placeholder(tf.float32, [None, num_indim * num_outdim]) 

# basis weights
w_B_HO = tf.Variable(tf.zeros([num_indim * num_infeat, num_outdim * num_outfeat]))
w_B_TO = tf.Variable(tf.zeros([num_indim + num_outdim, num_outdim * num_outfeat]))

# tensor weights
w_T_HO = tf.Variable(tf.zeros([num_outdim * num_indim * num_infeat, num_outdim * num_outfeat]))
w_T_TO = tf.Variable(tf.zeros([num_indim * num_outdim, num_outdim * num_outfeat]))

# bias
b = tf.Variable(tf.zeros([num_outfeat * num_outdim]))

# output
y = tf.nn.sigmoid(tf.matmul(b_h, w_B_HO) + tf.matmul(b_t, w_B_TO) + tf.matmul(t_h, w_T_HO) + tf.matmul(t_t, w_T_TO) + b)

# label
y_ = tf.placeholder(tf.float32, [None, num_outdim * num_outfeat])

# use quadratic loss
loss = tf.reduce_mean(tf.squared_difference(y, y_))

# set up summarizer, global step
tf.summary.scalar('loss', loss)
global_step = tf.Variable(0, name='global_step', trainable=False)

# training op
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss, global_step=global_step)

# accuracy measure
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# build summary Tensor
summary = tf.summary.merge_all()

# TensorFlow setup/run
sess = tf.InteractiveSession()

# instantiate summary writer
summary_writer = tf.summary.FileWriter('./tfnetlogs/honet', sess.graph)

# init vars
tf.global_variables_initializer().run()

# basis training loop
print('basis data')
for t in range(10000):
	batch_hs, batch_ts, batch_ys = generate_basis_stimuli(300, num_indim, num_infeat, num_outdim, num_outfeat)
	null_hs, null_ts, null_ys = generate_tensor_stimuli(300, num_indim, num_infeat, num_outdim, num_outfeat)
	null_hs = np.zeros(null_hs.shape)
	null_ts = np.zeros(null_ts.shape)
	feed_dict= {
		b_h: batch_hs, 
		b_t: batch_ts, 
		t_h: null_hs, 
		t_t: null_ts, 
		y_: batch_ys
	}
	_, l = sess.run([train_step, loss], feed_dict=feed_dict)
	
	if t % 100 == 0:
		print('step %d: loss = %.8f' % (t, l))
		summary_str = sess.run(summary, feed_dict=feed_dict)
		summary_writer.add_summary(summary_str, t)
		summary_writer.flush()


test_h, test_t, test_y = generate_basis_stimuli(100, num_indim, num_infeat, num_outdim, num_outfeat)
null_h, null_t, null_y = generate_tensor_stimuli(100, num_indim, num_infeat, num_outdim, num_outfeat)
null_h = np.zeros(null_h.shape)
null_t = np.zeros(null_t.shape)
print('test accuracy: %.2f\n' % sess.run(accuracy, feed_dict={b_h:test_h, b_t:test_t, t_h:null_h, t_t:null_t, y_:test_y}))

# tensor training loop
print('tensor data')
for t in range(10000):
	batch_hs, batch_ts, batch_ys = generate_tensor_stimuli(300, num_indim, num_infeat, num_outdim, num_outfeat)
	null_hs, null_ts, null_ys = generate_basis_stimuli(300, num_indim, num_infeat, num_outdim, num_outfeat)
	null_hs = np.zeros(null_hs.shape)
	null_ts = np.zeros(null_ts.shape)
	feed_dict={
		t_h: batch_hs, 
		t_t: batch_ts, 
		b_h: null_hs, 
		b_t: null_ts, 
		y_: batch_ys
	}
	_, l = sess.run([train_step, loss], feed_dict=feed_dict)
	if t % 100 == 0:
		print('step %d: loss = %.8f' % (t, l))
		summary_str = sess.run(summary, feed_dict=feed_dict)
		summary_writer.add_summary(summary_str, t)
		summary_writer.flush()

test_h, test_t, test_y = generate_tensor_stimuli(100, num_indim, num_infeat, num_outdim, num_outfeat)
null_h, null_t, null_y = generate_basis_stimuli(100, num_indim, num_infeat, num_outdim, num_outfeat)
null_h = np.zeros(null_h.shape)
null_t = np.zeros(null_t.shape)
print('test accuracy: %.2f' % sess.run(accuracy, feed_dict={t_h:test_h, t_t:test_t, b_h:null_h, b_t:null_t, y_:test_y}))
print(sess.run(b))

batch_hs, batch_ts, batch_ys = generate_basis_stimuli(1, num_indim, num_infeat, num_outdim, num_outfeat)
null_hs, null_ts, null_ys = generate_tensor_stimuli(1, num_indim, num_infeat, num_outdim, num_outfeat)
null_hs = np.zeros(null_hs.shape)
null_ts = np.zeros(null_ts.shape)
feed_dict= {
	b_h: batch_hs, 
	b_t: batch_ts, 
	t_h: null_hs, 
	t_t: null_ts, 
	y_: batch_ys
}
print(sess.run(y, feed_dict=feed_dict).shape)
print(batch_hs)
print(batch_ts)
print(batch_ys)
print(sess.run(y, feed_dict=feed_dict))



















