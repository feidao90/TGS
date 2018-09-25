import tensorflow as tf

sess = tf.Session()

a = tf.Variable(tf.constant(4.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)
mul = tf.mul(a,x_data)
loss = tf.square(tf.sub(mul,50))
init = tf.initialize_all_variables()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(0.01)
my_opt2 = tf.train.GradientDescentOptimizer(0.1)
train_step = my_opt.minimize(loss)
train_step2 = my_opt2.minimize(loss)

for i in range(10):
    sess.run(train_step,feed_dict={x_data:x_val})
    sess.run(train_step2,feed_dict={x_data:x_val})
    a_val = sess.run(a)
    mul_output = sess.run(mul,feed_dict={x_data:x_val})
    print(str(a_val)+'*'+str(x_val)+'='+str(mul_output))
