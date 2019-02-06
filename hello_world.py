import tensorflow as tf

x=tf.constant("Hello Prateek")

sess = tf.Session()

print(sess.run(x))


