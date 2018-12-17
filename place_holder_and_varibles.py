import tensorflow as tf

Input1 = tf.placeholder('float',shape=[None,3], name="Input_1")
Input2 = tf.placeholder('float',shape=[None,3], name="Input_2")

x=tf.Variable(0,dtype="float")
output = tf.Variable(0,dtype="float")

x=Input1 * Input2

output = x

sess=tf.Session()

#print(sess.run(output,feed_dict={Input1:[[1,2,3],Input2:[[4,3,2]]]}))
print(sess.run(output,feed_dict={Input1:[[1,2,3]],Input2:[[4,3,2]]}))