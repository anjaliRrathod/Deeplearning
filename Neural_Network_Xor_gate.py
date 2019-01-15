#This will help you to create Neural Network for Xor gate


import tensorflow as tf

Input = tf.placeholder('float', shape=[None, 2], name="Input")
Target =tf.placeholder('float', shape=[None, 1], name="Target")
inputBias = tf.Variable(initial_value=tf.random_normal(shape=[3], stddev=0.4),dtype='float', name="input_bias")

weights = tf.Variable(initial_value=tf.random_normal(shape=[2, 3], stddev=0.4),dtype='float', name="hidden_weights")
hiddenBias = tf.Variable(initial_value=tf.random_normal(shape=[1], stddev=0.4),dtype='float', name="hidden_bias")

outputweights = tf.Variable(initial_value=tf.random_normal(shape=[3, 1], stddev=0.4),dtype='float', name="output_weights")

hiddenlayer = tf.matmul(Input, weights)+inputBias
hiddenlayer = tf.sigmoid(hiddenlayer, name="Hidden_layer_activation")

output = tf.matmul(hiddenlayer, outputweights) + hiddenBias
output = tf.sigmoid(output, name="output_layer_activation")

cost=tf.squared_difference(Target,output)
cost=tf.reduce_mean(cost)

optimizer=tf.train.AdamOptimizer().minimize(cost)

#Exor input and output

inp=[[1,1],[1,0],[0,1],[0,0]]
out=[[0],[1],[1],[0]]
epochs=7000


with tf.Session() as sess:

    tf.global_variables_initializer().run()
    for i in range(epochs) :
        err, _ =sess.run([cost,optimizer],feed_dict={Input:inp,Target:out})
        print(i,err)

    while  True :
        inp=[[0,0]]
        inp[0][0]=input(" first input")
        inp[0][1]=input("Second input")
        print(sess.run([output],feed_dict={Input:inp}))


#---------------------------------------------------------------------------output---------------------------------------------------------------
#  first input1
# Second input1
# [array([[0.10893457]], dtype=float32)]
#  first input0
# Second input0
# [array([[0.02925059]], dtype=float32)]
#  first input1
# Second input0
# [array([[0.9223157]], dtype=float32)]
#  first input0
# Second input1
# [array([[0.9196354]], dtype=float32)]
