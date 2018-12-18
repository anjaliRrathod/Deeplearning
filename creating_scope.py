#This will help you to create Neural Network for Xor gate
#command to run tensorboard --logdir=./summary_log


#This is for grouping the network to see in tensorboard graph

import tensorflow as tf

Target =tf.placeholder('float', shape=[None, 1], name="Target")

with tf.name_scope("Input_Layer") as scope:
    Input = tf.placeholder('float', shape=[None, 2], name="Input")
    inputBias = tf.Variable(initial_value=tf.random_normal(shape=[3], stddev=0.4),dtype='float', name="input_bias")


with tf.name_scope("Hidden_Layer") as scope :
    weights = tf.Variable(initial_value=tf.random_normal(shape=[2, 3], stddev=0.4),dtype='float', name="hidden_weights")
    hiddenBias = tf.Variable(initial_value=tf.random_normal(shape=[1], stddev=0.4),dtype='float', name="hidden_bias")
    tf.summary.histogram(name="Weights_1",values=weights)
    hiddenlayer = tf.matmul(Input, weights) + inputBias
    hiddenlayer = tf.sigmoid(hiddenlayer, name="Hidden_layer_activation")


with tf.name_scope("Output_Layer") as scope :
    outputweights = tf.Variable(initial_value=tf.random_normal(shape=[3, 1], stddev=0.4), dtype='float',
                                name="output_weights")
    tf.summary.histogram(name="Weights_2", values=outputweights)
    output = tf.matmul(hiddenlayer, outputweights) + hiddenBias
    output = tf.sigmoid(output, name="output_layer_activation")


with tf.name_scope("Optimization_group") as scope :
    cost = tf.squared_difference(Target, output)
    cost = tf.reduce_mean(cost)
    tf.summary.scalar("error", cost)
    optimizer = tf.train.AdamOptimizer().minimize(cost)


#Exor input and output

inp=[[1,1],[1,0],[0,1],[0,0]]
out=[[0],[1],[1],[0]]
epochs=4000

import datetime

with tf.Session() as sess:

    tf.global_variables_initializer().run()
    mergedSummary=tf.summary.merge_all()
    filename="./summary_log/run" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%s")
    writer=tf.summary.FileWriter(filename,sess.graph)
    for i in range(epochs) :
        err, _ ,summaryOutput=sess.run([cost,optimizer,mergedSummary],feed_dict={Input:inp,Target:out})
        writer.add_summary(summaryOutput,i)


    while  True :
        inp=[[0,0]]
        inp[0][0]=input(" first input")
        inp[0][1]=input("Second input")
        print(sess.run([output],feed_dict={Input:inp}))


#---------------------------------------------------------------------------output---------------------------------------------------------------
