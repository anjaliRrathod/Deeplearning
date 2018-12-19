import tensorflow as tf
from NetworkBuilder import NetworkBuilder
from DataSetGenerator import DataSetGenerator, seperateData
import datetime
import numpy as np
import os


with tf.name_scope("Input") as scope:
    input_img = tf.placeholder(dtype='float', shape=[None, 128, 128, 1], name="input")

with tf.name_scope("Target") as scope:
    target_labels = tf.placeholder(dtype='float', shape=[None, 2], name="Targets")

with tf.name_scope("Keep_prob_input") as scope:
    keep_prob = tf.placeholder(dtype='float',name='keep_prob')

nb = NetworkBuilder()

with tf.name_scope("ModelV2") as scope:
    model = input_img
    model = nb.attach_conv_layer(model, 32, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, 32, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooiling_layer(model)

    model = nb.attach_conv_layer(model, 64, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, 64, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooiling_layer(model)

    model = nb.attach_conv_layer(model, 128, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, 128, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooiling_layer(model)

    model = nb.attach_flatten_layer(model)
    model = nb.attach_dense_layer(model, 200, summary=True)
    model = nb.attach_sigmoid_layer(model)
    model = nb.attach_dense_layer(model, 32, summary=True)
    model = nb.attach_sigmoid_layer(model)
    model = nb.attach_dense_layer(model, 2)
    prediction = nb.attach_softmax_layer(model)


with tf.name_scope("Optimization") as scope:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=target_labels)
    cost = tf.reduce_mean(cost)
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.AdamOptimizer().minimize(cost, global_step=global_step)

with tf.name_scope('accuracy') as scope:
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(target_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

dg = DataSetGenerator("./train")


epochs = 10
batchSize = 10

saver = tf.train.Saver()
model_save_path="./saved model v2/"
model_name='model'

with tf.Session() as sess:
    summaryMerged = tf.summary.merge_all()

    filename = "./summary_log/run" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%s")
    # setting global steps
    tf.global_variables_initializer().run()

    if os.path.exists(model_save_path+'checkpoint'):
        # saver = tf.train.import_meta_graph('./saved '+modelName+'/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
    writer = tf.summary.FileWriter(filename, sess.graph)

    for epoch in range(epochs):
        batches = dg.get_mini_batches(batchSize,(128,128), allchannel=False)
        for imgs ,labels in batches:
            imgs=np.divide(imgs, 255)
            error, sumOut, acu, steps,_ = sess.run([cost, summaryMerged, accuracy,global_step,optimizer],
                                            feed_dict={input_img: imgs, target_labels: labels})
            writer.add_summary(sumOut, steps)
            print("epoch=", epoch, "Total Samples Trained=", steps*batchSize, "err=", error, "accuracy=", acu)
            if steps % 100 == 0:
                print("Saving the model")
                saver.save(sess, model_save_path+model_name, global_step=steps)