# -- encoding:utf-8 --
"""
@File : 06_tf案例
@Author: Octal_H
@Date : 2019/11/18
@Desc : 
"""
import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1.实现一个累加器,并且每一步均输出累加器的结果值

# 定义变量
# x = tf.Variable(0)
#
#
# # 变量更新
# op_x = tf.assign(ref=x, value=tf.add(x,1))
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for i in range(5):
#     print(type(x))
#     # print(sess.run(x))
#     # x = x + 1
#     print(sess.run(op_x))


# # 编写一段代码,实现动态的更新变量的维度数目
# x = tf.Variable(initial_value=[], validate_shape=False, trainable=False, dtype=tf.float32,name="x")
# # validate_shape设置为True,表示变量更新的时候进行shape检查,默认为True
# # trainable 不加载到内存空间中
#
# concat = tf.concat([x, [0., 0.]], axis=0)
# assign_op = tf.assign(ref=x, value=concat,validate_shape=False)
# # validate_shape 设置为True，表示变量更新的时候进行shape检查，默认为True
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(5):
#         r_x = sess.run(x)
#         print(r_x)
#         sess.run(assign_op)



# 阶乘
x = tf.Variable(1, dtype=tf.int32)
# 变量更新
i = tf.placeholder(dtype=tf.int32, shape=None)
op_x = tf.assign(ref=x, value=tf.multiply(x, i))
with tf.control_dependencies([op_x]):
    # data 呈现的控制器的数据形式
    # message 描述
    x1 = tf.Print(x, data=[x, x.read_value()], message='sum,sum read=')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for y in range(2, 5):
    print(sess.run(x1, feed_dict={i: y}))
    # x=x+1