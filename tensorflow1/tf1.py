import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

#tf.debugging.set_log_device_placement(True)
tf.executing_eagerly()


def test1():
    xo = tf.constant(3.0, name = "a")
    print(tf.is_tensor(xo))
    print(xo)
    print(xo.shape)
    print(xo.dtype)
    xn = xo.numpy()
    xn = np.square(xn) # sqrt
    print(xn)
    print(tf.convert_to_tensor(xn))
    print(xo+5)

    x1 = tf.constant([1.1,2.2,3.3,4.4])
    print(x1 + tf.constant([1.1,2.2,3.3,4.4]))
    print(tf.add(x1, tf.constant([1.1,2.2,3.3,4.4])))
    x2 = tf.cast(xo, tf.float32)
    print(x2)
    print(tf.multiply(x1, tf.constant([1.1, 2.2, 3.3, 4.4])))
    print(tf.zeros([3,5], tf.int32))
    print(tf.ones([3,5], tf.int32))
    print(tf.reshape(tf.ones([3,5], tf.int32), [1,15])) # reshape r1*c1=r2*c2

    v0 = tf.Variable([[1.1, 2.2], [3.3, 4.4]])
    v0[0, 0].assign(100)
    print(v0)
    v0.assign_add([[1.2, 2.2], [3.3, 4.4]]) # assign_sub
    print(v0)


@tf.function
def square(x):
    return tf.matmul(x,x)


@tf.function
def p(s):
    print(s) # executed only once per data type
    tf.print(s)


def test2():
    x= [[10.]]
    res = tf.matmul(x,x)
    print(res)
    print(square(x))
    # get concrete polymorphic function for decorated function
    concrete_function = square.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.int32))
    print(concrete_function)
    p("one")
    p("two")
    # use tf.range(x) to iterate over a vector

    ...

def test3():
    x = tf.Variable(4.0)
    # with tf.GradientTape(watch_accessed_variables=False) as tape: # disable auto tracking of variables
    with tf.GradientTape() as tape:
        # tape.watch(x) allows to track variables
        # variables that are directly  involved in gradient computation are auto-tracked only
        # if for ex. using variable in an if statement, tape.watch(x1) must be used
        y=tf.square(x)
        dydx =tape.gradient(y,x) # gradient resources are released after call to .gradient() method
        print(dydx)

    w = tf.Variable(tf.random.normal((2,2)))
    print(w)
    print(tf.reduce_mean(y**2))

    b = tf.Variable(7.0)
    x = tf.Variable([[7.0, 8.0]], trainable=True)
    with tf.GradientTape() as tape:
        y=tf.matmul(x,w) + b
        loss =tf.reduce_mean(y**2) # gradient resources are released after call to .gradient() method
        [sldw, dldb] = tape.gradient(loss, [w,b])
        print(sldw)

    with tf.GradientTape() as tape:
        layer = tf.keras.layers.Dense(2, activation='relu')
        y = layer(x)
        loss =tf.reduce_sum(y**2) # gradient resources are released after call to .gradient() method
        grad= tape.gradient(loss, layer.trainable_variables)
        print(grad)

def test4():
    W_true = 2
    b_true = 0.5
    x = np.linspace(0,3,130)
    y = W_true * x + b_true + np.random.randn(*x.shape) * 0.5
    plt.figure(figsize=(8,8))
    plt.scatter(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Training data")
    plt.show()

    lin_model = LineraModel()
    w, b = [], []
    epochs = 50
    lr = 0.15
    for epoch_cnt in range(epochs):
        w.append(lin_model.weight.numpy())
        b.append(lin_model.bias.numpy())
        real_loss = loss(y, lin_model(y))
        train(lin_model, x ,y ,lr=lr)
        print(f"Epoch cnt {epoch_cnt} Loss value: {real_loss.numpy()}")

    plt.figure(figsize=(8,8))
    plt.plot(list(range(epochs)), w, 'r', range(epochs), b, 'b')
    plt.plot([W_true]*epochs, 'r--', [b_true] * epochs, 'b--')
    plt.legend(['W', 'b', 'true W', 'true b'])
    plt.show()
    print(f"{lin_model.weight.numpy()}, {lin_model.bias.numpy()}")

class LineraModel:
    def __init__(self):
        self.weight = tf.Variable(np.random.randn(), name = "W")
        self.bias = tf.Variable(np.random.randn(), name = "b")

    def __call__(self, x):
        return self.weight * x +self.bias

def loss(y, y_pred):
    return tf.reduce_mean(tf.square(y-y_pred))

def train(liner_model, x, y, lr= 0.01):
    with tf.GradientTape() as tape:
        y_pred = liner_model(x)
        cur_loss = loss(y, y_pred)
        d_weight, d_bias = tape.gradient(cur_loss, [liner_model.weight, liner_model.bias])
        liner_model.weight.assign_sub(lr * d_weight)
        liner_model.bias.assign_sub(lr * d_bias)


if __name__ == "__main__":
    #test1()
    #test2()
    #test3()
    test4()
