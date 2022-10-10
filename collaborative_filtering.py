import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from recsys_utils import *

# Load data
Y, R = load_ratings_small()

num_movies, num_users = Y.shape
num_features = 100

X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float32), name='X')
W = tf.Variable(tf.random.normal((num_users,  num_features), dtype=tf.float32), name='W')
b = tf.Variable(tf.random.normal((1,          num_users), dtype=tf.float32), name='b')

optimizer = tf.keras.optimizers.Adam(learning_rate=0.3)
lambda_ = 1
num_iters = 200

print("Y", Y.shape, "R", R.shape)
print("X", X.shape)
print("W", W.shape)
print("b", b.shape)
print("num_features", num_features)
print("num_movies",   num_movies)
print("num_users",    num_users)


# From the matrix, we can compute statistics like average rating.
tsmean =  np.mean(Y[0, R[0, :].astype(bool)])
print(f"Average rating for movie 1 : {tsmean:0.3f} / 5" )

class collaborativeFiltering:
    def __init__(self, X, W, b, Y, R, lambda_, optimizer, num_iters):
        self.X = X
        self.W = W
        self.b = b
        self.Y = Y
        self.R = R
        self.lambda_ = lambda_
        self.optimizer = optimizer
        self.num_iters = num_iters

    def cofi_cost_func(self):
        preds = (tf.linalg.matmul(self.X, tf.transpose(self.W)) + self.b) * self.R
        cost = (0.5 * tf.reduce_sum((self.Y - preds)**2))+\
               (0.5 * self.lambda_ * tf.reduce_sum(self.X**2)) + (0.5 * self.lambda_ * tf.reduce_sum(self.W**2))
        return cost

    @tf.function
    def train_step(self):
        trainable_params = [self.X, self.W, self.b]
        with tf.GradientTape() as tape:
            cost = self.cofi_cost_func()
        grads = tape.gradient(cost, trainable_params)
        self.optimizer.apply_gradients(zip(grads, trainable_params))
        return cost
    
    def __call__(self):
        t1 = time.time()
        for i in range(self.num_iters):
            cost = self.train_step()
            if (i+1) % 20 == 0:
                print(f'\rCost at iteration {i+1}: {cost:.2f}', end='')
        print(f'\n\nTime required: {(time.time() - t1):.2f} sec')

cofi_object = collaborativeFiltering(X, W, b, Y, R, lambda_, optimizer, num_iters)
cofi_object()
