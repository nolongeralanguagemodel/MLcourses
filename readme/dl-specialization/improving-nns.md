# Improving NNs

_Guideline_:

* Choose a dev set and test set to reflect data you expect to get in the future.
* The dev and test sets should be just big enough to represent accurately the performance of the model.

<figure><img src="../../.gitbook/assets/image (54).png" alt=""><figcaption></figcaption></figure>

**Bias / Variance**

| error type      | high variance | high bias | high bias, high variance | low bias, low variance |
| --------------- | :-----------: | :-------: | :----------------------: | :--------------------: |
| Train set error |       1%      |    15%    |            15%           |          0.5%          |
| Dev set error   |      11%      |    16%    |            30%           |           1%           |

> When we discuss prediction models, prediction errors can be decomposed into two main subcomponents we care about: error due to "bias" and error due to "variance". There is a tradeoff between a model's ability to minimize bias and variance. Understanding these two types of error can help us diagnose model results and avoid the mistake of over- or under-fitting.

&#x20;[Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html).

<figure><img src="../../.gitbook/assets/image (55).png" alt=""><figcaption></figcaption></figure>

* For a high bias problem, getting more training data is actually not going to help.
* Back in the pre-deep learning era, we didn't have as many tools that just reduce bias or that just reduce variance without hurting the other one.
* In the modern deep learning, big data era, getting a bigger network and more data almost always just reduces bias without necessarily hurting your variance, so long as you regularize appropriately.
* This has been one of the big reasons that deep learning has been so useful for supervised learning.
* The main cost of training a big neural network is just computational time, so long as you're regularizing.

#### Regularization

<figure><img src="../../.gitbook/assets/image (14).png" alt=""><figcaption></figcaption></figure>

`b` is just one parameter over a very large number of parameters, so no need to include it in the regularization.

| regularization    | description                                                                     |
| ----------------- | ------------------------------------------------------------------------------- |
| L2 regularization | most common type of regularization                                              |
| L1 regularization | w vector will have a lot of zeros, so L1 regularization makes your model sparse |

<figure><img src="../../.gitbook/assets/image (15).png" alt="" width="375"><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (17).png" alt="" width="220"><figcaption></figcaption></figure>

**Why regularization reduces overfitting**

* If we make regularization lambda to be very big, then weight matrices will be set to be reasonably close to zero, effectively zeroing out a lot of the impact of the hidden units. Then the simplified neural network becomes a much smaller neural network, eventually almost like a logistic regression. We'll end up with a much smaller network that is therefore less prone to overfitting.
* Taking activation function `g(Z)=tanh(Z)` as example, if lambda is large, then weights `W` are small and subsequently `Z` ends up taking relatively small values, where `g` and `Z` will be roughly linear which is not able to fit those very complicated decision boundary, i.e., less able to overfit.

_Implementation tips_:

Without regularization term, we should see the cost function decreases monotonically in the plot. Whereas in the case of regularization, to debug gradient descent make sure that we plot `J` with a regularization term; otherwise, if we plot only the first term (the old J), we might not see a decrease monotonically.

#### Dropout Regularization

* With dropout, what we're going to do is go through each of the layers of the network and set some probability of eliminating a node in neural network. It's as if on every iteration you're working with a smaller neural network, which has a regularizing effect.
* Inverted dropout technique, `a3 = a3 / keep_prob`, ensures that the expected value of `a3` remains the same, which makes test time easier because you have less of a scaling problem.

![](<../../.gitbook/assets/image (18).png>)\


**Other regularization methods**

* **Data augmentation**: getting more training data can be expensive and somtimes can't get more data, so flipping horizontally, random cropping, random distortion and translation of image can make additional fake training examples.
*   **Early stopping**: stopping halfway to get a mid-size `w`.

    * _Disadvantage_: early stopping couples two tasks of machine learning, optimizing the cost function `J` and not overfitting, which are supposed to be completely separate tasks, to make things more complicated.
    * _Advantage_: running the gradient descent process just once, you get to try out values of small `w`, mid-size `w`, and large `w`, without needing to try a lot of values of the L2 regularization hyperparameter lambda.



**Vanishing / Exploding gradients**

* In a very deep network derivatives or slopes can sometimes get either very big or very small, maybe even exponentially, and this makes training difficult.
* The weights W, if they're all just a little bit bigger than one or just a little bit bigger than the identity matrix, then with a very deep network the activations can explode. And if W is just a little bit less than identity, the activations will decrease exponentially.

**Weight Initialization for Deep Networks**

A partial solution to the problems of vanishing and exploding gradients is better or more careful choice of the random initialization for neural network.

For a single neuron, suppose we have `n` features for the input layer, then we want `Z = W1X1 + W2X2 + ... + WnXn` not blow up and not become too small, so the larger `n` is, the smaller we want `Wi` to be.

* It's reasonable to set variance of `Wi` to be equal to `1/n`
* It helps reduce the vanishing and exploding gradients problem, because it's trying to set each of the weight matrices `W` not too much bigger than `1` and not too much less than `1`.
* Generally for layer `l`, set `W[l]=np.random.randn(shape) * np.sqrt(1/n[l-1])`.
  * For `relu` activation, set `Var(W)=2/n` by `W[l]=np.random.randn(shape) * np.sqrt(2/n[l-1])`.&#x20;
  * For `tanh` activation, `W[l]=np.random.randn(shape) * np.sqrt(1/n[l-1])`. (Xavier initialization)
  * `W[l]=np.random.randn(shape) * np.sqrt(2/(n[l-1]+n[l]))` (Yoshua Bengio)
* `1` or `2` in variance `Var(W)=1/n or 2/n` can be a hyperparameter, but not as important as other hyperparameters.

_A well chosen initialization can_:

* Speed up the convergence of gradient descent
* Increase the odds of gradient descent converging to a lower training (and generalization) error

_Implementation tips_:

* The weights `W[l]` should be initialized randomly to _break symmetry_ and make sure different hidden units can learn different things. Initializing all the weights to zero results in the network failing to break symmetry. This means that every neuron in each layer will learn the same thing.
* It is however okay to initialize the biases `b[l]` to zeros. Symmetry is still broken so long as `W[l]` is initialized randomly.
* Initializing weights to very large random values does not work well.
* Hopefully intializing with small random values does better. The important question is: how small should be these random values be? He initialization works well for networks with ReLU activations. In other cases, try other initializations.

#### Optimization Algorithms

**Understanding mini-batch gradient descent**

| batch size          |            method           | description                                                                                                                                                                                                                                                                                               | guidelines                                                                                                                                                  |
| ------------------- | :-------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| =`m`                |    batch gradient descent   | <p>cost function decreases on every iteration;<br>but too long per iteration.</p>                                                                                                                                                                                                                         | for a small training set (<2000).                                                                                                                           |
| =`1`                | stochastic gradient descent | <p>cost function oscillates, can be extremely noisy;<br>wander around minimum;<br>lose speedup from vectorization, inefficient.</p>                                                                                                                                                                       | use a smaller learning rate when it oscillates too much.                                                                                                    |
| between `1` and `m` | mini-batch gradient descent | <p>somewhere in between, vectorization advantage, faster;<br>not guaranteed to always head toward the minimum but more consistently in that direction than stochastic descent;<br>not always exactly converge, may oscillate in a very small region, reducing the learning rate slowly may also help.</p> | <p>mini-batch size is a hyperparameter;<br>batch size better in [64, 128, 256, 512], a power of 2;<br>make sure that mini-batch fits in CPU/GPU memory.</p> |

**Exponentially Weighted Averages**

Moving averages are favored statistical tools of active traders to measure momentum. There are three MA methods:

| MA methods                        | calculations                                                                                                                                           |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| simple moving average (SMA)       | calculated from the average closing prices for a specified period                                                                                      |
| weighted moving average (WMA)     | calculated by multiplying the given price by its associated weighting (assign a heavier weighting to more current data points) and totaling the values |
| exponential moving average (EWMA) | also weighted toward the most recent prices, but the rate of decrease is exponential                                                                   |

**Gradient descent with momentum**

Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, and so the path taken by mini-batch gradient descent will "oscillate" toward convergence. Using momentum can reduce these oscillations.

* gradient descent with momentum, which computes an EWA of gradients to update weights almost always works faster than the standard gradient descent algorithm.
* algorithm has two hyperparameters of `alpha`, the learning rate, and `beta` which controls your exponentially weighted average. common value for `beta` is `0.9`.
* don't bother with bias correction

[![momentum-algo](https://github.com/lijqhs/deeplearning-notes/raw/main/C2-Improving-Deep-Neural-Networks/img/momentum-algo.png)](https://github.com/lijqhs/deeplearning-notes/blob/main/C2-Improving-Deep-Neural-Networks/img/momentum-algo.png)

_Implementation tips_:

* If `β = 0`, then this just becomes standard gradient descent without momentum.
* The larger the momentum `β` is, the smoother the update because the more we take the past gradients into account. But if `β` is too big, it could also smooth out the updates too much.
* Common values for `β` range from `0.8` to `0.999`. If you don't feel inclined to tune this, `β = 0.9` is often a reasonable default.
* It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.

**RMSprop**

RMSprop(root mean square), similar to momentum, has the effects of damping out the oscillations in gradient descent and mini-batch gradient descent and allowing you to maybe use a larger learning rate alpha.

The algorithm computes the exponentially weighted averages of the squared gradients and updates weights by the square root of the EWA.

```
for iteration t:
  # compute dW, db on mini-batch

  S_dW = (beta * S_dW) + (1 - beta) * dW^2
  S_db = (beta * S_db) + (1 - beta) * db^2
  W = W - alpha * dW / sqrt(S_dW + 𝜀)   # 𝜀: small number(10^-8) to avoid dividing by zero
  b = b - alpha * db / sqrt(S_db + 𝜀)
```

**Adam optimization algorithm**

* Adam (Adaptive Moment Estimation) optimization algorithm is basically putting momentum and RMSprop together and combines the effect of gradient descent with momentum together with gradient descent with RMSprop.
* This is a commonly used learning algorithm that is proven to be very effective for many different neural networks of a very wide variety of architectures.
* In the typical implementation of Adam, bias correction is on.

```
V_dW = 0
V_db = 0
S_dW = 0
S_db = 0

for iteration t:
  # compute dW, db using mini-batch                
  
  # momentum
  V_dW = (beta1 * V_dW) + (1 - beta1) * dW     
  V_db = (beta1 * V_db) + (1 - beta1) * db     
  
  # RMSprop
  S_dW = (beta2 * S_dW) + (1 - beta2) * dW^2   
  S_db = (beta2 * S_db) + (1 - beta2) * db^2   
  
  # bias correction
  V_dW_c = V_dW / (1 - beta1^t)      
  V_db_c = V_db / (1 - beta1^t)
  S_dW_c = S_dW / (1 - beta2^t)
  S_db_c = S_db / (1 - beta2^t)
          
  W = W - alpha * V_dW_c / (sqrt(S_dW_c) + 𝜀)
  b = b - alpha * V_db_c / (sqrt(S_db_c) + 𝜀)
```

_Implementation tips_:

1. It calculates an exponentially weighted average of past gradients, and stores it in variables `V_dW,V_db` (before bias correction) and `V_dW_c,V_db_c` (with bias correction).
2. It calculates an exponentially weighted average of the squares of the past gradients, and stores it in variables `S_dW,S_db` (before bias correction) and `S_dW_c,S_db_c` (with bias correction).
3. It updates parameters in a direction based on combining information from "1" and "2".

| hyperparameter                                 | guideline |
| ---------------------------------------------- | --------- |
| `learning rate`                                | tune      |
| `beta1` (parameter of the momentum, for `dW`)  | `0.9`     |
| `beta2` (parameter of the RMSprop, for `dW^2`) | `0.999`   |
| `𝜀` (avoid dividing by zero)                  | `10^-8`   |

Adam paper: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

**Learning rate decay**

The learning algorithm might just end up wandering around, and never really converge, because you're using some fixed value for alpha. Learning rate decay methods can help by making learning rate smaller when optimum is near. There are several decay methods:

| decay factor                       | description        |
| ---------------------------------- | ------------------ |
| `0.95^epoch_num`                   | exponential decay  |
| `k/sqrt(epoch_num)` or `k/sqrt(t)` | polynomial decay   |
| discrete staircase                 | piecewise constant |
| manual decay                       | --                 |

**The problem of local optima**

* First, you're actually pretty unlikely to get stuck in bad local optima, but much more likely to run into a saddle point, so long as you're training a reasonably large neural network, save a lot of parameters, and the cost function J is defined over a **relatively high dimensional space**.
* Second, that plateaus are a problem and you can actually make learning pretty slow. And this is where algorithms like **momentum** or **RMSProp** or **Adam** can really help your learning algorithm.

This is what a saddle point look like.

[![saddle-point](https://github.com/lijqhs/deeplearning-notes/raw/main/C2-Improving-Deep-Neural-Networks/img/saddle-point.png)](https://github.com/lijqhs/deeplearning-notes/blob/main/C2-Improving-Deep-Neural-Networks/img/saddle-point.png)

**Quick notes for optimization algorithms**

there are several steps in the neural network implementation:

1. Initialize parameters / Define hyperparameters
2. Loop for num\_iterations:
   1. Forward propagation
   2. Compute cost function
   3. Backward propagation
   4. **Update parameters (using parameters, and grads from backprop)**
3. Use trained parameters to predict labels

When we create `momentum`, `RMSprop` or `Adam` optimization methods, what we do is to implement algorithms in the **update parameters** step. A good practice is to wrap them up as options so we can compare them during our alchemy training：

```
if optimizer == "gd":
    parameters = update_parameters_with_gd(parameters, grads, learning_rate)
elif optimizer == "momentum":
    parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
elif optimizer == "adam":
    t = t + 1 # Adam counter
    parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
```

\


#### Hyperparameter tuning <a href="#user-content-hyperparameter-tuning" id="user-content-hyperparameter-tuning"></a>

**Tuning process**

Importance of hyperparameters (roughly):

| importance level | hyperparameters                                                                           |
| :--------------: | ----------------------------------------------------------------------------------------- |
|       first      | learning rate `alpha`                                                                     |
|      second      | <p>momentum term <code>beta</code><br>mini-batch size<br>number of hidden units</p>       |
|       third      | <p>number of layers<br>learning rate decay<br>Adam <code>beta1, beta2, epsilon</code></p> |

_Tuning tips_:

* Choose points at random, not in a grid
* Optionally use a coarse to fine search process

#### Batch Normalization <a href="#user-content-batch-normalization" id="user-content-batch-normalization"></a>

**Normalizing activations in a network**

* Batch normalization makes your hyperparameter search problem much easier, makes your neural network much more robust.
* What batch norm does is it applies that normalization process not just to the input layer, but to the values even deep in some hidden layer in the neural network. So it will apply this type of normalization to normalize the mean and variance of `z[i]` of hidden units.
* One difference between the training input and these hidden unit values is that you might not want your hidden unit values be forced to have mean 0 and variance 1.
  * For example, if you have a sigmoid activation function, you don't want your values to always be clustered in the normal distribution around `0`. You might want them to have a larger variance or have a mean that's different than 0, in order to better take advantage of the nonlinearity of the sigmoid function rather than have all your values be in just this linear region (near `0` on sigmoid function).
  * What it does really is it then shows that your hidden units have standardized mean and variance, where the mean and variance are controlled by two explicit parameters `gamma` and `beta` which the learning algorithm can set to whatever it wants.

<figure><img src="../../.gitbook/assets/image (20).png" alt=""><figcaption></figcaption></figure>

**Fitting Batch Norm into a neural network**

* `𝛽[1],𝛾[1],𝛽[2],𝛾[2],⋯,𝛽[𝐿],𝛾[𝐿]` can also be updated using gradient descent with momentum (or RMSprop, Adam). `𝛽[l],𝛾[l]` have the shape with `z[l]`.
* Similar computation can also be applied to mini-batches.
* With batch normalization, the parameter `b[l]` can be eliminated. So `w[l],𝛽[l],𝛾[l]` need to be trained.
* The parameter `𝛽` here has nothing to do with the `beta` in the momentum, RMSprop or Adam algorithms.

[<img src="https://github.com/lijqhs/deeplearning-notes/raw/main/C2-Improving-Deep-Neural-Networks/img/batch-norm-nn.png" alt="batch-norm-nn" data-size="original">](https://github.com/lijqhs/deeplearning-notes/blob/main/C2-Improving-Deep-Neural-Networks/img/batch-norm-nn.png)

#### Multi-class classification <a href="#user-content-multi-class-classification" id="user-content-multi-class-classification"></a>

**Softmax Regression**

Use softmax activation function.

```
def softmax(z):
    return np.exp(z) / sum(np.exp(z))

z = [1,0.5,-2,1,3]
print(softmax(z)) 
# array([0.09954831, 0.0603791 , 0.00495622, 0.09954831, 0.73556806])
```

**Training a softmax classifier**

Softmax regression is a generalization of logistic regression to more than two classes.

#### Introduction to programming frameworks <a href="#user-content-introduction-to-programming-frameworks" id="user-content-introduction-to-programming-frameworks"></a>

**Deep learning frameworks**

* Caffe/Caffe2
* CNTK
* DL4J
* Keras
* Lasagne
* mxnet
* PaddlePaddle
* TensorFlow
* Theano
* Torch

_Choosing deep learning frameworks_:

* Ease of programming (development and deployment)
* Running speed
* Truly open (open source with good governance)

**Tensorflow**

* The two main object classes in tensorflow are _Tensors_ and _Operators_.
* When we code in tensorflow we have to take the following steps:
  * Create a graph containing Tensors (_Variables_, _Placeholders_ ...) and _Operations_ (`tf.matmul`, `tf.add`, ...)
  * Create a _session_
  * Initialize the _session_
  * Run the _session_ to execute the graph
* We might need to execute the graph multiple times when implementing `model()`
* The backpropagation and optimization is automatically done when running the session on the "optimizer" object.

```
import numpy as np 
import tensorflow as tf

coefficients = np.array([[1], [-20], [25]])
w = tf.Variable([0],dtype=tf.float32)
x = tf.placeholder(tf.float32, [3,1])
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]    # (w-5)**2
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init) 
print(session.run(w))

for i in range(1000):
  session.run(train, feed_dict={x:coefficients})
print(session.run(w))
```
