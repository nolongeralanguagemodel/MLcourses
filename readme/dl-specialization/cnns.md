# CNNs

<figure><img src="../../.gitbook/assets/image (21).png" alt=""><figcaption></figcaption></figure>

**Padding**

In order to fix the following two problems, padding is usually applied in the convolutional operation.

* Every time you apply a convolutional operator the image shrinks.
* A lot of information from the edges of the image is thrown away.

_Notations_:

* image size: `n x n`
* convolution size: `f x f`
* padding size: `p`

_Output size after convolution_:

* without padding: `(n-f+1) x (n-f+1)`
* with padding: `(n+2p-f+1) x (n+2p-f+1)`

_Convention_:

* Valid convolutions: no padding
* Same convolutions: output size is the same as the input size
* `f` is usually odd

**Strided Convolutions**

_Notation_:

* stride `s`

_Output size after convolution_: `floor((n+2p-f)/s+1) x floor((n+2p-f)/s+1)`

_Conventions_:

* The filter must lie entirely within the image or the image plus the padding region.
* In the deep learning literature by convention, a convolutional operation (maybe better _called cross-correlation_) is what we usually do not bother with a flipping operation, which is included before the product and summing step in a typical math textbook or a signal processing textbook.
  * In the latter case, the filter is flipped vertically and horizontally.

**Convolutions Over Volume**

For a RGB image, the filter itself has three layers corresponding to the red, green, and blue channels.

`height x width x channel`

`n x n x nc` \* `f x f x nc` --> `(n-f+1) x (n-f+1) x nc'`

**Simple Convolutional Network**

Types of layer in a convolutional network:

* Convolution (CONV)
* Pooling (POOL)
* Fully connected (FC)

**Pooling Layers**

* One interesting property of max pooling is that it has a set of hyperparameters but it has no parameters to learn. There's actually nothing for gradient descent to learn.
* Formulas that we had developed previously for figuring out the output size for conv layer also work for max pooling.
* The max pooling is used much more often than the average pooling.
* When you do max pooling, usually, you do not use any padding.

**CNN Example**

* Because the pooling layer has no weights, has no parameters, only a few hyper parameters, I'm going to use a convention that `CONV1` and `POOL1` shared together.
* As you go deeper usually the _height_ and _width_ will decrease, whereas the number of _channels_ will increase.
* max pooling layers don't have any parameters
* The conv layers tend to have relatively few parameters and a lot of the parameters tend to be in the fully collected layers of the neural network.
* The activation size tends to maybe go down _gradually_ as you go deeper in the neural network. If it drops too quickly, that's usually not great for performance as well.

<figure><img src="../../.gitbook/assets/image (22).png" alt=""><figcaption></figcaption></figure>

**Why Convolutions**

There are two main advantages of convolutional layers over just using fully connected layers.

* Parameter sharing: A feature detector (such as a vertical edge detector) that’s useful in one part of the image is probably useful in another part of the image.
* Sparsity of connections: In each layer, each output value depends only on a small number of inputs.

Through these two mechanisms, a neural network has a lot fewer parameters which allows it to be trained with smaller training cells and is less prone to be overfitting.

* Convolutional structure helps the neural network encode the fact that an image shifted a few pixels should result in pretty similar features and should probably be assigned the same output label.
* And the fact that you are applying the same filter in all the positions of the image, both in the early layers and in the late layers that helps a neural network automatically learn to be more robust or to better capture the desirable property of translation invariance.
* Classic networks
  * LeNet-5
  * AlexNet
  * VGG
* ResNet
* Inception

**Classic Networks**

**LeNet-5**

\
![](<../../.gitbook/assets/image (23).png>)Some difficult points about reading the LeNet-5 paper:

* Back then, people used sigmoid and tanh nonlinearities, not relu.
* To save on computation as well as some parameters, the original LeNet-5 had some crazy complicated way where different filters would look at different channels of the input block. And so the paper talks about those details, but the more modern implementation wouldn't have that type of complexity these days.
* One last thing that was done back then I guess but isn't really done right now is that the original LeNet-5 had a non-linearity after pooling, and I think it actually uses sigmoid non-linearity after the pooling layer.
* Andrew Ng recommend focusing on section two which talks about this architecture, and take a quick look at section three which has a bunch of experiments and results, which is pretty interesting. Later sections talked about the graph transformer network, which isn't widely used today.

**AlexNet**

\
![](<../../.gitbook/assets/image (24).png>)

* AlexNet has a lot of similarities to LeNet (60,000 parameters), but it is much bigger (60 million parameters).
* The paper had a complicated way of training on two GPUs since GPU was still a little bit slower back then.
* The original AlexNet architecture had another set of a layer called local response normalization, which isn't really used much.
* Before AlexNet, deep learning was starting to gain traction in speech recognition and a few other areas, but it was really just paper that convinced a lot of the computer vision community to take a serious look at deep learning, to convince them that deep learning really works in computer vision.

**VGG-16**

\
![](<../../.gitbook/assets/image (25).png>)

* Filters are always `3x3` with a stride of `1` and are always `same` convolutions.
* VGG-16 has 16 layers that have weights. A total of about 138 million parameters. Pretty large even by modern standards.
* It is the simplicity, or the uniformity, of the VGG-16 architecture made it quite appealing.
  * There is a few conv-layers followed by a pooling layer which reduces the height and width by a factor of `2`.
  * Doubling through every stack of conv-layers is a simple principle used to design the architecture of this network.
* The main downside is that you have to train a large number of parameters.

**ResNets**

Paper: Deep Residual Learning for Image Recognition

<figure><img src="../../.gitbook/assets/image (26).png" alt=""><figcaption></figcaption></figure>

* Deeper neural networks are more difficult to train. They present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.
* When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly. The paper address the degradation problem by introducing a deep residual learning framework. Instead of hoping each few stacked layers directly fit a desired underlying mapping, they explicitly let these layers fit a residual mapping.
* The paper authors show that: 1) Their extremely deep residual nets are easy to optimize, but the counterpart "plain" nets (that simply stack layers) exhibit higher training error when the depth increases; 2) Their deep residual nets can easily enjoy accuracy gains from greatly increased depth, producing results substantially better than previous networks.

<figure><img src="../../.gitbook/assets/image (27).png" alt=""><figcaption></figcaption></figure>

Formally, denoting the desired underlying mapping as `H(x)`, they let the stacked nonlinear layers fit another mapping of `F(x):=H(x)-x`. The original mapping `H(x)` is recast into `F(x)+x`. If the added layers can be constructed as identity mappings, a deeper model should have training error no greater than its shallower counterpart.

![](<../../.gitbook/assets/image (28).png>)\


**Why ResNets**

* Doing well on the training set is usually a prerequisite to doing well on your hold up or on your depth or on your test sets. So, being able to at least train ResNet to do well on the training set is a good first step toward that.
* But if you make a network deeper, it can hurt your ability to train the network to do well on the training set. It is not true or at least less true when training a ResNet.
  * If we use `L2` regularization on `a[l+2]=g(Z[l+2]+a[l])=g(W[l+2]a[l+1]+b[l+2]+a[l])`, and if the value of `W[l+2],b[l+2]` shrink to zero, then `a[l+2]=g(a[l])=a[l]` since we use `relu` activation and `a[l]` is also non-negative. So we just get back `a[l]`. This shows that the identity function is easy for residual block to learn.
  * It's easy to get `a[l+2]` equals to `a[l]` because of this skip connection. What this means is that adding these two layers in the neural network doesn't really hurt the neural network's ability to do as well as this simpler network without these two extra layers, because it's quite easy for it to learn the identity function to just copy `a[l]` to `a[l+2]` despite the addition of these two layers.
  * So adding two extra layers or adding this residual block to somewhere in the middle or the end of this big neural network doesn't hurt performance. It is easier to go from a decent baseline of not hurting performance and then gradient descent can only improve the solution from there.

**Networks in Networks and 1x1 Convolutions**

Paper: Network in Network

* At first, a 1×1 convolution does not seem to make much sense. After all, a convolution correlates adjacent pixels. A 1×1 convolution obviously does not.
* Because the minimum window is used, the 1×1 convolution loses the ability of larger convolutional layers to recognize patterns consisting of interactions among adjacent elements in the height and width dimensions. The only computation of the 1×1 convolution occurs on the channel dimension.
* The 1×1 convolutional layer is typically used to _adjust the number of channels_ between network layers and to control model complexity.

<figure><img src="../../.gitbook/assets/image (29).png" alt=""><figcaption></figcaption></figure>

**Inception Network Motivation**

Paper: Going Deeper with Convolutions

When designing a layer for a ConvNet, you might have to pick, do you want a 1 by 3 filter, or 3 by 3, or 5 by 5, or do you want a pooling layer? What the inception network does is it says, why shouldn't do them all? And this makes the network architecture more complicated, but it also works remarkably well.

<figure><img src="../../.gitbook/assets/image (30).png" alt=""><figcaption></figcaption></figure>

And the basic idea is that instead of you need to pick one of these filter sizes or pooling you want and commit to that, you can do them all and just concatenate all the outputs, and let the network learn whatever parameters it wants to use, whatever the combinations of these filter sizes it wants. Now it turns out that there is a problem with the inception layer as we've described it here, which is _computational cost_.

_The analysis of computational cost_:

<figure><img src="../../.gitbook/assets/image (31).png" alt=""><figcaption></figcaption></figure>

**Inception Network**

![](<../../.gitbook/assets/image (32).png>)\
![](<../../.gitbook/assets/image (33).png>)

**Data Augmentation**

Having more data will help all computer vision tasks.

_Some common data augmentation in computer vision_:

* Mirroring
* Random cropping
* Rotation
* Shearing
* Local warping

<figure><img src="../../.gitbook/assets/image (34).png" alt=""><figcaption></figcaption></figure>

#### Detection algorithms <a href="#user-content-detection-algorithms" id="user-content-detection-algorithms"></a>

**Object Localization**

\
![](<../../.gitbook/assets/image (35).png>)

* The classification and the classification of localization problems usually have one object.
* In the detection problem there can be multiple objects.
* The ideas you learn about image classification will be useful for classification with localization, and the ideas you learn for localization will be useful for detection.

\
![](<../../.gitbook/assets/image (36).png>)Giving the bounding box then you can use supervised learning to make your algorithm outputs not just a class label but also the four parameters to tell you where is the bounding box of the object you detected.

<figure><img src="../../.gitbook/assets/image (37).png" alt=""><figcaption></figcaption></figure>

The squared error is used just to simplify the description here. In practice you could probably use a log like feature loss for the `c1, c2, c3` to the softmax output.

**Landmark Detection**

In more general cases, you can have a neural network just output x and y coordinates of important points in image, sometimes called landmarks.

<figure><img src="../../.gitbook/assets/image (38).png" alt=""><figcaption></figcaption></figure>

If you are interested in people pose detection, you could also define a few key positions like the midpoint of the chest, the left shoulder, left elbow, the wrist, and so on.

The identity of landmark one must be consistent across different images like maybe landmark one is always this corner of the eye, landmark two is always this corner of the eye, landmark three, landmark four, and so on.

**Object Detection**

<figure><img src="../../.gitbook/assets/image (39).png" alt=""><figcaption></figcaption></figure>

able to localize the objects accurately within the image.

**Convolutional Implementation of Sliding Windows**

To build up towards the convolutional implementation of sliding windows let's first see how you can turn fully connected layers in neural network into convolutional layers.

<figure><img src="../../.gitbook/assets/image (40).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (41).png" alt=""><figcaption></figcaption></figure>

**Bounding Box Predictions (YOLO)**

The convolutional implementation of sliding windows is more computationally efficient, but it still has a problem of not quite outputting the most accurate bounding boxes. The perfect bounding box isn't even quite square, it's actually has a slightly wider rectangle or slightly horizontal aspect ratio.

![](<../../.gitbook/assets/image (42).png>)\
**YOLO algorithm**:

The basic idea is you're going to take the image classification and localization algorithm and apply that to each of the nine grid cells of the image. If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.

The advantage of this algorithm is that the neural network outputs precise bounding boxes as follows.

* First, this allows in your network to output bounding boxes of any aspect ratio, as well as, output much more precise coordinates than are just dictated by the stride size of your sliding windows classifier.
* Second, this is a convolutional implementation and you're not implementing this algorithm nine times on the 3 by 3 grid or 361 times on 19 by 19 grid.

**Intersection Over Union**

`IoU` is a measure of the overlap between two bounding boxes. If we use `IoU` in the output assessment step, then the higher the `IoU` the more accurate the bounding box. However `IoU` is a nice tool for the YOLO algorithm to discard redundant bounding boxes.

<figure><img src="../../.gitbook/assets/image (43).png" alt=""><figcaption></figcaption></figure>

**Non-max Suppression**

One of the problems of Object Detection as you've learned about this so far, is that your algorithm may find multiple detections of the same objects. Rather than detecting an object just once, it might detect it multiple times. Non-max suppression is a way for you to make sure that your algorithm detects each object only once.

* It first takes the largest `Pc` with the probability of a detection.
* Then, the non-max suppression part is to get rid of any other ones with a high (defined by a threshold) `IoU` between the box chosen in the first step.

<figure><img src="../../.gitbook/assets/image (44).png" alt=""><figcaption></figcaption></figure>

**Anchor Boxes**

One of the problems with object detection as you have seen it so far is that each of the grid cells can detect only one object. What if a grid cell wants to detect multiple objects? This is what the idea of anchor boxes does.

_Anchor box algorithm_:

| previous box                                                                                 | with two anchor boxes                                                                                                                       |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Each object in training image is assigned to grid cell that contains that object’s midpoint. | Each object in training image is assigned to grid cell that contains object’s midpoint and anchor box for the grid cell with highest `IoU`. |
| Output `y`: `3x3x8`                                                                          | Output `y`: `3x3x16` or `3x3x2x8`                                                                                                           |

\
![](<../../.gitbook/assets/image (45).png>)

**YOLO Algorithm**

_YOLO algorithm steps_:

* If you're using two anchor boxes, then for each of the nine grid cells, you get two predicted bounding boxes.
* Next, you then get rid of the low probability predictions.
* And then finally if you have three classes you're trying to detect, you're trying to detect pedestrians, cars and motorcycles. What you do is, for each of the three classes, independently run non-max suppression for the objects that were predicted to come from that class.



<figure><img src="../../.gitbook/assets/image (46).png" alt=""><figcaption></figcaption></figure>

#### Face Recognition <a href="#user-content-face-recognition" id="user-content-face-recognition"></a>

**What is face recognition**

* Verification
  * Input image, name/ID
  * Output whether the input image is that of the claimed person
* Recognition
  * Has a database of K persons
  * Get an input image
  * Output ID if the image is any of the K persons (or “not recognized”)

**One Shot Learning**

One-shot learning problem: to recognize a person given just one single image.

* So one approach is to input the image of the person, feed it too a ConvNet. And have it output a label, y, using a softmax unit with four outputs or maybe five outputs corresponding to each of these four persons or none of the above. However, this doesn't work well.
* Instead, to make this work, what you're going to do instead is learn a **similarity function** `d(img1,img2) = degree of difference between images`. So long as you can learn this function, which inputs a pair of images and tells you, basically, if they're the same person or different persons. Then if you have someone new join your team, you can add a fifth person to your database, and it just works fine.

**Siamese network**

A good way to implement a _similarity function_ `d(img1, img2)` is to use a Siamese network.![](<../../.gitbook/assets/image (47).png>)

_Goal of learning_:

* Parameters of NN define an encoding `𝑓(𝑥_𝑖)`
* Learn parameters so that:
  * If `𝑥_𝑖,𝑥_𝑗` are the same person, `‖f(𝑥_𝑖)−f(𝑥_𝑗)‖^2` is small.
  * If `𝑥_𝑖,𝑥_𝑗` are different persons, `‖f(𝑥_𝑖)−f(𝑥_𝑗)‖^2` is large.

**Triplet Loss**

One way to learn the parameters of the neural network so that it gives you a good encoding for your pictures of faces is to define an applied gradient descent on the triplet loss function.

In the terminology of the triplet loss, what you're going do is always look at one anchor image and then you want to distance between the anchor and the positive image, really a positive example, meaning as the same person to be similar. Whereas, you want the anchor when pairs are compared to the negative example for their distances to be much further apart. You'll always be looking at three images at a time:

* an anchor image (A)
* a positive image (P)
* a negative image (N)

As before we have `d(A,P)=‖f(A)−f(P)‖^2` and `d(A,N)=‖f(A)−f(N)‖^2`, the learning objective is to have `d(A,P) ≤ d(A,N)`. But if `f` always equals zero or `f` always outputs the same, i.e., the encoding for every image is identical, the objective is easily achieved, which is not what we want. So we need to add an `𝛼` to the left, a margin, which is a terminology you can see on support vector machines.

_The learning objective_:

`d(A,P) + 𝛼 ≤ d(A,N)` or `d(A,P) - d(A,N) + 𝛼 ≤ 0`

_Loss function_:

```
Given 3 images A,P,N:
L(A,P,N) = max(d(A,P) - d(A,N) + 𝛼, 0)
J = sum(L(A[i],P[i],N[i]))
```

<figure><img src="../../.gitbook/assets/image (48).png" alt=""><figcaption></figcaption></figure>

**Summary of Face Recognition**

_Key points to remember_:

* Face verification solves an easier 1:1 matching problem; face recognition addresses a harder 1:K matching problem.
* The triplet loss is an effective loss function for training a neural network to learn an encoding of a face image.
* The same encoding can be used for verification and recognition. Measuring distances between two images' encodings allows you to determine whether they are pictures of the same person.

_More references_:

* Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering
* Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). DeepFace: Closing the gap to human-level performance in face verification
* The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
* Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet

#### Neural Style Transfer <a href="#user-content-neural-style-transfer" id="user-content-neural-style-transfer"></a>

**What is neural style transfer**

Paper: A Neural Algorithm of Artistic Style

\
![](<../../.gitbook/assets/image (49).png>)

In order to implement Neural Style Transfer, you need to look at the features extracted by ConvNet at various layers, the shallow and the deeper layers of a ConvNet.

**What are deep ConvNets learning**

Paper: Visualizing and Understanding Convolutional Networks

<figure><img src="../../.gitbook/assets/image (50).png" alt=""><figcaption></figcaption></figure>

**Content Cost Function**

* Say you use hidden layer 𝑙 to compute content cost. (Usually, choose some layer in the middle, neither too shallow nor too deep)
* Use pre-trained ConvNet. (E.g., VGG network)
* Let `𝑎[𝑙](𝐶)` and `𝑎[𝑙](𝐺)` be the activation of layer 𝑙 on the images
* If `𝑎[𝑙](𝐶)` and `𝑎[𝑙](𝐺)` are similar, both images have similar content

```
J_content(C, G) = 1/2 * ‖𝑎[𝑙](𝐶)−𝑎[𝑙](𝐺)‖^2
```

**Style Cost Function**

Style is defined as correlation between activations across channels.

![](<../../.gitbook/assets/image (51).png>)\
![](<../../.gitbook/assets/image (52).png>)![](<../../.gitbook/assets/image (53).png>)

**1D and 3D Generalizations**

ConvNets can apply not just to 2D images but also to 1D data as well as to 3D data.

For 1D data, like ECG signal (electrocardiogram), it's a time series showing the voltage at each instant time. Maybe we have a 14 dimensional input. With 1D data applications, we actually use a recurrent neural network.

```
14 x 1 * 5 x 1 --> 10 x 16 (16 filters)
```

For 3D data, we can think the data has some height, some width, and then also some depth. For example, we want to apply a ConvNet to detect features in a 3D CT scan, for simplifying purpose, we have 14 x 14 x 14 input here.

```
14 x 14 x 14 x 1 * 5 x 5 x 5 x 1 --> 10 x 10 x 10 x 16 (16 filters)
```

Other 3D data can be movie data where the different slices could be different slices in time through a movie. We could use ConvNets to detect motion or people taking actions in movies.
