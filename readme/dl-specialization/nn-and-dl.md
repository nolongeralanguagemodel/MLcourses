# NN and DL

#### Logistic Regression

In Logistic regression, we want to train the parameters `w` and `b`, we need to define a cost function.

![](../../.gitbook/assets/image.png), where ![](<../../.gitbook/assets/image (1).png>)\
Given ![](<../../.gitbook/assets/image (2).png>), we want ![](<../../.gitbook/assets/image (3).png>)

The loss function measures the discrepancy between the prediction (𝑦̂(𝑖)) and the desired output (𝑦(𝑖)). In other words, the loss function computes the error for a single training example.

<figure><img src="../../.gitbook/assets/image (4).png" alt=""><figcaption></figcaption></figure>

The cost function is the average of the loss function of the entire training set. We are going to find the parameters 𝑤 𝑎𝑛𝑑 𝑏 that minimize the overall cost function

<figure><img src="../../.gitbook/assets/image (5).png" alt=""><figcaption></figcaption></figure>

The loss function measures how well the model is doing on the single training example, whereas the cost function measures how well the parameters w and b are doing on the entire training set.
