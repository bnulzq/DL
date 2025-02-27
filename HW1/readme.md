We are going to write a neural network to classify FashionMNIST (or MNIST) images (10 classes of clothing or digits). You can only use basic Python, numpy, 
and matplotlib. One exception: You can use pytorch to download the data and to load it into the training loop.

## (a) Download the FashionMNIST (or MNIST) dataset (this is in the python code provided):
train_dataset = datasets.FashionMNIST(root=’./data’, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root=’./data’, train=False, download=True, transform=transform)

## (b) Plot 8-10 random images from dataset that you used, and their labels.
Are the labels correct? (Some human labeled them, who may have been having a bad day.)

## (c) Code a neural network to classify these images:

- In either MNIST dataset, there are classes from 0–9. Which loss function might you choose?
- Keep ≈50 percent of each class for both training and validation datasets. Write code to subsample these. Remember to also subsample the corresponding labels.
- You can try with a different percentage and see how the results compare (how many samples do you need to effectively train this model?) but you are not required to. Instead of exactly 50 percent of each class, you can also try something like randomizing the
order of the samples and keeping the first half; you want enough samples from each class for it to train effectively.
- Each image is 28x28, for a total of 784 pixels. Each of these pixels has one
value/feature; these are not RGB which would have 3 values per pixel.
Normalize the data to get all pixels in the range [0, 1]. You could also
try to normalize to [-1,1] instead if you wanted to.
- Use subplots to plot 8-10 images after the subsampling and normalization
to make sure that labels are still correct and images are as expected (e.g.
you didn’t mess up the order of the labels wrt the images, or mess up
the images themselves in some way in your pre-processing, always good
to check).

## (d) Create a FashionMNIST or MNIST dataset using PyTorch: Code is in the skeleton python code for this. They are batched with a batch size of 64
(you can try changing the batch size after you have the code running to see
what the impacts are).

## (e) Start your implementation by writing a fully-connected layer
First, compute the linear transformation (the feed forward part):
zl = W al−1 + b
where:
• zl ∈ Rm is the output before the activation function is applied.
• al−1 ∈ Rd is the input to the layer (the outputs or activations from the
previous layer l − 1).
• W ∈ Rm×d is the weight matrix, where:
– d is the number of input features.
– m is the number of output features (neurons in layer l).
• b ∈ Rm is the bias vector.

## (f) Then, apply the activation function (e.g., ReLU):
al = ReLU(zl) = max(0, zl)
where al is the output of the layer after the activation.
Use matrix multiplication for the forward computations, using ‘@‘ or np.matmul
(https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) for speed.

## (g) The fully connected layer also needs a backward pass function (see the
example at the end of this document, and see Ch. 8 of Bishop, Deep Learning):
– Hint: Start by computing ∂L
This step will depend on the specific loss function you are using.
- Gradient of Loss w.r.t Weights W l: After computing the gradient of
the loss with respect to the output, you’ll propagate the error backward
to the weights W l. The gradient of the loss with respect to the weights is
obtained by applying the chain rule to the linear transformation. Using
the chain rule:
∂zl is the gradient of the loss with respect to zl, the output of the
linear transformation,
– (al−1)T is the transpose of the activations from the previous layer
al−1. Recall that zl = W lal−1 + bl. If you take the derivative of zl
wrt W l, what is the result? al−1
- Gradient of Loss w.r.t Bias bl:
Once again apply the chain rule:
where σ is the ReLU function; the derivative is either 1 or 0. Also, since:
- Recursion of the Gradient for Deeper Layers: To compute the gra-
dients for deeper layers, you use the chain rule recursively. For example,
This involves the weights W l−1 of the previous layer and the gradient
– Hint: The term ∂zl
∂al−1 is simply the weights W l of the next layer.
So, this step involves propagating the error from the current layer
back to the previous layer:
Once you’ve computed this, you can recursively apply the same steps
for earlier layers.

## (h) Implement a ReLU unit where al = max(0, zl). 
This needs both a forward
function and a backward function.

## (i) Implement the softmax function for probabilities and cross-entropy loss,
both forward and backward.
- Softmax Equation:
where C is the number of classes. Note that there is no linear transfor-
mation for the final layer; the outputs from the prior layer go directly
into softmax.
- Cross-Entropy Loss (CCE):
where:
– al
k(xi) is the softmax probability for class k given input xi,
– 1yi=k is an indicator function, equal to 1 if the label for input xi is
k, otherwise 0,
– B is the batch size.
EAS 6995, SPRING 2025 HOMEWORK 1 7
- Error Calculation:
This computes the average number of misclassifications in the batch,
which is used for monitoring training progress.

## (j) Test your forward and backward operators. The forward operator can
be tested with simple inputs. The backward operator can be tested using
numerical differentiation:
- Gradient Checking for Backpropagation: To test if backpropaga-
tion is working correctly in a neural network coded from scratch, we can
use gradient checking. This method compares the gradients computed
by backpropagation with those computed using numerical differentiation,
e.g.:
https://medium.com/farmart-blog/understanding-backpropagation-and-gradient-checking-6a5c0ba73a68
- Numerical Gradient Calculation: Compute the numerical gradients
using finite differences. For a parameter θ, the numerical gradient is
calculated as:
where L(θ) is the loss function and ε is a small value, typically 10−4.
- Compare Gradients: Compare the gradients obtained from backprop-
agation with the numerical gradients. The difference should be very small
(typically less than 10−5) if the implementation is correct.

## (k) Train your neural network. Tune (e.g., change and see how your loss and
error change) the hyperparameters of the number of iterations and learning
rate. You can also test number of nodes in each layer, and batch size, and/or
add more layers, etc.
## (l) Test your model by passing the images from the validation dataset
through the forward pass every 500 iterations. Compute the loss and
error for these iterations.
## (m) Plot training loss and training error as a function of the number of weight
updates of the neural network you trained. Describe what you see.
## (n) Plot the test loss and test error for every 500th weight update. How does
this compare to training loss and error? Is it what you would expect, and why
or why not?
## (o) Plot 8 images that were labeled correctly, and 8 that were labeled incorrectly.
Do you have any observations?
