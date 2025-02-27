import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
# from deprecated import deprecated

'''
Functions to look at that may be useful:
np.sum()
np.where()
np.maximum()
np.log()
np.exp()
np.argmax()
np.dot()
.append()
np.random.choice()

For torch tensors:
X.view()
X.numpy()
X.item()
dataset.targets
dataset.data
'''

# Load the FashionMNIST dataset 
def load_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])  # Only convert to tensor

    # Download the dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Subsampling: 50% from each class <- 0-9 class
    train_indices = subsample_50_percent_per_class(train_dataset)
    train_subset = Subset(train_dataset, train_indices)

    # --- TODO: Normalize data manually to the range [0, 1] ---
    # Normalize to [0, 1]
    # train_dataset.data = train_dataset.data/255.0
    # test_dataset.data = test_dataset.data/255.0

    # DataLoader for batching
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Function to perform subsampling 50% from each class
def subsample_50_percent_per_class(dataset):
    """
    Subsample 50% of the data from each class.
    dataset: The full dataset (e.g., FashionMNIST)
    Returns: A list of indices for the subsampled dataset
    """
    # --- TODO: Implement subsampling logic here ---
    class_indices = {label: [] for label in range(10)}

    # Populate the dictionary with indices for each class
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    ## Set the random seed for reproducibility
    np.random.seed(42)
    sampled_indices = []
    # Subsample indices for each class
    for label, indices in class_indices.items():
        # Determine the number of samples to select (50% of the class size)
        num_samples = len(indices) // 2
        # Randomly select the indices
        selected_indices = np.random.choice(indices, num_samples, replace=False)
        # Add the selected indices to the subsampled list
        sampled_indices.extend(selected_indices)

    return sampled_indices


# Forward pass for Fully Connected Layer
def fully_connected_forward(X, W, b):
    """
    Perform forward pass for a fully connected (linear) layer.
    X: Input data
    W: Weight matrix
    b: Bias vector
    """
    Z = X.dot(W) + b#TODO  # TODO: Compute the linear transformation (X * W + b)
    return Z

# Forward pass for ReLU activation
def relu_forward(Z):
    """
    ReLU activation function forward pass.
    Z: Linear output (input to ReLU)
    """
    A = np.maximum(Z, 0) # TODO: Apply ReLU function (element-wise)
    return A

# Forward pass for Softmax activation
def softmax_forward(Z):
    """
    Softmax activation function forward pass.
    Z: Output logits (before softmax)
    """
    Z -= np.max(Z, axis=1, keepdims=True)  # Stability trick
    exp_z = np.exp(Z)  # TODO: Apply softmax function (numerical stability)
    output = exp_z / np.nansum(exp_z, axis=1, keepdims=True)  # TODO: Normalize exp_z to get the softmax output
    return output

# Backward pass for Fully Connected Layer (Linear)
def fully_connected_backward(A, W, dZ):
    """
    NOTE CLARIFICATION HERE; dZ is input instead of Y, Z
    Compute gradients for the fully connected (linear) layer.
    A: Input data from the prior layer (either initial input X if first layer is prior layer, else activation A) (N x d)
    W: Weight matrix (dxK)
    dZ: Gradient of the loss with respect to Z (from the next layer)
    e.g. if current layer is 2, W is W2, dZ is dZ2, and input is A1 output of first layer
    """
    dW = A.T.dot(dZ)  # TODO: Compute gradient of weights (X^T * dZ)
    db = np.sum(dZ, axis = 0)  # TODO: Compute gradient of bias (sum of dZ)
    dA = dZ.dot(W.T)  # TODO: Compute gradient of loss with respect to input

    # Gradient Clipping
    # clip_value = 5
    # dW = np.clip(dW, -clip_value, clip_value)
    # db = np.clip(db, -clip_value, clip_value)
    return dW, db, dA

# Backward pass for ReLU activation
def relu_backward(Z, dA):
    """
    Compute the gradient for ReLU activation.
    Z: Input to ReLU (before activation)
    dA: Gradient of the loss with respect to activations (from the next layer)
    """
    dZ = np.where(Z > 0, dA, 0) # TODO: Compute dZ for ReLU (gradient is 0 for Z <= 0 and dA for Z > 0)
    return dZ

# Backward pass for Softmax Layer
def softmax_backward(S, Y):
    """
    NOTE THE CORRECTION/EFFICIENCY GAIN HERE in using softmax output instead of Z
    Compute the gradient of the loss with respect to softmax output.
    S: Output of softmax 
    Y: True labels (one-hot encoded)
    """
    dZ = (S-Y)*S + 1E-9  # TODO: Compute dZ for softmax (S - Y)
    return dZ

# Weight update function (gradient descent)
def update_weights(weights, biases, grads_W, grads_b, learning_rate=0.01):
    """
    --- TODO: Implement the weight update step ---
    weights: Current weights
    biases: Current biases
    grads_W: Gradient of the weights
    grads_b: Gradient of the biases
    learning_rate: Learning rate for gradient descent
    """
    weights = weights - learning_rate * grads_W
    biases = biases - learning_rate * grads_b
    return weights, biases


# Define the neural network 
def train(train_loader, test_loader, epochs=10000, learning_rate=0.01):
    # Initialize weights and biases
    input_dim = 784# TODO
    hidden_dim1 = 128   #could set differently
    hidden_dim2 = 64    #could set differently
    output_dim = 10# TODO
    
    # Initialize weights randomly
    # NOTE THE CORRECTION HERE! I HAD it done using torch but needs to be numpy
    # Note also that this is not using the specific methods I had mentioned for
    #   weight initialization (e.g. Xavier or He), this is just random
    W1 = np.random.randn(input_dim, hidden_dim1) * 0.01
    b1 = np.zeros(hidden_dim1)
    W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.01#TODO
    b2 = np.zeros(hidden_dim2)#TODO
    W3 = np.random.randn(hidden_dim2, output_dim) * 0.01#TODO
    b3 = np.zeros(output_dim)#TODO
    
    # ADD THESE to save training and test loss, accuracy
    training_loss = []
    test_loss = []
    training_accuracy = []
    test_accuracy = []
    
    # Loop through epochs
    for epoch in range(epochs):
        # print(epoch)
        epoch_loss = 0
        test_epoch_loss = 0
        correct_predictions = 0
        total_correct_predictions = 0
        total_samples = 0
        # print(f'W1: {np.isnan(W1).any()}')
        # print(f'W2: {np.isnan(W2).any()}')
        # print(f'W3: {np.isnan(W3).any()}')
        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            # print(batch_idx)
            # Flatten images to vectors
            X_batch = X_batch.flatten(1)#TODO  # Flatten images to vector
            Y_batch = torch.eye(output_dim)[Y_batch]  # Map label indices to corresponding one-hot encoded vectors
            
            # CONVERT TORCH TENSORS to numpy
            X = X_batch.numpy()# TODO
            y = Y_batch.numpy()# TODO

            # --- TODO: Implement the forward pass ---
            Z1 = fully_connected_forward(X, W1, b1)
            A1 = relu_forward(Z1)#TODO
            Z2 = fully_connected_forward(A1, W2, b2)#TODO
            A2 = relu_forward(Z2)#TODO
            Z3 = fully_connected_forward(A2, W3, b3)#TODO
            Y_pred = softmax_forward(Z3) #TODO
            
            # --- TODO: Implement loss computation --- MSE
            loss = np.power((y - Y_pred), 2)/2
            epoch_loss = epoch_loss + np.sum(loss) #TODO
            
            # print(np.nansum(loss))
            # --- TODO: Implement backward pass ---
            dZ3 = softmax_backward(Y_pred, y)#TODO
            dW3, db3, dA2 = fully_connected_backward(A2, W3, dZ3)#TODO
            dZ2 = relu_backward(Z2, dA2)#TODO
            dW2, db2, dA1 = fully_connected_backward(A1, W2, dZ2)#TODO
            dZ1 = relu_backward(Z1, dA1)#TODO
            dW1, db1, dX = fully_connected_backward(X, W1, dZ1)#TODO

            # --- TODO: Implement weight update ---
            W1, b1 = update_weights(W1, b1, dW1, db1, learning_rate)
            W2, b2 = update_weights(W2, b2, dW2, db2, learning_rate)
            W3, b3 = update_weights(W3, b3, dW3, db3, learning_rate)

            # Track accuracy
            correct_predictions = np.sum(np.argmax(Y_pred, axis=1) == np.argmax(y, axis=1))#for this batch; TODO
            total_correct_predictions = total_correct_predictions + correct_predictions#for the entire epoch; TODO
            total_samples = total_samples + y.shape[0]#for entire epoch; TODO
            # if np.nansum(loss) < 1:
            #     print(f'{batch_idx}: {loss}')
            #     break
        # Print out the progress - CLARIFIED
        train_accuracy = total_correct_predictions / total_samples
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}, Accuracy: {train_accuracy * 100}%")
        # print(f'Z1: {np.isnan(Z1).any()}')
        # print(f'Z2: {np.isnan(Z2).any()}')
        # print(f'Z3: {np.isnan(Z3).any()}')
        # print(f'A1: {np.isnan(A1).any()}')
        # print(f'A2: {np.isnan(A2).any()}')
        # print(f'Y_pred: {np.isnan(Y_pred).any()}')
        # Save the training loss and accuracy for each epoch to plot later
        training_loss.append(epoch_loss/len(train_loader))
        training_accuracy.append(train_accuracy)
        # TODO: For every 100 epochs, get the validation loss and error
        # FREQUENCY OF THIS IS CHANGED FROM EVERY 1000 to EVERY 100
        if np.isnan(epoch_loss).any():
            break
        if epoch % 100 == 0 and epoch > 0:
            # per = {epoch}/{epochs} * 100
            # print(f'{per:.0f}')
            test_epoch_loss = 0
            test_correct_predictions = 0
            test_total_samples = 0
            for X_test, Y_test in test_loader:
                # print(X_test)
                # Flatten test images
                X_test = X_test.flatten(1)
                Y_test = torch.eye(output_dim)[Y_test]

                # Convert to NumPy
                X_test = X_test.numpy()
                y_test = Y_test.numpy()

                # --- Forward Pass (No Backpropagation) ---
                Z1 = fully_connected_forward(X_test, W1, b1)
                A1 = relu_forward(Z1)
                Z2 = fully_connected_forward(A1, W2, b2)
                A2 = relu_forward(Z2)
                Z3 = fully_connected_forward(A2, W3, b3)
                Y_test_pred = softmax_forward(Z3)

                # --- Loss Calculation ---
                test_loss_batch = np.power((y_test - Y_test_pred), 2) / 2
                test_epoch_loss += np.sum(test_loss_batch)

                # --- Accuracy Calculation ---
                test_correct_predictions += np.sum(np.argmax(Y_test_pred, axis=1) == np.argmax(y_test, axis=1))
                test_total_samples += y_test.shape[0]

            # Compute test accuracy and store results
            test_epoch_accuracy = test_correct_predictions / test_total_samples
            # Save the test loss and accuracy for every 100th epoch to plot later
            test_loss.append(test_epoch_loss / len(test_loader))
            test_accuracy.append(test_epoch_accuracy)
            print(f"Test: Loss: {test_epoch_loss / len(test_loader)}, Accuracy: {test_epoch_accuracy * 100}%")

    print("Training complete!")    
    return training_loss, training_accuracy, test_loss, test_accuracy
    

# Display Images
def imshow(img, label):
    # Convert tensor to numpy array and transpose dimensions for plotting
    # img = img.numpy().transpose((1, 2, 0))
    img = img.numpy()[0]
    # Denormalize the image if it was normalized
    # img = np.clip(img * 0.5 + 0.5, 0, 1)
    plt.imshow(img)
    plt.title(f'Label: {label}')
    plt.show()


# Main function
def main():
    batch_size = 64
    train_loader, test_loader = load_data(batch_size)

    # random image
    train_iter = iter(train_loader)
    train_images, train_labels = next(train_iter)
    test_iter = iter(test_loader)
    test_images, test_labels = next(test_iter)
    random_idx_train = np.random.randint(0, len(train_images))
    # imshow(train_images[random_idx_train], train_labels[random_idx_train])
    random_idx_test = np.random.randint(0, len(test_images))
    # imshow(test_images[random_idx_test], test_labels[random_idx_test])


    # Start training
    training_loss, training_accuracy, test_loss, test_accuracy = train(train_loader, test_loader, epochs=10000, learning_rate=0.00001)
    
    # PLOT TRAINING LOSS AND TEST LOSS ON ONE SUBPLOT (epoch vs loss)
    # PLOT TRAINING ACCURACY AND TEST ACCURACY ON A SECOND SUBPLOT (epoch vs accuracy)
    
    epochs_train = list(range(1, len(training_loss) + 1))  # Epochs for training loss (1, 2, ..., N)
    epochs_test = list(range(100, (len(test_loss) + 1) * 100, 100))  # Epochs for test loss (100, 200, ..., N*100)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Training and Test Loss on the first subplot
    ax1.plot(..., ..., label='Training Loss', color='blue', marker='o')
    ax1.plot(..., ..., label='Test Loss', color='red', marker='x')
    ax1.set_title('Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot Training and Test Accuracy on the second subplot
    ax2.plot(..., ..., label='Training Accuracy', color='blue', marker='o')
    ax2.plot(..., ..., label='Test Accuracy', color='red', marker='x')
    ax2.set_title('Accuracy vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
