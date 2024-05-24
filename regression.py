import torch
from torch import nn


def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    Hint: use nn.Linear
    """
    model = nn.Linear(input_size, output_size)
    return model


def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    Hint: use the train_iteration function.
    Hint 2: while woring you can use the print function to print the loss every 1000 epochs.
    Hint 3: you can use the previos_loss variable to stop the training when the loss is not changing much.
    """

    learning_rate = 0.0001 # Using a small learning rate to avoid divergence (0.1 wasn't working for me)
    num_epochs = 10000 # The initial 100 was not enough to get a good model
    input_features = X.shape[1] # Instead of 0, we need to extract the number of features from the input `shape` of X
    output_features = y.shape[1] # Instead of 0, we need to extract the number of features from the input `shape` of y
    model = create_linear_regression_model(input_features, output_features)

    # Initialize the model weights and bias
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    loss_fn = nn.MSELoss() # Use mean squared error loss as requested by the professor

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")

    print('=== Starting the training the model ===')
    print(f'= Input features: {input_features}')
    print(f'= Learning rate: {learning_rate}')
    print(f'= Number of epochs: {num_epochs}')
    print(f'= Initial loss: {previous_loss}')

    for epoch in range(num_epochs):
        loss = train_iteration(X, y, model, loss_fn, optimizer)

        # Stop the training if the loss is not changing much (less than 1e-5)
        if abs(previous_loss - loss.item()) < 1e-5:
            print(f'[STOP] [{epoch}] Current loss: {loss.item()}. Previous loss: {previous_loss}. Stopping training.')
            break

        previous_loss = loss.item()

        # Let's print the loss every 500 epochs
        if epoch % 500 == 0:
            print(f'[{epoch}] Current loss: {loss.item()}')
    return model, loss

