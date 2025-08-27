using Flux
using MLDatasets
using Images
using LinearAlgebra
using Random
using Statistics
using MLUtils

using Flux: onehotbatch

rng = MersenneTwister()

X_train_raw, y_train_raw = MLDatasets.MNIST(split=:train)[:]

X_test_raw, y_test_raw = MLDatasets.MNIST(split=:test)[:]

X_train = MLUtils.flatten(X_train_raw)

X_test = MLUtils.flatten(X_test_raw)

y_train = onehotbatch(y_train_raw, 0:9)

y_test = onehotbatch(y_test_raw, 0:9)

function init_params()
    n_samples, n_features = size(X_train)
    n_classes = size(y_train, 2)
    # Initialize weights and biases with random values
    # Ensure the weights and biases are matrices
    if n_features == 1
        W1 = rand(rng, Float64, (10, 1)) .- 0.5
    else
        W1 = rand(rng, Float64, (10, n_features)) .- 0.5
    end
    b1 = rand(rng, Float64, (10, 1)) .- 0.5
    W2 = rand(rng, Float64, (10, n_classes)) .- 0.5
    b2 = rand(rng, Float64, (10, 1)) .- 0.5
    # Ensure the weights and biases are matrices
    if size(W1, 2) == 1
        W1 = reshape(W1, size(W1, 1), 1)
    end
    if size(b1, 2) == 1
        b1 = reshape(b1, size(b1, 1), 1)
    end
    if size(W2, 2) == 1
        W2 = reshape(W2, size(W2, 1), 1)
    end
    if size(b2, 2) == 1
        b2 = reshape(b2, size(b2, 1), 1)
    end
    # Return the initialized weights and biases
    # Ensure the weights and biases are matrices
    if size(W1, 2) == 1
        return reshape(W1, size(W1, 1), 1), reshape(b1, size(b1, 1), 1), reshape(W2, size(W2, 1), 1), reshape(b2, size(b2, 1), 1)   
    end 
    return W1, b1, W2, b2
end

function ReLU(Z)
    return max.(0, Z)
end

function softmax(Z)
    Z = Z .- maximum(Z, dims=1)  # for numerical stability
    Z = Z ./ sum(exp.(Z), dims=1)  # normalize to get probabilities
    # Ensure the output is a matrix
    if size(Z, 2) == 1
        return reshape(Z, size(Z, 1), 1)
    end
    # Return the softmax probabilities
    return Z ./ sum(Z, dims=1)
end

function forward_prop(W1, b1, W2, b2, X)
    # Forward pass through the network
    Z1 = W1 ⋅ X .+ b1
    A1 = ReLU(Z1)
    Z2 = W2 ⋅ A1 .+ b2
    A2 = softmax(Z2)
    # Ensure the output is a matrix
    if size(A2, 2) == 1
        return Z1, A1, Z2, reshape(A2, size(A2, 1), 1)
    end
    # Return the activations
    return Z1, A1, Z2, A2
end

function d_ReLU(Z)
    # Derivative of ReLU
    Z = ReLU(Z)
    Z[Z .> 0] .= 1
    Z[Z .<= 0] .= 0
    # Ensure the output is a matrix
    if size(Z, 2) == 1
        return reshape(Z, size(Z, 1), 1)
    end
    # Return the derivative of ReLU 
    return Z .> 0
end

function back_prop(Z1, A1, Z2, A2, W2, X, Y)
    # Backward pass through the network
    # Ensure A2 is a matrix
    if size(A2, 2) == 1
        A2 = reshape(A2, size(A2, 1), 1)
    end
    # Ensure Y is a matrix
    if size(Y, 2) == 1
        Y = reshape(Y, size(Y, 1), 1)
    end
    # Calculate gradients   
    m = length(Y)
    one_hot_Y = Y
    dZ2 = A2 .- one_hot_Y
    dW2 = 1 / m * dZ2 ⋅ A1'
    db2 = 1 / m * sum(dZ2, dims = 2)
    dZ1 = W2' .* dZ2 * d_ReLU(Z1)
    dW1 = 1 / m * dZ1 .⋅ X'
    db1 = 1 / m * sum(dZ1, dims = 2)
    return dW1, db1, dW2, db2
end

function update_parms(W1, b1, W2, b2, dW1, db1, dW2, db2, α)
    W1 = W1 .- α * dW1
    b1 = b1 .- α * db1
    W2 = W2 .- α * dW2
    b2 = b2 .- α * db2
    return W1, b1, W2, b2
end

function get_predictions(A2)
    return argmax(A2, 0)
end

function get_accuracy(predictions, Y)
    # Ensure predictions and Y are vectors
    if size(predictions, 2) == 1
        predictions = vec(predictions)
    end
    if size(Y, 2) == 1
        Y = vec(Y)
    end
    # Calculate accuracy
    if length(predictions) != length(Y)
        error("Predictions and Y must have the same length.")
    end
    # Print predictions and Y for debugging
    println("Predictions: ", predictions)
    println("Y: ", Y)
    # Print the accuracy
    println("Accuracy: ", sum(predictions .== Y) / length(Y))
    # Return the accuracy
    return sum(predictions .== Y) / length(Y)
end

function gradient_descent(X, Y, iterations, α)
    # Initialize parameters
    if size(X, 2) == 1
        X = reshape(X, size(X, 1), 1)
    end
    if size(Y, 2) == 1
        Y = reshape(Y, size(Y, 1), 1)
    end
    n_samples, n_features = size(X)
    if size(Y, 2) == 1
        n_classes = 1
    else
        n_classes = size(Y, 2)
    end
    # Initialize weights and biases     
    W1, b1, W2, b2 = init_params()
    for i in 1:iterations
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_parms(W1, b1, W2, b2, dW1, db1, dW2, db2, α)
        if i % 50 == 0
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
        end
    end
    return W1, b1, W2, b2
end


W1, b2, W2, b2 = gradient_descent(X_train, y_train, 500, 0.1)
