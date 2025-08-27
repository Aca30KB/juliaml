using CSV
using Random
using Flux
using DataFrames
using LinearAlgebra
using Statistics
using Flux: logitcrossentropy, Chain, Dense, softmax
using Flux: DataLoader
using MLUtils

using Flux: crossentropy, onecold, onehotbatch, train!


rng = MersenneTwister()

raw_df = CSV.read("improved_disease_dataset.csv", DataFrame)
features = Matrix(Int64.(raw_df[:, 1:(end-1)]))
labels = string.(raw_df.disease)
class_names = sort(unique(labels))

function perclass_splits(y, percent)
	unique_classes = unique(y)
	keep_index = []
	for class in unique_classes
		class_index = findall(y .== class)
		row_index = randsubseq(rng, class_index, percent)
		# Ensure the row_index is a vector
		if typeof(row_index) == Integer
			row_index = [row_index]
		end
		# Append the selected indices to keep_index
		if !isempty(row_index)
			row_index = collect(row_index)  # Ensure it's a vector
		end
		# Append the indices to keep_index
		push!(keep_index, row_index...)
	end
	return keep_index
end

train_index = perclass_splits(labels, 0.8)
test_index = setdiff(1:length(labels), train_index)
# split features
X_train_raw = features[train_index, :]
X_test_raw = features[test_index, :]
y_train_raw = labels[train_index]
y_test_raw = labels[test_index]

X_train = Flux.flatten(X_train_raw)
X_test = Flux.flatten(X_test_raw)
y_train = onehotbatch(y_train_raw, unique(labels))
y_test = onehotbatch(y_test_raw, unique(labels))

num_sample, num_features = size(X_train)
num_classes = length(unique(labels))
# Reshape the training and testing data
X_train = reshape(X_train, num_features, num_sample)
X_test = reshape(X_test, num_features, length(y_test_raw))
# Ensure the labels are in the correct format
y_train = reshape(y_train, num_classes, num_sample)
y_test = reshape(y_test, num_classes, length(y_test_raw))


# Define the model
model = Chain(
	Dense(num_features, 256, relu),
	Dense(256, 128, relu),
	Dense(128, 64, relu),
	Dense(64, 32, relu),
	Dense(32, num_classes),
	softmax,
)

loss(x, y) = Flux.logitcrossentropy(model(x), y)

optimizer = Flux.ADAM(0.001)
opt = Flux.setup(optimizer, model)

# Set the learning rate and loss history
learning_rate = 0.05

parameters = Flux.trainable(model)

loss_history = Float64[]

data = MLUtils.DataLoader((X_train, y_train), batchsize = 32, shuffle = true)


epochs = 500
# Train the model
for epoch in 1:epochs
	for (x, y) in data
		loss_value, grad = Flux.withgradient(model) do m
			y_hat = m(x)
			Flux.logitcrossentropy(y_hat, y)
		end
		Flux.update!(opt, model, grad[1])
		push!(loss_history, loss_value)
	end
	if epoch % 10 == 0
		current_loss = loss(X_train, y_train)
		println("Epoch: $epoch, Loss: $current_loss")
	end
end

y_hat = softmax(model(X_test))

y_hat = onecold(y_hat)

y = onecold(y_test)


accuracy = mean(y_hat .== y)
index = collect(1:length(y))
check = [y_hat[i] == y[i] for i in eachindex(y)]

println("Accuracy: ", mean(check))
println("Number of correct predictions: ", sum(check))
println("Number of total predictions: ", length(check))
println("Indices of correct predictions: ", index[check])
println("Indices of incorrect predictions: ", index[.!check])
println("Unique labels: ", unique(labels))
println("Number of unique labels: ", length(unique(labels)))
println("Loss history: ", loss_history)
println("Final model parameters: ", Flux.trainable(model))
println("Final model structure: ", model)
println("Training data shape: ", size(X_train))
println("Testing data shape: ", size(X_test))
println("Training labels shape: ", size(y_train))
println("Testing labels shape: ", size(y_test))