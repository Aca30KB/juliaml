using Flux
using Images
using MLDatasets
using Plots

using Flux: crossentropy, onecold, onehotbatch, train!

using LinearAlgebra
using Random
using Statistics

rng = MersenneTwister()

X_train_raw, y_train_raw = MLDatasets.MNIST(split = :train)[:]

X_test_raw, y_test_raw = MLDatasets.MNIST(split = :test)[:]

X_train_raw

index = 1

img = X_train_raw[:, :, index]

colorview(Gray, img')

y_train_raw

y_train_raw[index]

img = X_test_raw[:, :, index]

colorview(Gray, img')

y_test_raw

y_test_raw[index]

X_train = Flux.flatten(X_train_raw)

X_test = Flux.flatten(X_test_raw)

y_train = onehotbatch(y_train_raw, 0:9)

y_test = onehotbatch(y_test_raw, 0:9)

model = Chain(
	Dense(28 * 28, 32, relu),
	Dense(32, 10),
	softmax,
)

loss(x, y) = crossentropy(model(x), y)

#track parameters

ps = Flux.trainable(model)

#select optimizer

learning_rate = 0.05

opt = Flux.setup(Adam(0.01), model)

#trian model

loss_history = []

epochs = 50

data = Flux.DataLoader((X_train, y_train), batchsize = 64, shuffle = true)

for epoch in 1:epochs
	#train model
	for (x, y) in data
		loss_value, grad = Flux.withgradient(model) do m
			y_hat = m(x)
			crossentropy(y_hat, y)
		end
		Flux.update!(opt, model, grad[1])
		push!(loss_history, loss_value)
	end
	if epoch % 10 == 0
		current_loss = loss(X_train, y_train)	
		println("Epoch = $epoch: Training Loss = $current_loss")
	end
end

#make predicitions

y_hat_raw = model(X_test)

y_hat = onecold(y_hat_raw)

y = y_test_raw

mean(y_hat .== y)

check = [y_hat[i] == y[i] for i in eachindex(y)]

index = collect(1:length(y))

check_display = [index y_hat y check]

vscodedisplay(check_display)

gr(size = (600, 600))

p_l_curve = plot(1:epochs, loss_history,
	xlabel = "Epochs",
	ylabel = "Loss",
	title = "Learning Curve",
	legend = false,
	color = :blue,
	linewidth = 2,
)
