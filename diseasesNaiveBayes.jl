using Random
using CSV
using DataFrames
using Statistics
using Distributions

rng = MersenneTwister()

raw_df = CSV.read("improved_disease_dataset.csv", DataFrame)

features = Matrix(float.(raw_df[:, 1:(end-1)]))
labels = string.(raw_df.disease)

unique_labels = unique(labels)
println("Unique labels: ", unique_labels)
println("Number of unique labels: ", length(unique_labels))

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
x_train = features[train_index, :]
x_test = features[test_index, :]

y_train = labels[train_index]
y_test = labels[test_index]

mutable struct GaussianNaiveBayes
	priors::Dict{Any, Float64}
	means::Dict{Any, Vector{Float64}}
	stds::Dict{Any, Vector{Float64}}
	features_dim::Int
	classes::Vector{Any}
end

function GaussianNaiveBayes()
	return GaussianNaiveBayes(Dict(), Dict(), Dict(), 0, [])
end

function fit!(model::GaussianNaiveBayes, X::AbstractMatrix{<:Real}, y::AbstractVector)
	n_samples, n_features = size(X)
	model.features_dim = n_features
	model.classes = unique(y)

	for class in model.classes
		class_indices = findall(==(class), y)
		X_class = X[class_indices, :]

		model.priors[class] = length(class_indices) / n_samples

		model.means[class] = [mean(X_class[:, i]) for i in 1:n_features]
		model.stds[class] = [std(X_class[:, i]) for i in 1:n_features]
	end
	return model
end

function predict_proba(model::GaussianNaiveBayes, X_test_sample::AbstractVector{<:Real})
	probabilities = Dict{Any, Float64}()

	for class in model.classes
		prior = model.priors[class]
		likelhood = 1.0

		for i in 1:model.features_dim
			mu = model.means[class][i]
			sigma = model.stds[class][i]
			if sigma == 0.0
				sigma = 1e-6
			end
			dist = Normal(mu, sigma)
			likelhood *= pdf(dist, X_test_sample[i])
		end
		probabilities[class] = prior * likelhood
	end

	sum_probs = sum(values(probabilities))
	if sum_probs > 0.0
		for class in keys(probabilities)
			probabilities[class] /= sum_probs
		end
	end
	return probabilities
end

function predict(model::GaussianNaiveBayes, X_test::AbstractMatrix{<:Real})
	predictions = Vector{Any}(undef, size(X_test, 1))
	for i in 1:size(X_test, 1)
		sample = X_test[i, :]
		probs = predict_proba(model, sample)
		predictions[i] = argmax(probs)
	end
	return predictions
end

# Gaussian Naive Bayes model
model_gnb = GaussianNaiveBayes()
fit!(model_gnb, x_train, y_train)

y_hat_gnb = predict(model_gnb, x_test)
accuracy_gnb = mean(y_hat_gnb .== y_test)
println("Gaussian Naive Bayes Accuracy: ", accuracy_gnb)