using CSV
using DataFrames
using Random
using DecisionTree
using Statistics
using MLDataUtils

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
		row_index = rand(class_index, round(Int, length(class_index) * percent))
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
y_test =  labels[test_index]

# Decision tree model

model_dt = DecisionTreeClassifier(max_depth = 3, rng = rng)
DecisionTree.fit!(model_dt, x_train, y_train)
print_tree(model_dt)

y_hat = DecisionTree.predict(model_dt, x_test)

accuracy = mean(y_hat .== y_test)
println("Accuracy: ", accuracy)

DecisionTree.confusion_matrix(y_test, y_hat)

# Random forest model

model_rf = RandomForestClassifier(n_trees = 100, max_depth = 3, rng = rng)

DecisionTree.fit!(model_rf, x_train, y_train)

y_hat_rf = DecisionTree.predict(model_rf, x_test)
accuracy_rf = mean(y_hat_rf .== y_test)
println("Random Forest Accuracy: ", accuracy_rf)