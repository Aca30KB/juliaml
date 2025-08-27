using DecisionTree
using Random
using Statistics

#load data

X, y = load_data("iris")

X

y

X = float.(X)

y = string.(y)

iris = [X y]

vscodedisplay(iris)

function perclass_splits(y, percent)
	uniq_class = unique(y)
	keep_index = []
	for class in uniq_class
		class_index = findall(y .== class)
		row_index = randsubseq(class_index, percent)
		push!(keep_index, row_index...)
	end
	return keep_index
end

#split data between train and test

Random.seed!(getpid())

train_index = perclass_splits(y, 0.67)

test_index = setdiff(1:length(y), train_index)

# split features

X_train = X[train_index, :]

X_test = X[test_index, :]

# split classes

y_train = y[train_index]

y_test = y[test_index]

# Decision Tree 

#run model

model = DecisionTreeClassifier(max_depth = 2)

DecisionTree.fit!(model, X_train, y_train)

#print tree 

print_tree(model)

# view training data 

train = [X_train y_train]

vscodedisplay(train)

# view decision node data subset

train_R = train[train[:, 4].>0.8, :]

vscodedisplay(train_R)

#make predictions

y_hat = DecisionTree.predict(model, X_test)

# check accuracy

accuracy = mean(y_hat .== y_test)

# display confusion matrix

DecisionTree.confusion_matrix(y_test, y_hat)

# display results

check = [y_hat[i] == y_test[i] for i in eachindex(y_hat)]

chek_display = [y_hat y_test check]

vscodedisplay(chek_display)

prob = DecisionTree.predict_proba(model, X_test)

vscodedisplay(prob)

# Random forest

#run model

model = RandomForestClassifier(n_trees = 20)

fit!(model, X_train, y_train)

#make predictions

y_hat = predict(model, X_test)

#check accuracy

accuracy = mean(y_hat .== y_test)

# display confusion matrix

DecisionTree.confusion_matrix(y_test, y_hat)

# display results

check = [y_hat[i] == y_test[i] for i in eachindex(y_hat)]

check_display = [y_hat y_test check]

vscodedisplay(check_display)

# display probability of each prediction

prob = DecisionTree.predict_proba(model, X_test)

vscodedisplay(prob)