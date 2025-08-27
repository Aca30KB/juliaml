using DecisionTree
using Statistics
using CSV
using TextAnalysis
using DataFrames
using MLDataUtils
using Random

raw_df = CSV.read("emails.csv", DataFrame)

all_words = names(raw_df)[2:end-1]
all_words_text = join(all_words, " ")
document = StringDocument(all_words_text)

prepare!(document, strip_articles)
prepare!(document, strip_pronouns)

vocabulary = split(TextAnalysis.text(document))
clean_words_df = raw_df[!, vocabulary]
data_matrix = Matrix(clean_words_df)

labels = raw_df.Prediction

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

rng = MersenneTwister()

train_index = perclass_splits(labels, 0.67)

test_index = setdiff(1:length(labels), train_index)

# split features

x_train = data_matrix[train_index, :]

x_test = data_matrix[test_index, :]

# split classes

y_train = labels[train_index]

y_test = labels[test_index]

# Decision Tree 

#run model

model = DecisionTreeClassifier(max_depth = 2)

DecisionTree.fit!(model, x_train, y_train)

#print tree 

print_tree(model)

# view training data 

train = [x_train y_train]

# view decision node data subset

train_R = train[train[:, 4].>0.8, :]

#make predictions

y_hat = DecisionTree.predict(model, x_test)

# check accuracy

accuracy = mean(y_hat .== y_test)

# display confusion matrix

DecisionTree.confusion_matrix(y_test, y_hat)

# display results

check = [y_hat[i] == y_test[i] for i in eachindex(y_hat)]

chek_display = [y_hat y_test check]

prob = DecisionTree.predict_proba(model, x_test)

# Random forest

#run model

model = RandomForestClassifier(n_trees = 20)

DecisionTree.fit!(model, x_train, y_train)

#make predictions

#y_hat = MLDataUtils.predict(model, x_test)

y_hat = DecisionTree.predict(model, x_test)

#check accuracy

accuracy = mean(y_hat .== y_test)

# display confusion matrix

DecisionTree.confusion_matrix(y_test, y_hat)

# display results

check = [y_hat[i] == y_test[i] for i in eachindex(y_hat)]

check_display = [y_hat y_test check]

# display probability of each prediction

prob = DecisionTree.predict_proba(model, x_test)