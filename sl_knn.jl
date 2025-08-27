using NearestNeighbors
using Plots
using Random


gr(size = (600, 600))

Random.seed!(getpid())

f1_train = rand(100)

f2_train = rand(100)

p_knn = scatter(f1_train, f2_train,
	xlabel = "Feature 1",
	ylabel = "Feature 2",
	title = "k-NN & k-D Tree Demo",
	legend = false,
	color = :blue)

X_train = [f1_train f2_train]

X_train_t = permutedims(X_train)

kdtree = KDTree(X_train_t)

k = 11

f1_test = rand()

f2_test = rand()

X_test = [f1_test, f2_test]

scatter!([f1_test], [f2_test],
	color = :red, markersize = 10)

index_knn, distances = knn(kdtree, X_test, k, true)

output = [index_knn distances]

#vscodedisplay(output)

f1_knn = [f1_train[i] for i in index_knn]

f2_knn = [f2_train[i] for i in index_knn]

scatter!(f1_knn, f2_knn,
	color = :yellow,
	markersize = 10, alpha = 0.5)

for i in 1:k
	plot!([f1_test, f1_knn[i]], [f2_test, f2_knn[i]],
		color = :green)
end

p_knn

#savefig(p_knn, "knn_concept_plot.svg")