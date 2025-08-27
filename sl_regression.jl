using CSV
using GLM
using Plots
using TypedTables

data = CSV.File("data.csv")
X = data.size
Y = round.(Int, data.price / 1000)
t = Table(X = X, Y = Y)

gr(size = (600, 600))

p_scatter = scatter(X, Y, xlims = (0, 5000), ylims = (0, 800), xlabel = "Size (sqft)", ylabel = "Price (in thousands of dollars)", title = "Housing Prices in Portland", legend = false, color = :red)

ols = lm(@formula(Y ~ X), t)

plot!(X, predict(ols), color = :green, linewidth = 3)

newX = Table(X = [1250])

predict(ols, newX)

#initialize parameters

epochs = 0

theta_0 = 0.0

theta_1 = 0.0

p_scatter = scatter(X, Y, xlims = (0, 5000), ylims = (0, 800), xlabel = "Size (sqft)", ylabel = "Price (in thousands of dollars)", title = "Housing Prices in Portland", legend = false, color = :red)


h(x) = theta_0 .+ theta_1 * x

plot!(X, h(X), color = :blue, linewidth = 3)

m = length(X)

y_hat = h(X)

function cost(X, Y)
	(1 / (2 * m)) * sum((y_hat - Y) .^ 2)
end

J = cost(X, Y)

J_history = []

push!(J_history, J)

function pd_theta_0(X, Y)
	(1 / m) * sum(y_hat - Y)
end

function pd_theta_1(X, Y)
	(1 / m) * sum((y_hat - Y) .* X)
end

alpha_0 = 0.09
alpha_1 = 0.00000008
for i ∈ 1:100
	theta_0_temp = pd_theta_0(X, Y)
	theta_1_temp = pd_theta_1(X, Y)

	theta_0 -= alpha_0 * theta_0_temp
	theta_1 -= alpha_1 * theta_1_temp

	y_hat = h(X)

	J = cost(X, Y)

	push!(J_history, J)

	epochs += 1

	plot!(X, y_hat, color = :blue, alpha = 0.5, title = "Housing Prices in Portland (epochs = $epochs)")
end

p_scatter
