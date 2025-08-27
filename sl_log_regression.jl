using CSV
using Plots

gr(size = (600, 600))

logistic(x) = 1.0 / (1.0 + exp(-x))

p_logistic = plot(-6:0.1:6, logistic, xlabel = "Inputs (x)", ylabel = "Outputs (y)", title = "Logistic (Sigmoid) Curve", legend = false,
	color = :blue)

theta_0 = 0.0

theta_1 = -0.5

z(x) = theta_0 .+ theta_1 * x

h(x) = 1.0 ./ (1.0 .+ exp.(-z(x)))

plot!(h, color = :red, linestyle = :dash)

data = CSV.File("wolfspider.csv")

X = data.feature

Y_temp = data.class

Y = []

for i in eachindex(Y_temp)
	if Y_temp[i] == "present"
		y = 1.0
	else
		y = 0.0
	end
	push!(Y, y)
end

p_data = scatter(X, Y, xlabel = "Size of Grains of Sand (mm)",
	ylabel = "Probability of observation (Absent = 0 | Present = 1)",
	title = "Wolf Spider Presence Classifier",
	legend = false,
	color = :red,
	markersize = 5)

theta_0 = 0.0

theta_1 = 5.0

t0_history = []
t1_history = []

push!(t0_history, theta_0)
push!(t1_history, theta_1)

z(x) = theta_0 .+ theta_1 * x

h(x) = 1.0 ./ (1.0 .+ exp.(-z(x)))

plot!(0:0.1:2, h, color = :green)

m = length(X)

y_hat = h(X)

function cost()
	(-1 / m) * sum(Y .* log.(y_hat) + (1 .- Y) .* log.(1 .- y_hat))
end

J = cost()

J_history = []

push!(J_history, J)

function pd_theta_0()
	sum(y_hat - Y)
end

function pd_theta_1()
	sum((y_hat - Y) .* X)
end

alpha = 0.01

epochs = 0


for i in 1:100

	theta_0_temp = pd_theta_0()
	theta_1_temp = pd_theta_1()

	theta_0 -= alpha * theta_0_temp
	theta_1 -= alpha * theta_1_temp

	push!(t0_history, theta_0)
	push!(t1_history, theta_1)

	y_hat = h(X)
	J = cost()

	push!(J_history, J)

	epochs += 1

	plot!(0:0.1:1.2, h, color = :blue, alpha = 0.025,
		title = "Wolf Spider Presence Classifier (epochs = $epochs)")

end

p_data

p_l_curve = plot(0:epochs, J_history)

p_params = scatter(t1_history, t0_history)