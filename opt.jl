using JuMP
using GLPK

model = Model(GLPK.Optimizer)

@variable(model, x >= 0)
@variable(model, y >= 0)

@constraint(model, 6x + 8y >= 100)
@constraint(model, 7x + 12y >= 120)
@objective(model, Min, 12x + 20y)

optimize!(model)

@show value(x)
@show value(y)

@show objective_value(model)

# Knapsak problem binary

model = Model(GLPK.Optimizer)

@variable(model, green, Bin)
@variable(model, blue, Bin)
@variable(model, orange, Bin)
@variable(model, yellow, Bin)
@variable(model, gray, Bin)

@constraint(model, weight, green * 12 + blue * 2 + orange * 1 + yellow * 4 + gray * 1 <= 15)

@objective(model, Max, green * 4 + blue * 2 + orange * 1 + yellow * 10 + gray * 2)

optimize!(model)

boxes = [green, blue, yellow, orange, gray]

for box in boxes
	println(box, "\t = ", value(box))
end

value(weight)

objective_value(model)



# Knapsak problem int

model = Model(GLPK.Optimizer)

@variable(model, green >= 0, Int)
@variable(model, blue >= 0, Int)
@variable(model, orange >= 0, Int)
@variable(model, yellow >= 0, Int)
@variable(model, gray >= 0, Int)

@constraint(model, weight, green * 12 + blue * 2 + orange * 1 + yellow * 4 + gray * 1 <= 15)

@objective(model, Max, green * 4 + blue * 2 + orange * 1 + yellow * 10 + gray * 2)

optimize!(model)

boxes = [green, blue, yellow, orange, gray]

for box in boxes
	println(box, "\t = ", value(box))
end

value(weight)

objective_value(model)


#Nonlinear

using Ipopt

model = Model(Ipopt.Optimizer)

@variable(model, x >= 0, start = 0)
@variable(model, y >= 0, start = 0)

@NLconstraint(model, 2x + 2y == 100)

@NLobjective(model, Max, x * y)

optimize!(model)

value(x)

value(y)

objective_value(model)

using Plots

plotlyjs(size = (760, 570))

x = 0:50
area(x) = x * (100 - 2x) / 2

p = plot(x, area,
	title = "Maximize area",
	xlabel = "Length of x (feet)",
	ylabel = "Area (square feet)")
