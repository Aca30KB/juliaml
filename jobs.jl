using CSV
using GLM
using Plots
using TypedTables
using Polynomials

n0 = 1
n1 = 2
n2 = 3
n3 = 4
n4 = 5
n5 = 6

data = CSV.File("jobs.csv")

X = data.Level
Y = data.Salary
t = Table(X=X, Y=Y)

gr(size=(600, 600))

p_scatter = scatter(X, Y)

function polyfit(X, Y, n)
    A = zeros(Float64, n + 1, n + 1)
    b = zeros(Float64, n + 1)


    Threads.@threads for i in 1:n+1
        Threads.@threads for j in 1:n+1
            A[i, j] = sum(X .^ (j - 1 + (i - 1)))
        end
    end

    Threads.@threads for i in 1:n+1
        b[i] = sum(Y .* X .^ (i - 1))
    end

    s = A \ b

    P = Polynomial(s)
    return P
end

Pol0 = polyfit(X, Y, n0)
Pol1 = polyfit(X, Y, n1)
Pol2 = polyfit(X, Y, n2)
Pol3 = polyfit(X, Y, n3)
Pol4 = polyfit(X, Y, n4)
Pol5 = polyfit(X, Y, n5)

plot!(Pol0, extrema(X)..., label="Linear Fit")
plot!(Pol1, extrema(X)..., label="Quadratic Fit")
plot!(Pol2, extrema(X)..., label="Qubic Fit")
plot!(Pol3, extrema(X)..., label="Quarter Fit")
plot!(Pol4, extrema(X)..., label="Fifth Fit")
plot!(Pol5, extrema(X)..., label="Sixth Fit")

Pol5(6.5)

GC.gc()
