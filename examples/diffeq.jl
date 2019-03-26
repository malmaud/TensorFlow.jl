using DifferentialEquations

f(u,p,t)=1.01 .* u

u0=constant(0.5)
tspan=(0.0,1.0)
prob=ODEProblem(f, u0, tspan)
s=solve(prob)
