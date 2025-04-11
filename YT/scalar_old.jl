using ForwardDiff, DifferentialEquations, QuadGK, Interpolations, SpecialFunctions, LinearAlgebra

Mpl = 2.435e18; # reduced Planck mass in GeV
c = 299792458; # speed of light in m/s
KinGeV = (1.160e4)^-1 * 10^-9; # Kelvin in GeV
Mpcinm = 3.086e16 * 10^6; # Mpc in m
GeVinminv = 10^9/(1.973e-7); # GeV in m^-1
GeVinMpcinv = GeVinminv * Mpcinm; # GeV in Mpc^-1
MpcinvinHz = c/Mpcinm; # Mpc^inv in Hz
yrins = 365.2422 * 24 * 60 * 60; # yr in s

Or0h2 = 4.2e-5; # current radiation density parameter * h^2
Hubblh = 0.674; # normalised Hubble parameter h

T0 = 2.725*KinGeV; # current temperature
grho0 = 3.383; # current grho
gs0 = 3.931; # current gs


f = open("data/bg.dat","r");
body = readlines(f);
close(f)

strarray = [split(body[i],r"\s+",keepempty=false) for i=1:length(body)];
numarray = [parse.(Float64, strarray[i]) for i=1:length(strarray)];

lnetaList = [numarray[i][1] for i=1:length(numarray)];
anormList = [numarray[i][2] for i=1:length(numarray)];
calHList = [numarray[i][3] for i=1:length(numarray)];
EoSwList = [numarray[i][4] for i=1:length(numarray)];
cs2List = [numarray[i][5] for i=1:length(numarray)];
grhoList = [numarray[i][6] for i=1:length(numarray)];
gsList = [numarray[i][7] for i=1:length(numarray)];

lnetaspan = range(lnetaList[1], lnetaList[length(lnetaList)], length=length(lnetaList))
etai = 10^lnetaList[1];
etaf = 10^lnetaList[length(lnetaList)];

aCSI = CubicSplineInterpolation(lnetaspan, anormList);
aint(eta) = aCSI(log10(eta));

calHCSI = CubicSplineInterpolation(lnetaspan, calHList);
function calHint(eta) 
    if eta < etai
        return 1/eta
    else
        return calHCSI(log10(eta))
    end
end;

EoSwCSI = CubicSplineInterpolation(lnetaspan, EoSwList);
function EoSwint(eta) 
    if eta < etai
        return 1/3
    else
        return EoSwCSI(log10(eta))
    end
end;

cs2CSI = CubicSplineInterpolation(lnetaspan, cs2List);
function cs2int(eta) 
    if eta < etai
        return 1/3
    else
        return cs2CSI(log10(eta))
    end
end;

grhoCSI = CubicSplineInterpolation(lnetaspan, grhoList);
grhoint(eta) = grhoCSI(log10(eta));

gsCSI = CubicSplineInterpolation(lnetaspan, gsList);
gsint(eta) = gsCSI(log10(eta));


kmin = 1e4;
kmax = 1e9;
kList = [10^lnk for lnk=log10(kmin):0.01:log10(kmax)];

xi = 0.01;
xf = 400;
xList = [x for x=xi:0.01:xf];

PhiRD(x) = 9/x^2 * (sin(x/sqrt(3))/(x/sqrt(3)) - cos(x/sqrt(3))); # Phi in exact RD
PhipRD(x) = ForwardDiff.derivative(PhiRD,x);

wsk2(k,eta) = cs2int(eta) + 3*calHint(eta)^2/k^2*(cs2int(eta) - EoSwint(eta));

function scalarEoM(du,u,p,x) # u[1] = Phi, u[2] = dPhidx, p[1] = k
    eta = x/p[1]
    du[1] = u[2]
    du[2] = - 3*calHint(eta)/p[1]*(1 + cs2int(eta))*u[2] - wsk2(p[1],eta)*u[1]
end;

ui = [Float64(PhiRD(big(xi))),Float64(PhipRD(big(xi)))];
xspan = (xi,xf);

function scalarList()
    scalarList = [[] for i=1:length(kList)]
    Threads.@threads for i=1:length(kList)
        p = [kList[i]]
        scalarprob = ODEProblem(scalarEoM,ui,xspan,p)
        scalarsol = solve(scalarprob,Tsit5(),reltol=1e-10,abstol=1e-10)
        Phisol(x) = scalarsol(x)[1]
        Pisol(x) = scalarsol(x)[2]
        scalarList[i] = [p[1],Phisol,Pisol]
    end
    return scalarList
end;

println("Scalar EoM being solved...")
@time scalarsol = scalarList();

println("Data being exported...")
@time begin
    open("data/Phi.dat","w") do out
        Base.print_array(out, [scalarsol[j][2](xList[i]) for i=1:length(xList), j=1:length(kList)])
    end;

    open("data/Pi.dat","w") do out
        Base.print_array(out, [scalarsol[j][3](xList[i]) for i=1:length(xList), j=1:length(kList)])
    end
end;
