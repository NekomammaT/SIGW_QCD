using ForwardDiff, Plots, LaTeXStrings, DifferentialEquations, Roots, QuadGK, HCubature, MultiQuad, Interpolations, Dierckx, SpecialFunctions, LinearAlgebra, DelimitedFiles

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

println("# of threads : ", Threads.nthreads())


bgdata = readdlm("data/bg.csv", ',');

etaList = bgdata[:,1];
anormList = bgdata[:,2];
calHList = bgdata[:,3];
EoSwList = bgdata[:,4];
cs2List = bgdata[:,5];
grhoList = bgdata[:,6];
gsList = bgdata[:,7];

lnetaList = log10.(etaList);
etai = etaList[1];
etaf = etaList[length(etaList)];

alneta = Spline1D(lnetaList, anormList, k=3);
aint(eta) = alneta(log10(eta))
calHlneta = Spline1D(lnetaList, calHList, k=3);
calHint(eta) = calHlneta(log10(eta));
EoSwlneta = Spline1D(lnetaList, EoSwList, k=3);
EoSwint(eta) = EoSwlneta(log10(eta))
cs2lneta = Spline1D(lnetaList, cs2List, k=3);
cs2int(eta) = cs2lneta(log10(eta));
grholneta = Spline1D(lnetaList, grhoList, k=3);
grhoint(eta) = grholneta(log10(eta))
gslneta = Spline1D(lnetaList, gsList, k=3);
gsint(eta) = gslneta(log10(eta));


kmin = 1e4;
kmax = 1e9;
kList = [10^lnk for lnk=log10(kmin):0.01:log10(kmax)];
lnkList = log10.(kList);

xi = 1e-2;
xf = 400;

function gkEoM(du,u,p,x) # u[i] = g, u[i+length(kList)] = dgdx, p[i] = k
    for i=1:length(kList)
        du[i] = u[i+length(kList)]
        du[i+length(kList)] = - (1 - (1-3*EoSwint(x/p[i]))/2*calHint(x/p[i])^2/p[i]^2)*u[i]
    end
end;

xspan = (xi, xf);

g1RD(x) = cos(x);
g1pRD(x) = -sin(x);
g2RD(x) = sin(x);
g2pRD(x) = cos(x);

ui1 = vcat([g1RD(xi) for i=1:length(kList)], [g1pRD(xi) for i=1:length(kList)]);
ui2 = vcat([g2RD(xi) for i=1:length(kList)], [g2pRD(xi) for i=1:length(kList)]);

g1prob = ODEProblem(gkEoM,ui1,xspan,kList);
g2prob = ODEProblem(gkEoM,ui2,xspan,kList);

println("Tensor EoM begin solved...")
@time g1sol = solve(g1prob,Tsit5(),reltol=1e-10,abstol=1e-10);
@time g2sol = solve(g2prob,Tsit5(),reltol=1e-10,abstol=1e-10);

writedlm("data/g1.csv", vcat(hcat(g1sol.t...),hcat(g1sol.u...)), ',');
writedlm("data/g2.csv", vcat(hcat(g2sol.t...),hcat(g2sol.u...)), ',');

hkRD(x) = sin(x)/x;
hpRD(x) = ForwardDiff.derivative(hkRD,x);

function hkEoM(du,u,p,x) # u[i] = h, u[i+length(kList)] = dhdx, p[i] = k
    for i=1:length(kList)
        du[i] = u[i+length(kList)]
        du[i+length(kList)] = -2*calHint(x/p[i])/p[i]*u[i+length(kList)] - u[i]
    end
end;

xspan = (xi, xf);
ui = vcat([hkRD(xi) for i=1:length(kList)], [hpRD(xi) for i=1:length(kList)]);

hkprob = ODEProblem(hkEoM,ui,xspan,kList);

println("Linear tensor begin solved...")
@time hksol = solve(hkprob,Tsit5(),reltol=1e-10,abstol=1e-10);

writedlm("data/hk.csv", vcat(hcat(hksol.t...),hcat(hksol.u...)), ',');

g1data = readdlm("data/g1.csv",',');
x1List = g1data[1,:];
g1List = g1data[2:1+length(kList),:];
g1pList = g1data[2+length(kList):size(g1data)[1],:];

g2data = readdlm("data/g2.csv",',');
x2List = g2data[1,:];
g2List = g2data[2:1+length(kList),:];
g2pList = g2data[2+length(kList):size(g2data)[1],:];

g1CSI = Spline2D(lnkList,x1List,g1List,kx=3,ky=3);
g1pCSI = Spline2D(lnkList,x1List,g1pList,kx=3,ky=3);
g2CSI = Spline2D(lnkList,x2List,g2List,kx=3,ky=3);
g2pCSI = Spline2D(lnkList,x2List,g2pList,kx=3,ky=3);

g1int(k,x) = g1CSI(log10(k),x);
g1pint(k,x) = g1pCSI(log10(k),x);
g2int(k,x) = g2CSI(log10(k),x);
g2pint(k,x) = g2pCSI(log10(k),x);

normNk(k,x) = g1pint(k,x)g2int(k,x)-g2pint(k,x)g1int(k,x);

dx = 2*2*π;

function g1g1bar(k)
    f(x) = g1int(k,x)*g1int(k,x)
    return quadgk(f,xf-dx,xf)[1]/dx
end;

function g2g2bar(k)
    f(x) = g2int(k,x)*g2int(k,x)
    return quadgk(f,xf-dx,xf)[1]/dx
end;

function g1g2bar(k)
    f(x) = g1int(k,x)*g2int(k,x)
    return quadgk(f,xf-dx,xf)[1]/dx
end;

hkdata = readdlm("data/hk.csv",',');
xList = hkdata[1,:];
hkList = hkdata[2:1+length(kList),:];
hpList = hkdata[2+length(kList):size(hkdata)[1],:];

hkCSI = Spline2D(lnkList,xList,hkList,kx=3,ky=3);
hpCSI = Spline2D(lnkList,xList,hpList,kx=3,ky=3);

hint(k,x) = hkCSI(log10(k),x);
hpint(k,x) = hpCSI(log10(k),x);

function h2bar(k)
    f(x) = hint(k,x)*hint(k,x)
    return quadgk(f,xf-dx,xf)[1]/dx
end;

hRD(x) = sin(x)/x;
h2intRD(x) = hRD(x)*hRD(x);
h2barRD = quadgk(h2intRD,xf-dx,xf)[1]/dx;

g1g1List = [g1g1bar(kList[i]) for i=1:length(kList)];
g1g2List = [g1g2bar(kList[i]) for i=1:length(kList)];
g2g2List = [g2g2bar(kList[i]) for i=1:length(kList)];
coeffList = [(aint(xf/kList[i])*calHint(xf/kList[i])/aint(etaf)/calHint(etaf))^2 /24*(kList[i]/calHint(xf/kList[i]))^2 for i=1:length(kList)];
OlinList = [(aint(xf/kList[i])*calHint(xf/kList[i])/aint(etaf)/calHint(etaf))^2 /12*(kList[i]/calHint(xf/kList[i]))^2 * h2bar(kList[i]) for i=1:length(kList)];

writedlm("data/g1g2coeff.csv", hcat(log10.(kList),g1g1List,g1g2List,g2g2List,coeffList,OlinList), ',');

println("Completed.")
