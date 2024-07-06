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

xi = 1e-2;
xf = 400;
xList = [x for x=xi:0.01:xf];

function gkEoM(du,u,p,x) # u[1] = g, u[2] = dgdx, p[1] = k
    du[1] = u[2]
    du[2] = - (1 - (1-3*EoSwint(x/p[1]))/2*calHint(x/p[1])^2/p[1]^2)*u[1]
end;

xspan = (xi, xf);

function gkList(ui)
    gkList = [[] for i=1:length(kList)]
    Threads.@threads for i=1:length(kList)
        p = [kList[i]]
        tensorprob = ODEProblem(gkEoM,ui,xspan,p)
        tensorsol = solve(tensorprob,Tsit5(),reltol=1e-10,abstol=1e-10)
        gsol(x) = tensorsol(x)[1]
        gpsol(x) = tensorsol(x)[2]
        gkList[i] = [p[1],gsol,gpsol]
    end
    return gkList
end;

g1RD(x) = cos(x);
g1pRD(x) = -sin(x);
g2RD(x) = sin(x);
g2pRD(x) = cos(x);

ui1 = [g1RD(xi),g1pRD(xi)];
ui2 = [g2RD(xi),g2pRD(xi)];

println("Tensor EoM being solved...")
@time g1sol = gkList(ui1);
@time g2sol = gkList(ui2);

println("Data being exported...")
@time begin  
    open("data/g1.dat","w") do out
        Base.print_array(out, [g1sol[j][2](xList[i]) for i=1:length(xList), j=1:length(kList)])
    end

    open("data/g1p.dat","w") do out
        Base.print_array(out, [g1sol[j][3](xList[i]) for i=1:length(xList), j=1:length(kList)])
    end

    open("data/g2.dat","w") do out
        Base.print_array(out, [g2sol[j][2](xList[i]) for i=1:length(xList), j=1:length(kList)])
    end

    open("data/g2p.dat","w") do out
        Base.print_array(out, [g2sol[j][3](xList[i]) for i=1:length(xList), j=1:length(kList)])
    end
end;

hkRD(x) = sin(x)/x;
hpRD(x) = ForwardDiff.derivative(hkRD,x);

function hkEoM(du,u,p,x) # u[1] = h, u[2] = dhdx, p[1] = k
    du[1] = u[2]
    du[2] = -2*calHint(x/p[1])/p[1]*u[2] - u[1]
end;

xspan = (xi, xf);
ui = [hkRD(xi),hpRD(xi)];

function hkList()
    hkList = [[] for i=1:length(kList)]
    Threads.@threads for i=1:length(kList)
        p = [kList[i]]
        tensorprob = ODEProblem(hkEoM,ui,xspan,p)
        tensorsol = solve(tensorprob,Tsit5(),reltol=1e-10,abstol=1e-10)
        hsol(x) = tensorsol(x)[1]
        hpsol(x) = tensorsol(x)[2]
        hkList[i] = [p[1],hsol,hpsol]
    end
    return hkList
end;

println("Linear tensor being solved...")
@time hksol = hkList();

println("Data being exported...")
@time begin
    open("data/hk.dat","w") do out
        Base.print_array(out, [hksol[j][2](xList[i]) for i=1:length(xList), j=1:length(kList)])
    end

    open("data/hkp.dat","w") do out
        Base.print_array(out, [hksol[j][3](xList[i]) for i=1:length(xList), j=1:length(kList)])
    end
end;

println("Tensor data being loaded...")
@time begin
    f = open("data/g1.dat","r");
    body = readlines(f);
    close(f)

    strarray = [split(body[i],r"\s+",keepempty=false) for i=1:length(body)];
    numarray = [parse.(Float64, strarray[i]) for i=1:length(strarray)];

    g1List = reduce(hcat,numarray);


    f = open("data/g1p.dat","r");
    body = readlines(f);
    close(f)

    strarray = [split(body[i],r"\s+",keepempty=false) for i=1:length(body)];
    numarray = [parse.(Float64, strarray[i]) for i=1:length(strarray)];

    g1pList = reduce(hcat,numarray);


    f = open("data/g2.dat","r");
    body = readlines(f);
    close(f)

    strarray = [split(body[i],r"\s+",keepempty=false) for i=1:length(body)];
    numarray = [parse.(Float64, strarray[i]) for i=1:length(strarray)];

    g2List = reduce(hcat,numarray);


    f = open("data/g2p.dat","r");
    body = readlines(f);
    close(f)

    strarray = [split(body[i],r"\s+",keepempty=false) for i=1:length(body)];
    numarray = [parse.(Float64, strarray[i]) for i=1:length(strarray)];

    g2pList = reduce(hcat,numarray);
end;

kmin = 1e4;
kmax = 1e9;
kList = [10^lnk for lnk=log10(kmin):0.01:log10(kmax)];

xi = 1e-2;
xf = 400;
xList = [x for x=xi:0.01:xf];

lnkspan = log10(kmin):0.01:log10(kmax);
xspan = xi:0.01:xf;

g1CSI = CubicSplineInterpolation((lnkspan, xspan), g1List);
g1int(k,x) = g1CSI(log10(k),x);

g1pCSI = CubicSplineInterpolation((lnkspan, xspan), g1pList);
g1pint(k,x) = g1pCSI(log10(k),x);

g2CSI = CubicSplineInterpolation((lnkspan, xspan), g2List);
g2int(k,x) = g2CSI(log10(k),x);

g2pCSI = CubicSplineInterpolation((lnkspan, xspan), g2pList);
g2pint(k,x) = g2pCSI(log10(k),x);

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

println("Linear tensor being loaded...")
@time begin
    f = open("data/hk.dat","r");
    body = readlines(f);
    close(f)

    strarray = [split(body[i],r"\s+",keepempty=false) for i=1:length(body)];
    numarray = [parse.(Float64, strarray[i]) for i=1:length(strarray)];

    hList = reduce(hcat,numarray);


    f = open("data/hkp.dat","r");
    body = readlines(f);
    close(f)

    strarray = [split(body[i],r"\s+",keepempty=false) for i=1:length(body)];
    numarray = [parse.(Float64, strarray[i]) for i=1:length(strarray)];

    hpList = reduce(hcat,numarray);
end;

kmin = 1e4;
kmax = 1e9;
kList = [10^lnk for lnk=log10(kmin):0.01:log10(kmax)];

xi = 1e-2;
xf = 400;
xList = [x for x=xi:0.01:xf];

lnkspan = log10(kmin):0.01:log10(kmax);
xspan = xi:0.01:xf;

hCSI = CubicSplineInterpolation((lnkspan, xspan), hList);
hint(k,x) = hCSI(log10(k),x);

hpCSI = CubicSplineInterpolation((lnkspan, xspan), hpList);
hpint(k,x) = hpCSI(log10(k),x);

dx = 2*2*π;

function h2bar(k)
    f(x) = hint(k,x)*hint(k,x)
    return quadgk(f,xf-dx,xf)[1]/dx
end;

hRD(x) = sin(x)/x;
h2intRD(x) = hRD(x)*hRD(x);
h2barRD = quadgk(h2intRD,xf-dx,xf)[1]/dx

println("Coefficient data being calculated and exported...")
@time begin
    g1g1List = [g1g1bar(kList[i]) for i=1:length(kList)];
    g1g2List = [g1g2bar(kList[i]) for i=1:length(kList)];
    g2g2List = [g2g2bar(kList[i]) for i=1:length(kList)];
    coeffList = [(aint(xf/kList[i])*calHint(xf/kList[i])/aint(etaf)/calHint(etaf))^2 /24*(kList[i]/calHint(xf/kList[i]))^2 #*64/81/aint(xf/kList[i])^2 
                 for i=1:length(kList)];
    OlinList = [(aint(xf/kList[i])*calHint(xf/kList[i])/aint(etaf)/calHint(etaf))^2 /12*(kList[i]/calHint(xf/kList[i]))^2 * h2bar(kList[i]) for i=1:length(kList)];
    
    open("data/g1g2coeff.dat","w") do out
        Base.print_array(out, hcat(log10.(kList[:]),g1g1List[:],g1g2List[:],g2g2List[:],coeffList[:],OlinList[:]))
    end
end


