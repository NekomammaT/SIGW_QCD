using ForwardDiff, DifferentialEquations, QuadGK, MultiQuad, Interpolations, SpecialFunctions, LinearAlgebra

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

@time begin
    f = open("data/Phi.dat","r");
    body = readlines(f);
    close(f)

    strarray = [split(body[i],r"\s+",keepempty=false) for i=1:length(body)];
    numarray = [parse.(Float64, strarray[i]) for i=1:length(strarray)];

    PhiList = reduce(hcat,numarray);

    f = open("data/Pi.dat","r");
    body = readlines(f);
    close(f)

    strarray = [split(body[i],r"\s+",keepempty=false) for i=1:length(body)];
    numarray = [parse.(Float64, strarray[i]) for i=1:length(strarray)];

    PiList = reduce(hcat,numarray);
end;

PhiRD1(x) = 9/x^2 * (sin(x/sqrt(3))/(x/sqrt(3)) - cos(x/sqrt(3))); # Phi in exact RD
PhiRD2(x) = 9/x^2 * (cos(x/sqrt(3))/(x/sqrt(3)) + sin(x/sqrt(3)));
PhipRD1(x) = ForwardDiff.derivative(PhiRD1,x);
PhipRD2(x) = ForwardDiff.derivative(PhiRD2,x);
MPhi(x) = [PhiRD1(x) PhiRD2(x); PhipRD1(x) PhipRD2(x)];
PhiRDc(x,xc,Phic,Pic) = (MPhi(x)*inv(MPhi(xc))*[Phic,Pic])[1];
PhipRDc(x,xc,Phic,Pic) = (MPhi(x)*inv(MPhi(xc))*[Phic,Pic])[2];

kmin = 1e4;
kmax = 1e9;
kList = [10^lnk for lnk=log10(kmin):0.01:log10(kmax)];

xi = 1e-2;
xf = 400;
xList = [x for x=xi:0.01:xf];

lnkspan = log10(kmin):0.01:log10(kmax);
xspan = xi:0.01:xf;

PhiCSI = CubicSplineInterpolation((lnkspan, xspan), PhiList);
function Phiint(k,x)
    if x < xi
        return PhiRD1(x)
    elseif x > xf
        return PhiRDc(x,xf,PhiCSI(log10(k),xf),PiCSI(log10(k),xf)) 
    else
        return PhiCSI(log10(k),x)
    end;
end;

PiCSI = CubicSplineInterpolation((lnkspan, xspan), PiList);
function Piint(k,x) 
    if x < xi
        return PhipRD1(x)
    elseif x > xf
        return PhipRDc(x,xf,PhiCSI(log10(k),xf),PiCSI(log10(k),xf))
    else
        return PiCSI(log10(k),x)
    end
end;

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

g1RD(x) = cos(x);
g1pRD(x) = -sin(x);
g2RD(x) = sin(x);
g2pRD(x) = cos(x);
Mg(x) = [g1RD(x) g2RD(x); g1pRD(x) g2pRD(x)];
gRDc(x,xc,gc,gpc) = (Mg(x)*inv(Mg(xc))*[gc,gpc])[1];
gpRDc(x,xc,gc,gpc) = (Mg(x)*inv(Mg(xc))*[gc,gpc])[2];

kmin = 1e4;
kmax = 1e9;
kList = [10^lnk for lnk=log10(kmin):0.01:log10(kmax)];

xi = 1e-2;
xf = 400;
xList = [x for x=xi:0.01:xf];

lnkspan = log10(kmin):0.01:log10(kmax);
xspan = xi:0.01:xf;

g1CSI = CubicSplineInterpolation((lnkspan, xspan), g1List);
function g1int(k,x) 
    if x < xi
        return g1RD(x)
    elseif x > xf
        return gRDc(x,xf,g1CSI(log10(k),xf),g1pCSI(log10(k),xf)) 
    else
        return g1CSI(log10(k),x)
    end
end;

g1pCSI = CubicSplineInterpolation((lnkspan, xspan), g1pList);
function g1pint(k,x) 
    if x < xi
        return g1pRD(x)
    elseif x > xf
        return gpRDc(x,xf,g1CSI(log10(k),xf),g1pCSI(log10(k),xf)) 
    else
        return g1pCSI(log10(k),x)
    end
end;

g2CSI = CubicSplineInterpolation((lnkspan, xspan), g2List);
function g2int(k,x) 
    if x < xi
        return g2RD(x)
    elseif x > xf
        return gRDc(x,xf,g2CSI(log10(k),xf),g2pCSI(log10(k),xf)) 
    else
        return g2CSI(log10(k),x)
    end
end;

g2pCSI = CubicSplineInterpolation((lnkspan, xspan), g2pList);
function g2pint(k,x) 
    if x < xi
        return g2pRD(x)
    elseif x > xf
        return gpRDc(x,xf,g2CSI(log10(k),xf),g2pCSI(log10(k),xf)) 
    else
        return g2pCSI(log10(k),x)
    end
end;

f = open("data/g1g2coeff.dat","r");
body = readlines(f);
close(f)

strarray = [split(body[i],r"\s+",keepempty=false) for i=1:length(body)];
numarray = [parse.(Float64, strarray[i]) for i=1:length(strarray)];

lnkList = [numarray[i][1] for i=1:length(numarray)];
g1g1List = [numarray[i][2] for i=1:length(numarray)];
g1g2List = [numarray[i][3] for i=1:length(numarray)];
g2g2List = [numarray[i][4] for i=1:length(numarray)];
coeffList = [numarray[i][5] for i=1:length(numarray)];
OlinList = [numarray[i][6] for i=1:length(numarray)];

lnkspan = range(lnkList[1], lnkList[length(lnkList)], length=length(lnkList))
kList = [10^lnk for lnk in lnkspan]

g1g1CSI = CubicSplineInterpolation(lnkspan, g1g1List);
g1g1int(k) = g1g1CSI(log10(k));

g1g2CSI = CubicSplineInterpolation(lnkspan, g1g2List);
g1g2int(k) = g1g2CSI(log10(k));

g2g2CSI = CubicSplineInterpolation(lnkspan, g2g2List);
g2g2int(k) = g2g2CSI(log10(k));

coeffCSI = CubicSplineInterpolation(lnkspan, coeffList);
coeffint(k) = coeffCSI(log10(k));

OlinCSI = CubicSplineInterpolation(lnkspan, OlinList);
Olinint(k) = OlinCSI(log10(k));

function I1(k,s,t)
    u = (t+s+1)/2
    v = (t-s+1)/2
    k1 = u*k
    k2 = v*k
    if (k1 < kmin || k1 > kmax || k2 < kmin || k2 > kmax)
        return 0
    else
        function f(x)
            x1 = k1/k*x
            x2 = k2/k*x
            eta = x/k
            return (4/9*aint(eta)/aint(xf/k)*g1int(k,x) * (2*Phiint(k1,x1)*Phiint(k2,x2) 
                    + 4/3/(1+EoSwint(eta))*(Phiint(k1,x1)+Piint(k1,x1)*k1/calHint(eta))*(Phiint(k2,x2)+Piint(k2,x2)*k2/calHint(eta))))
        end
        x1i = k/k1*xi
        x2i = k/k2*xi
        x1f = k/k1*xf
        x2f = k/k2*xf
        if u > 10 && v > 10
            return -quadgk(f,min(xi,x1i,x2i),max(min(xf,x1f,x2f),10);rtol=1e-10,atol=1e-11)[1]
        else
            return -quadgk(f,min(xi,x1i,x2i),xf)[1]
        end
    end
end;

function I2(k,s,t)
    u = (t+s+1)/2
    v = (t-s+1)/2
    k1 = u*k
    k2 = v*k
    if (k1 < kmin || k1 > kmax || k2 < kmin || k2 > kmax)
        return 0
    else
        function f(x)
            x1 = k1/k*x
            x2 = k2/k*x
            eta = x/k
            return (4/9*aint(eta)/aint(xf/k)*g2int(k,x) * (2*Phiint(k1,x1)*Phiint(k2,x2) 
                    + 4/3/(1+EoSwint(eta))*(Phiint(k1,x1)+Piint(k1,x1)*k1/calHint(eta))*(Phiint(k2,x2)+Piint(k2,x2)*k2/calHint(eta))))
        end
        x1i = k/k1*xi
        x2i = k/k2*xi
        x1f = k/k1*xf
        x2f = k/k2*xf
        if u > 10 && v > 10
            return -quadgk(f,min(xi,x1i,x2i),max(min(xf,x1f,x2f),10);rtol=1e-10,atol=1e-11)[1]
        else
            return -quadgk(f,min(xi,x1i,x2i),xf)[1]
        end
    end
end;

lntspan = -5.:0.01:5;
sspan = 0:0.01:1;
#lntspan = -5:0.1:5;
#sspan = 0:0.1:1;
tList = [10^lnt for lnt=lntspan];
sList = [s for s=sspan];

function I1List(k)
    I1List = zeros(length(sList),length(tList))
    Threads.@threads for i=1:length(sList)
        for j=1:length(tList)
            I1List[i,j] = I1(k,sList[i],tList[j])
        end
    end
    return I1List
end;

function I2List(k)
    I2List = zeros(length(sList),length(tList))
    Threads.@threads for i=1:length(sList)
        for j=1:length(tList)
            I2List[i,j] = I2(k,sList[i],tList[j])
        end
    end
    return I2List
end;

@time for i=71:10:91
    #if i%10 == 1
    #    continue
    #else
        open(string("data/Is/I1_", i, ".dat"),"w") do out
            Base.print_array(out, I1List(kList[i]))
        end

        open(string("data/Is/I2_", i, ".dat"),"w") do out
            Base.print_array(out, I2List(kList[i]))
        end
    #end
end;
