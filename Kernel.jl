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

kmin = 1e4;
kmax = 1e9;
kList = [10^lnk for lnk=log10(kmin):0.01:log10(kmax)];
lnkList = log10.(kList);


println("# of threads : ", Threads.nthreads())

if !(length(ARGS) == 1)
    println("Specify a correct value of iGW.")
    exit(0)
end

iGW = parse(Int,ARGS[1]);

if !(1 <= iGW <= length(kList))
    println("Specify a correct value of iGW.")
    exit(0)
end

println("iGW = ", iGW)


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

PhiRD1(x) = 9/x^2 * (sin(x/sqrt(3))/(x/sqrt(3)) - cos(x/sqrt(3))); # Phi in exact RD
PhiRD2(x) = 9/x^2 * (cos(x/sqrt(3))/(x/sqrt(3)) + sin(x/sqrt(3)));
PhipRD1(x) = ForwardDiff.derivative(PhiRD1,x);
PhipRD2(x) = ForwardDiff.derivative(PhiRD2,x);
MPhi(x) = [PhiRD1(x) PhiRD2(x); PhipRD1(x) PhipRD2(x)];
PhiRDc(x,xc,Phic,Pic) = (MPhi(x)*inv(MPhi(xc))*[Phic,Pic])[1];
PhipRDc(x,xc,Phic,Pic) = (MPhi(x)*inv(MPhi(xc))*[Phic,Pic])[2];

xi = 1e-2;
xf = 400;

scalardata = readdlm("data/scalar.csv", ',');
xList = scalardata[1,:];
PhiList = scalardata[2:1+length(kList),:];
PiList = scalardata[2+length(kList):size(scalardata)[1],:];

PhiCSI = Spline2D(lnkList,xList,PhiList,kx=3,ky=3);
PiCSI = Spline2D(lnkList,xList,PiList,kx=3,ky=3);

function Phiint(k,x) 
    if x < xi
        return PhiRD1(x)
    elseif x > xf
        return PhiRDc(x,xf,PhiCSI(log10(k),xf),PiCSI(log10(k),xf))
    else 
        return PhiCSI(log10(k),x)
    end
end;

function Piint(k,x) 
    if x < xi
        return PhipRD1(x)
    elseif x > xf
        return PhipRDc(x,xf,PhiCSI(log10(k),xf),PiCSI(log10(k),xf))
    else 
        return PiCSI(log10(k),x)
    end
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
lnkList = log10.(kList);

xi = 1e-2;
xf = 400;

g1data = readdlm("data/g1.csv", ',');
x1List = g1data[1,:];
g1List = g1data[2:1+length(kList),:];
g1pList = g1data[2+length(kList):size(g1data)[1],:];

g2data = readdlm("data/g2.csv", ',');
x2List = g2data[1,:];
g2List = g2data[2:1+length(kList),:];
g2pList = g2data[2+length(kList):size(g2data)[1],:];

g1CSI = Spline2D(lnkList,x1List,g1List,kx=3,ky=3);
g1pCSI = Spline2D(lnkList,x1List,g1pList,kx=3,ky=3);
g2CSI = Spline2D(lnkList,x2List,g2List,kx=3,ky=3);
g2pCSI = Spline2D(lnkList,x2List,g2pList,kx=3,ky=3);

function g1int(k,x) 
    if x < xi
        return g1RD(x)
    elseif x > xf
        return gRDc(x,xf,g1CSI(log10(k),xf),g1pCSI(log10(k),xf))
    else 
        return g1CSI(log10(k),x)
    end
end;

function g1pint(k,x) 
    if x < xi
        return g1pRD(x)
    elseif x > xf
        return gpRDc(x,xf,g1CSI(log10(k),xf),g1pCSI(log10(k),xf))
    else 
        return g1pCSI(log10(k),x)
    end
end;

function g2int(k,x) 
    if x < xi
        return g2RD(x)
    elseif x > xf
        return gRDc(x,xf,g2CSI(log10(k),xf),g2pCSI(log10(k),xf))
    else 
        return g2CSI(log10(k),x)
    end
end;

function g2pint(k,x) 
    if x < xi
        return g2pRD(x)
    elseif x > xf
        return gpRDc(x,xf,g2CSI(log10(k),xf),g2pCSI(log10(k),xf))
    else 
        return g2pCSI(log10(k),x)
    end
end;

coeffdata = readdlm("data/g1g2coeff.csv", ',');

lnkList = coeffdata[:,1];
g1g1List = coeffdata[:,2];
g1g2List = coeffdata[:,3];
g2g2List = coeffdata[:,4];
coeffList = coeffdata[:,5];
OlinList = coeffdata[:,6];

g1g1CSI = Spline1D(lnkList, g1g1List, k=3);
g1g1int(k) = g1g1CSI(log10(k));

g1g2CSI = Spline1D(lnkList, g1g2List, k=3);
g1g2int(k) = g1g2CSI(log10(k));

g2g2CSI = Spline1D(lnkList, g2g2List, k=3);
g2g2int(k) = g2g2CSI(log10(k));

coeffCSI = Spline1D(lnkList, coeffList, k=3);
coeffint(k) = coeffCSI(log10(k));

OlinCSI = Spline1D(lnkList, OlinList, k=3);
Olinint(k) = OlinCSI(log10(k));


xcut = 10;

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
        if u > xcut && v > xcut
            return -quadgk(f,min(xi,x1i,x2i),max(min(xf,x1f,x2f),xcut);rtol=1e-10,atol=1e-12)[1]
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
        if u > xcut && v > xcut
            return -quadgk(f,min(xi,x1i,x2i),max(min(xf,x1f,x2f),xcut);rtol=1e-10,atol=1e-12)[1]
        else
            return -quadgk(f,min(xi,x1i,x2i),xf)[1]
        end
    end
end;

#lntspan = -5.:0.01:5;
#sspan = 0:0.01:1;
lntspan = -5:0.1:5;
sspan = 0:0.1:1;
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

@time writedlm(string("data/Is/I1_", iGW, ".dat"), I1List(kList[iGW]));
@time writedlm(string("data/Is/I2_", iGW, ".dat"), I2List(kList[iGW]));

println("Completed.")

