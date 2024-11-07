using Dierckx

export ρ_T_cop

#In what follows:
# L spatial parameter
# N discretisation parameter
# nel number of electrons
# nspatial spatial fine discretisation parameter

function ρ_T_cop(ρ,ρ2,nel,L,N,nspatial)
    xc = range(-L,L,length=N-1) #spatial grid
    xf = range(-L,L,length=nspatial) #fine spatial grid

    sρ = Spline1D(xc,ρ) #spline representation of density
    Tf = [integrate(sρ, -L, y) for y in xf]./nel #T: cumulative distr. function
    Tf = Tf./maximum(Tf) #transport map
    indf = [findall(x->(x!=0),Tf[2:end]-Tf[1:end-1]);length(Tf)] #remove singularity
    sT = Spline1D(xf,Tf) #spline representation of T
    sF = Spline1D(Tf[indf],xf[indf]) #spline representation of F (inverse of T)
    ratio_MF = 2*nel/(nel-1)* ρ2 ./ (ρ*ρ') #ratio to mean-field pair density

    Tc = [integrate(sρ, -L, y) for y in xc]./nel #T: cumulative distr. function
    Tc = Tc./maximum(Tc) #transport map
    indc = [findall(x->(x!=0),Tc[2:end]-Tc[1:end-1]);length(Tc)] #remove singularity
    scop = Spline2D(Tc[indc],Tc[indc],ratio_MF[indc,indc]) #repr. of cop with spline
    return sρ,sT,sF,scop
end
