using LinearAlgebra

function Greedy_IDF(kmax,IDF;nspatial = nspatial)
    nb_ex = length(IDF)
    Ind = Int[]
    ErrL2IDF = [] #L2 norm on inverse distribution function
    Err_all = zeros(nb_ex) #L2 norm on inverse distribution function
    xf01 = range(0.,1.,length=nspatial)
    for k in 1:kmax
        # select worse density
        if k == 1
            push!(Ind,1)
        else # find index with maximum error
            m = maximum(Err_all)
            ind = findfirst(x->(x==m),Err_all)
            push!(Ind,ind)
            @show Ind, m
        end
        # compute the approximations for the training densities
        IDFtrain = [IDF[i] for i in Ind]
        for itarget in 1:nb_ex
            IDFtest = IDF[itarget]
            coef = PairDensitiesTests.opti_coef(IDFtrain,IDFtest)
            IDFapprox = sum(coef[i]*IDF[Ind[i]](xf01) for i in 1:k)
            Err_all[itarget] = norm(IDFapprox.-IDF[itarget](xf01),2)/nspatial
        end
        push!(ErrL2IDF,sqrt(sum(Err_all)./length(Err_all)))
    end
    return Ind, ErrL2IDF
end
