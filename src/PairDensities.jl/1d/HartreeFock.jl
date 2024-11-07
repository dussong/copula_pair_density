"""
Call the Hartree-Fock solver for Helium:
vee = x->1.0/sqrt(x^2+1);
N = 4;
Z = 4.;
Norb = 2;
L = 8.;
ats = [Atom([0.0], 1, N, x->-Z*vee(x))];
C = Cluster(-L, L, ats);
n = 100;
E, λ, ρ, rho = hf(C, Norb, n);
"""

function hf(C::Cluster, 
            Norb::Int, 
            n::Int;
            ng = 4,
            max_iter = 80, 
            mixing = 0.8,
            scf_tol = 1e-5)
    # ===== solve linear problem to initialize ρ =====#
    L = (C.x1+C.x0>0) ? C.x1 : -C.x0;
    # h = 2 * L / n;
    M = mass1b(L, n);
    AΔ = lapl1d(L, n);
    vext = x -> atom_V(C,x);
    AV = potl1b(L, n, vext, ng);
    A = 0.5 * AΔ + AV;
    λ, W = eigs(A, M; nev=Norb, which=:SR);
    ρ1 = zeros(n-1,n-1);
    for k = 1 : Norb
        ρ1 += 2.0 * W[:,k] * W[:,k]';
    end
    ρ = copy(ρ1);

    # ===== start the SCF iterations =====#
    err = 1.;
    k1 = mixing; k2 = 1. - k1;  # charge mixing parameter
    Bee = vee2B(L, n, x->1.0/sqrt(x^2+1));
    rho = zeros(n-1, max_iter);
    for k = 1 : max_iter
        if err < scf_tol 
            break;
        end
        AH = genH(n, ρ, Bee);
        AF = genF(n, ρ, Bee);
        H = A + AH - 0.5 * AF;
        λ, W = eigs(H, M; nev=Norb, which=:SR); 
        ρ2 = zeros(n-1, n-1);
        for j = 1 : Norb
            ρ2 += 2.0 * W[:,j] * W[:,j]';
        end
        ρ = k1.*ρ1 + k2.*ρ2;
        err = norm(ρ2-ρ1);
        ρ1 = ρ;
        println(" step : $(k),  err : $(err)");
        rho[:,k] = [ρ[j,j] for j = 1:n-1];
    end

    #===== evaluate the Hartree-Fock energy =====#
    E = energyHF(n, λ, ρ, Bee);
    println("Hartree-Fock energy : ", E);

    return E, λ, ρ, rho
end


function genH(n::Int, ρ, B)
    AH = zeros(n-1,n-1);
    for i=1:n-1, j=1:n-1
        for a=1:n-1, b=1:n-1
            AH[i,j] += B[i,a,j,b] * ρ[a,b];
        end
    end
    return AH
end

function genF(n::Int, ρ, B)
    AF = zeros(n-1,n-1);
    for i=1:n-1, j=1:n-1
        for a=1:n-1, b=1:n-1
            AF[i,j] += B[a,b,i,j] * ρ[a,b];
        end
    end
    return AF
end

function energyHF(n::Int, λ, ρ, B)
    E = 2.0 * sum(λ);
    for i=1:n-1, j=1:n-1, a=1:n-1, b=1:n-1
        E -= 0.5 * B[i,a,j,b] * ρ[i,j] * ρ[a,b];
        E += 0.25 * B[a,b,i,j] * ρ[i,j] * ρ[a,b];
    end
    # for i=1:n-1, j=1:n-1
    #     E -= 0.5 * AH[i,j] * ρ[i,j]
    #     E += 0.25 * AF[i,j] * ρ[i,j]
    # end
    return E
end
