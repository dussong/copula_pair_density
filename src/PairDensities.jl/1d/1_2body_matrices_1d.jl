#----------------------------------------------------------------------------
#
#   Computation of the one-body and two-body matrices,
#   respectively matrices and 4-dim. tensors
#----------------------------------------------------------------------------

export oneB_over, oneB_V, oneB_lap, twoB_V, twoB_V_sp, twoB_V_sp2, matrix_1_2body_1d

#----------------------------------------------------------------------------
# One-body matrices
#----------------------------------------------------------------------------

# Overlap matrix: returns overlap matrix for P1 finite elements
function oneB_over(L::Float64, N::Int)
    # interval -L:L
    # N-1 degrees of freedom per dimension
    h = (2.0 * L) / N
    int_idem = 2.0 * h / 3.0
    int_next = h / 6.0

    # overlap i,i
    indi = collect(1:(N-1))
    indj = collect(1:(N-1))
    overij = repeat([int_idem], N - 1)

    # overlap i,i+1
    append!(indi, collect(1:(N-2)))
    append!(indj, collect(2:(N-1)))
    append!(overij, repeat([int_next], N - 2))

    # overlap i,i-1
    append!(indi, collect(2:(N-1)))
    append!(indj, collect(1:(N-2)))
    append!(overij, repeat([int_next], N - 2))

    return sparse(indi, indj, overij)
end


function oneB_V(L::Float64, N::Int, V::Function, nx::Int)
    # One-body matrix with P1 finite elements basis
    # computed with numerical integration using Gauss-Legendre points.
    h = 2.0 * L / N
    # Gauss-Legendre points
    x, w_x = gausslegendre(nx)
    wt = w_x .* h / 2.0

    indi1 = Int[]
    indi2 = Int[]
    one_body_values = Float64[]

    xl1 = zeros(length(x))
    xr1 = zeros(length(x))
    fl1 = zeros(length(x))
    fr1 = zeros(length(x))
    Potl1 = zeros(length(x))
    Potr1 = zeros(length(x))

    # i2 = i1
    for i1 in 1:(N-1)
        xl1 .= .-L .+ h * (i1 .- 0.5 .+ x / 2)
        xr1 .= .-L .+ h * (i1 .+ 0.5 .+ x / 2)
        #fl1 .= (x.-1.0)/2.
        #fr1 .= .-(x.+1.0)/2.
        fl1 .= (x .+ 1.0) / 2.0
        fr1 .= .-(x .- 1.0) / 2.0
        Potl1 = V.(xl1)
        Potr1 = V.(xr1)

        v = sum(fl1 .* fl1 .* Potl1 .* wt) + sum(fr1 .* fr1 .* Potr1 .* wt)
        push!(indi1, i1)
        push!(indi2, i1)
        push!(one_body_values, v)

        if i1 < N - 1
            v = sum(fl1 .* fr1 .* Potr1 .* wt)
            push!(indi1, i1)
            push!(indi2, i1 + 1)
            push!(one_body_values, v)

            # v = sum(fl1.*fr1.*Potl1.*wt)
            push!(indi1, i1 + 1)
            push!(indi2, i1)
            push!(one_body_values, v)
        end
    end
    return sparse(indi1, indi2, one_body_values)
end


# Laplace matrix (1 body) for P1 finite elements
function oneB_lap(L::Float64, N::Int)
    # interval -L:L
    # N-1 degrees of freedom per dimension (Dirichlet boundary conditions)
    h = (2 * L) / N
    int_idem = 2 / h
    int_next = -1 / h

    # laplace i,i
    indi = collect(1:(N-1))
    indj = collect(1:(N-1))
    lapij = repeat([int_idem], N - 1)

    # laplace i,i+1
    append!(indi, collect(1:(N-2)))
    append!(indj, collect(2:(N-1)))
    append!(lapij, repeat([int_next], N - 2))

    # laplace i-1,i
    append!(indi, collect(2:(N-1)))
    append!(indj, collect(1:(N-2)))
    append!(lapij, repeat([int_next], N - 2))

    return sparse(indi, indj, lapij)
end


#----------------------------------------------------------------------------
# Two-body matrices for a local potential V
#----------------------------------------------------------------------------
function twoB_V(L::Float64, N::Int, V::Function, nx::Int, ny::Int)
    # returns a 4-dim. tensor
    h = 2.0 * L / N
    mat = zeros(N - 1, N - 1, N - 1, N - 1) #initialise tensor

    # Gauss-Legendre points (for the numerical integration)
    p_x, w_x = gausslegendre(nx)
    p_y, w_y = gausslegendre(ny)

    w_X = [w_x[i] for i = 1:nx for j = 1:ny]
    w_Y = [w_y[j] for i = 1:nx for j = 1:ny]
    wt = w_X .* w_Y .* h^2 / 4.0

    x = [p_x[i] for i = 1:nx for j = 1:ny]
    y = [p_y[j] for i = 1:nx for j = 1:ny]

    # value and gradient of basis functions at (x,y)
    # the hat functions are separated into 2 parts,
    # and we compute the 4 possibilities of product of 2
    f = hcat((x .- 1.0) .* (y .- 1.0) / 4.0,
        .-(x .+ 1.0) .* (y .- 1.0) / 4.0,
        .-(x .- 1.0) .* (y .+ 1.0) / 4.0,
        (x .+ 1.0) .* (y .+ 1.0) / 4.0)

    for j = 1:N #loop over y dof
        yy = .-L .+ h * (j .- 0.5 .+ y / 2) # y coordinates
        ylabel = [j - 1, j - 1, j, j] # label for the dof on the edges
        for i = 1:N #loop over x dof
            xx = .-L .+ h * (i .- 0.5 .+ x / 2) # x coordinates
            xlabel = [i - 1, i, i - 1, i] # label for the dof on the edges

            r = V(xx - yy) #values for V
            # calculate stiff elements
            for k = 1:4
                for l = 1:4
                    ix = xlabel[k]
                    iy = ylabel[k]
                    jx = xlabel[l]
                    jy = ylabel[l]
                    if ix > 0 && ix < N && iy > 0 && iy < N && jx > 0 && jx < N && jy > 0 && jy < N #checking admissible basis functions
                        mat[ix, iy, jx, jy] += sum(r .* f[:, k] .* f[:, l] .* wt)
                    end
                end
            end
        end
    end
    return mat
end


function twoB_V_sp(L::Float64, N::Int, V::Function, nx::Int, ny::Int)
    #for P1 finite elements with Dirichlet boundary conditions
    h = 2.0 * L / N
    nb_dof = 9N^2 - 30N + 25
    indix = zeros(Int, 4, nb_dof) #initialise matrix of 4-indices
    vals = zeros(Float64, nb_dof) #initialise values
    nb_nz = 0
    for ix in 1:(N-1), jx in 1:(N-1), iy in 1:(N-1), jy in 1:(N-1)
        if (abs(ix - jx) <= 1) && (abs(iy - jy) <= 1) #check nonzero overlap
            nb_nz += 1
            indix[:, nb_nz] = [ix, iy, jx, jy]
        end
    end
    # Gauss-Legendre points
    p_x, w_x = gausslegendre(nx)
    p_y, w_y = gausslegendre(ny)
    wt = [w_x[i] * w_y[j] * h^2 / 4.0 for i = 1:nx for j = 1:ny] #weights for integration
    x = [p_x[i] for i = 1:nx for j = 1:ny] #spatial points
    y = [p_y[j] for i = 1:nx for j = 1:ny]

    # value of basis functions at (x,y)
    f = hcat((x .- 1.0) .* (y .- 1.0) / 4.0, .-(x .+ 1.0) .* (y .- 1.0) / 4.0,
        .-(x .- 1.0) .* (y .+ 1.0) / 4.0, (x .+ 1.0) .* (y .+ 1.0) / 4.0)

    for j = 1:N #loop over y
        yy = .-L .+ h * (j .- 0.5 .+ y / 2) # y coordinate
        ylabel = [j - 1, j - 1, j, j] # label
        indy = findall(x -> (x < N) && (x > 0), ylabel)
        for i = 1:N
            xx = .-L .+ h * (i .- 0.5 .+ x / 2) # x coordinate
            xlabel = [i - 1, i, i - 1, i] # label
            indx = findall(x -> (x < N) && (x > 0), xlabel)
            ind = intersect(indx, indy)

            r = V(xx - yy)
            for k in ind, l in ind
                ix = xlabel[k]
                iy = ylabel[k]
                jx = xlabel[l]
                jy = ylabel[l]
                index = ((3 * N - 5) * ((jx - ix + 1) + 3(ix - 1) - 1) + (jy - iy + 1) + 3(iy - 1))
                vals[index] += sum(1.0 .* r .* f[:, k] .* f[:, l] .* wt)
            end
        end
    end
    return (indix, vals)
end

function twoB_V_sp2(L::Float64, N::Int, V::Function, nx::Int, ny::Int)
    #for P1 finite elements with Dirichlet boundary conditions
    h = 2.0 * L / N
    I = Int[]
    J = Int[]
    vals = Float64[]

    # Gauss-Legendre points
    p_x, w_x = gausslegendre(nx)
    p_y, w_y = gausslegendre(ny)
    wt = [w_x[i] * w_y[j] * h^2 / 4.0 for i = 1:nx for j = 1:ny] #weights for integration
    x = [p_x[i] for i = 1:nx for j = 1:ny] #spatial points
    y = [p_y[j] for i = 1:nx for j = 1:ny]

    # value of basis functions at (x,y)
    f = hcat((x .- 1.0) .* (y .- 1.0) / 4.0, .-(x .+ 1.0) .* (y .- 1.0) / 4.0,
        .-(x .- 1.0) .* (y .+ 1.0) / 4.0, (x .+ 1.0) .* (y .+ 1.0) / 4.0)

    for j = 1:N #loop over y
        yy = .-L .+ h * (j .- 0.5 .+ y / 2) # y coordinate
        ylabel = [j - 1, j - 1, j, j] # label
        for i = 1:N
            xx = .-L .+ h * (i .- 0.5 .+ x / 2) # x coordinate
            xlabel = [i - 1, i, i - 1, i] # label

            r = V(xx - yy)
            for k in 1:4, l in 1:4
                ix = xlabel[k]
                iy = ylabel[k]
                jx = xlabel[l]
                jy = ylabel[l]
                if ix > 0 && ix < N && iy > 0 && iy < N && jx > 0 && jx < N && jy > 0 && jy < N #checking admissible basis functions
                    push!(I, (iy - 1) * (N - 1) + ix)
                    push!(J, (jy - 1) * (N - 1) + jx)
                    push!(vals, sum(@. 1.0 * r * f[:, k] * f[:, l] * wt))
                    #push!(vals,1.)
                end
            end
        end
    end
    return sparse(I, J, vals)
end

# generate 1-body and 2-body matrices
function matrix_1_2body_1d(L::Float64, N::Int, vext::Function, vee::Function, nx::Int64, ny::Int64)

    # construct the 1-body matrix for -1/2⋅Δ, external potential and overlap
    AΔ = oneB_lap(L, N)
    AV = oneB_V(L, N, x -> vext(x), nx)
    C = oneB_over(L, N)
    # construct the 2-body matrix for electron-electron interaction v_ee
    Bee1 = twoB_V(L, N, x -> vee(x), nx, ny)
    Bee2 = twoB_V(L, N, x -> vee(x), ny, nx)
    Bee = (Bee1 + Bee2) ./ 2
    return AΔ, AV, C, Bee
end
