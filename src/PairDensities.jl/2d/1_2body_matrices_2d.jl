
export oneB_over_2d, oneB_lap_2d, oneB_V_2d, twoB_V_2d_sp, matrix_1_2body_2d

#----------------------------------------------------------------------------
# One-body matrices
#----------------------------------------------------------------------------
function oneB_over_2d(Lx::T1, Ly::T1, Nx::T2, Ny::T2) where {T1<:AbstractFloat,T2<:Signed}

    hx = 2.0 * Lx / Nx
    hy = 2.0 * Ly / Ny
    int_idem_x = 2.0 * hx / 3.0
    int_next_x = hx / 6.0
    int_idem_y = 2.0 * hy / 3.0
    int_next_y = hy / 6.0
    mx = Int[]
    my = Int[]
    val = T1[]

    for j = 1:(Ny-1)
        #(i,j),(i,j)
        append!(mx, collect(1:(Nx-1)) .+ (j - 1) * (Nx - 1))
        append!(my, collect(1:(Nx-1)) .+ (j - 1) * (Nx - 1))
        append!(val, repeat([int_idem_x * int_idem_y], (Nx - 1)))

        #(i,j),(i+1,j)
        append!(mx, collect(1:(Nx-2)) .+ (j - 1) * (Nx - 1))
        append!(my, collect(2:(Nx-1)) .+ (j - 1) * (Nx - 1))
        append!(val, repeat([int_next_x * int_idem_y], (Nx - 2)))

        #(i-1,j),(i,j)
        append!(mx, collect(2:(Nx-1)) .+ (j - 1) * (Nx - 1))
        append!(my, collect(1:(Nx-2)) .+ (j - 1) * (Nx - 1))
        append!(val, repeat([int_next_x * int_idem_y], (Nx - 2)))

        if j < Ny - 1
            #(i,j),(i,j+1)
            append!(mx, collect(1:(Nx-1)) .+ (j - 1) * (Nx - 1))
            append!(my, collect(1:(Nx-1)) .+ j * (Nx - 1))
            append!(val, repeat([int_next_x * int_idem_y], (Nx - 1)))

            #(i,j),(i+1,j+1)
            append!(mx, collect(1:(Nx-2)) .+ (j - 1) * (Nx - 1))
            append!(my, collect(2:(Nx-1)) .+ j * (Nx - 1))
            append!(val, repeat([int_next_x * int_next_y], (Nx - 2)))

            #(i-1,j),(i,j+1)
            append!(mx, collect(2:(Nx-1)) .+ (j - 1) * (Nx - 1))
            append!(my, collect(1:(Nx-2)) .+ j * (Nx - 1))
            append!(val, repeat([int_next_x * int_next_y], (Nx - 2)))

            #(i,j-1),(i,j)
            append!(mx, collect(1:(Nx-1)) .+ j * (Nx - 1))
            append!(my, collect(1:(Nx-1)) .+ (j - 1) * (Nx - 1))
            append!(val, repeat([int_next_x * int_idem_y], (Nx - 1)))

            #(i,j-1),(i+1,j)
            append!(mx, collect(1:(Nx-2)) .+ j * (Nx - 1))
            append!(my, collect(2:(Nx-1)) .+ (j - 1) * (Nx - 1))
            append!(val, repeat([int_next_x * int_next_y], (Nx - 2)))

            #(i-1,j-1),(i,j)
            append!(mx, collect(2:(Nx-1)) .+ j * (Nx - 1))
            append!(my, collect(1:(Nx-2)) .+ (j - 1) * (Nx - 1))
            append!(val, repeat([int_next_x * int_next_y], (Nx - 2)))
        end
    end

    return sparse(mx, my, val)
end
oneB_over_2d(L::T1, N::T2) where {T1<:AbstractFloat,T2<:Signed} = oneB_over_2d(L, L, N, N)

# generate the stiff matrix for Δ
function oneB_lap_2d(Lx::T1, Ly::T1, Nx::T2, Ny::T2) where {T1<:AbstractFloat,T2<:Signed}

    hx = 2.0 * Lx / Nx
    hy = 2.0 * Ly / Ny
    int_idem_gx = 2 / hx
    int_next_gx = -1 / hx
    int_idem_ovx = 2.0 * hx / 3.0
    int_next_ovx = hx / 6.0
    int_idem_gy = 2 / hy
    int_next_gy = -1 / hy
    int_idem_ovy = 2.0 * hy / 3.0
    int_next_ovy = hy / 6.0
    mx = Int[]
    my = Int[]
    val = T1[]

    for j = 1:(Ny-1)
        #(i,j),(i,j)
        append!(mx, collect(1:(Nx-1)) .+ (j - 1) * (Nx - 1))
        append!(my, collect(1:(Nx-1)) .+ (j - 1) * (Nx - 1))
        append!(val, repeat([int_idem_gx * int_idem_ovy + int_idem_ovx * int_idem_gy], (Nx - 1)))

        #(i,j),(i+1,j)
        append!(mx, collect(1:(Nx-2)) .+ (j - 1) * (Nx - 1))
        append!(my, collect(2:(Nx-1)) .+ (j - 1) * (Nx - 1))
        append!(val, repeat([int_next_gx * int_idem_ovy + int_next_ovx * int_idem_gy], (Nx - 2)))

        #(i-1,j),(i,j)
        append!(mx, collect(2:(Nx-1)) .+ (j - 1) * (Nx - 1))
        append!(my, collect(1:(Nx-2)) .+ (j - 1) * (Nx - 1))
        append!(val, repeat([int_next_gx * int_idem_ovy + int_next_ovx * int_idem_gy], (Nx - 2)))

        if j < Ny - 1
            #(i,j),(i,j+1)
            append!(mx, collect(1:(Nx-1)) .+ (j - 1) * (Nx - 1))
            append!(my, collect(1:(Nx-1)) .+ j * (Nx - 1))
            append!(val, repeat([int_idem_gx * int_next_ovy + int_idem_ovx * int_next_gy], (Nx - 1)))

            #(i,j),(i+1,j+1)
            append!(mx, collect(1:(Nx-2)) .+ (j - 1) * (Nx - 1))
            append!(my, collect(2:(Nx-1)) .+ j * (Nx - 1))
            append!(val, repeat([int_next_gx * int_next_ovy + int_next_ovx * int_next_gy], (Nx - 2)))

            #(i-1,j),(i,j+1)
            append!(mx, collect(2:(Nx-1)) .+ (j - 1) * (Nx - 1))
            append!(my, collect(1:(Nx-2)) .+ j * (Nx - 1))
            append!(val, repeat([int_next_gx * int_next_ovy + int_next_ovx * int_next_gy], (Nx - 2)))

            #(i,j-1),(i,j)
            append!(mx, collect(1:(Nx-1)) .+ j * (Nx - 1))
            append!(my, collect(1:(Nx-1)) .+ (j - 1) * (Nx - 1))
            append!(val, repeat([int_idem_gx * int_next_ovy + int_idem_ovx * int_next_gy], (Nx - 1)))

            #(i,j-1),(i+1,j)
            append!(mx, collect(1:(Nx-2)) .+ j * (Nx - 1))
            append!(my, collect(2:(Nx-1)) .+ (j - 1) * (Nx - 1))
            append!(val, repeat([int_next_gx * int_next_ovy + int_next_ovx * int_next_gy], (Nx - 2)))

            #(i-1,j-1),(i,j)
            append!(mx, collect(2:(Nx-1)) .+ j * (Nx - 1))
            append!(my, collect(1:(Nx-2)) .+ (j - 1) * (Nx - 1))
            append!(val, repeat([int_next_gx * int_next_ovy + int_next_ovx * int_next_gy], (Nx - 2)))
        end
    end

    return sparse(mx, my, val)
end
oneB_lap_2d(L::T1, N::T2) where {T1<:AbstractFloat,T2<:Signed} = oneB_lap_2d(L, L, N, N)


function oneB_V_2d(Lx::T1, Ly::T1, Nx::T2, Ny::T2, V::Function, nx::T2, ny::T2) where {T1<:AbstractFloat,T2<:Signed}

    # computed with numerical integration using Gauss-Legendre points.
    hx = 2.0 * Lx / Nx
    hy = 2.0 * Ly / Ny
    dof = (Nx - 1) * (Ny - 1)
    mat = zeros(dof, dof)
    # Gauss-Legendre points
    p_x, w_x = gausslegendre(nx)
    p_y, w_y = gausslegendre(ny)

    w_X = [w_x[i] for i = 1:nx for j = 1:ny]
    w_Y = [w_y[j] for i = 1:nx for j = 1:ny]
    wt = @. w_X * w_Y * hx * hy / 4.0

    x = [p_x[i] for i = 1:nx for j = 1:ny]
    y = [p_y[j] for i = 1:nx for j = 1:ny]

    # value and gradient of basis functions at (x,y)
    # the hat functions are separated into 2 parts,
    # and we compute the 4 possibilities of product of 2
    f = hcat((x .- 1.0) .* (y .- 1.0) / 4.0,
        .-(x .+ 1.0) .* (y .- 1.0) / 4.0,
        .-(x .- 1.0) .* (y .+ 1.0) / 4.0,
        (x .+ 1.0) .* (y .+ 1.0) / 4.0)

    for j = 1:Ny #loop over y dof
        yy = .-Ly .+ hy * (j .- 0.5 .+ y / 2) # y coordinates
        ylabel = [j - 1, j - 1, j, j] # label for the dof on the edges
        for i = 1:Nx #loop over x dof
            xx = .-Lx .+ hx * (i .- 0.5 .+ x / 2) # x coordinates
            xlabel = [i - 1, i, i - 1, i] # label for the dof on the edges

            r = V.(xx, yy) #values for V
            # calculate stiff elements
            for k = 1:4
                for l = 1:4
                    ix = xlabel[k]
                    iy = ylabel[k]
                    jx = xlabel[l]
                    jy = ylabel[l]
                    mx = ix + (iy - 1) * (Nx - 1)
                    my = jx + (jy - 1) * (Nx - 1)
                    if ix > 0 && ix < Nx && iy > 0 && iy < Ny && jx > 0 && jx < Nx && jy > 0 && jy < Ny #checking admissible basis functions
                        mat[mx, my] += sum(r .* f[:, k] .* f[:, l] .* wt)
                    end
                end
            end
        end
    end

    return sparse(mat)
end
oneB_V_2d(L::T1, N::T2, V::Function, nx::T2, ny::T2) where {T1<:AbstractFloat,T2<:Signed} = oneB_V_2d(L, L, N, N, V, nx, ny)

#----------------------------------------------------------------------------
# Two-body matrices for a local potential V
#----------------------------------------------------------------------------
function twoB_V_2d_sp(Lx::T1, Ly::T1, Nx::T2, Ny::T2, V::Function, nx1::T2, ny1::T2, nx2::T2, ny2::T2) where {T1<:AbstractFloat,T2<:Signed}
    #for P1 finite elements with Dirichlet boundary conditions
    hx = 2.0 * Lx / Nx
    hy = 2.0 * Ly / Ny
    dof = (Nx - 1) * (Ny - 1)
    I = Int[]
    J = Int[]
    vals = T1[] #initialise values
    label1 = zeros(T2, 4) # save labels and coordinates for 4 points in each element, i,e,(n,x,y)
    label2 = zeros(T2, 4)
    #mat = zeros(dof, dof, dof, dof);

    # Gauss-Legendre points
    gauss_x1, weight_x1 = gausslegendre(nx1)
    gauss_y1, weight_y1 = gausslegendre(ny1)
    gauss_x2, weight_x2 = gausslegendre(nx2)
    gauss_y2, weight_y2 = gausslegendre(ny2)

    siz = nx1 * ny1 * nx2 * ny2
    wt = zeros(siz)
    x1 = zeros(siz)
    y1 = zeros(siz)
    x2 = zeros(siz)
    y2 = zeros(siz)
    i = 1
    for px1 = 1:nx1, py1 = 1:ny1, px2 = 1:nx2, py2 = 1:ny2
        wt[i] = weight_x1[px1] * weight_y1[py1] * weight_x2[px2] * weight_y2[py2] * hx^2 * hy^2 / 16.0
        x1[i] = gauss_x1[px1]
        y1[i] = gauss_y1[py1]
        x2[i] = gauss_x2[px2]
        y2[i] = gauss_y2[py2]
        i += 1
    end

    #f1 = hcat((x1.-1.0).*(y1.-1.0)/4.0, .-(x1.+1.0).*(y1.-1.0)/4.0, .-(x1.-1.0).*(y1.+1.0)/4.0, (x1.+1.0).*(y1.+1.0)/4.0);
    #f2 = hcat((x2.-1.0).*(y2.-1.0)/4.0, .-(x2.+1.0).*(y2.-1.0)/4.0, .-(x2.-1.0).*(y2.+1.0)/4.0, (x2.+1.0).*(y2.+1.0)/4.0);

    f = hcat((x1 .- 1.0) .* (y1 .- 1.0) .* (x2 .- 1.0) .* (y2 .- 1.0) / 16.0, .-(x1 .+ 1.0) .* (y1 .- 1.0) .* (x2 .- 1.0) .* (y2 .- 1.0) / 16.0,
        .-(x1 .- 1.0) .* (y1 .+ 1.0) .* (x2 .- 1.0) .* (y2 .- 1.0) / 16.0, (x1 .+ 1.0) .* (y1 .+ 1.0) .* (x2 .- 1.0) .* (y2 .- 1.0) / 16.0,
        .-(x1 .- 1.0) .* (y1 .- 1.0) .* (x2 .+ 1.0) .* (y2 .- 1.0) / 16.0, (x1 .+ 1.0) .* (y1 .- 1.0) .* (x2 .+ 1.0) .* (y2 .- 1.0) / 16.0,
        (x1 .- 1.0) .* (y1 .+ 1.0) .* (x2 .+ 1.0) .* (y2 .- 1.0) / 16.0, .-(x1 .+ 1.0) .* (y1 .+ 1.0) .* (x2 .+ 1.0) .* (y2 .- 1.0) / 16.0,
        .-(x1 .- 1.0) .* (y1 .- 1.0) .* (x2 .- 1.0) .* (y2 .+ 1.0) / 16.0, (x1 .+ 1.0) .* (y1 .- 1.0) .* (x2 .- 1.0) .* (y2 .+ 1.0) / 16.0,
        (x1 .- 1.0) .* (y1 .+ 1.0) .* (x2 .- 1.0) .* (y2 .+ 1.0) / 16.0, .-(x1 .+ 1.0) .* (y1 .+ 1.0) .* (x2 .- 1.0) .* (y2 .+ 1.0) / 16.0,
        (x1 .- 1.0) .* (y1 .- 1.0) .* (x2 .+ 1.0) .* (y2 .+ 1.0) / 16.0, .-(x1 .+ 1.0) .* (y1 .- 1.0) .* (x2 .+ 1.0) .* (y2 .+ 1.0) / 16.0,
        .-(x1 .- 1.0) .* (y1 .+ 1.0) .* (x2 .+ 1.0) .* (y2 .+ 1.0) / 16.0, (x1 .+ 1.0) .* (y1 .+ 1.0) .* (x2 .+ 1.0) .* (y2 .+ 1.0) / 16.0)
    fw = @. f * sqrt(wt)

    r = zeros(siz)
    ff = zeros(siz)
    xx1 = zeros(siz)
    yy1 = zeros(siz)
    xx2 = zeros(siz)
    yy2 = zeros(siz)
    for nj2 = 1:Ny
        for ni2 = 1:Nx
            for nj1 = 1:Ny
                for ni1 = 1:Nx
                    #----------- label the 4 nodes in each elements ----------
                    #i+(j-1)*(Nx-1)
                    label1[1] = (Nx - 1) * (nj1 - 2) + ni1 - 1#(i-1,j-1)
                    label1[2] = label1[1] + 1#(i,j-1)
                    label1[3] = (Nx - 1) * (nj1 - 1) + ni1 - 1#(i-1,j)
                    label1[4] = label1[3] + 1#(i,j)
                    label2[1] = (Nx - 1) * (nj2 - 2) + ni2 - 1#(i-1,j-1)
                    label2[2] = label2[1] + 1#(i,j-1)
                    label2[3] = (Nx - 1) * (nj2 - 1) + ni2 - 1#(i-1,j)
                    label2[4] = label2[3] + 1#(i,j)
                    if nj1 == 1
                        label1[1] = -1
                        label1[2] = -1
                    end
                    if nj1 == Ny
                        label1[3] = -1
                        label1[4] = -1
                    end
                    if ni1 == 1
                        label1[1] = -1
                        label1[3] = -1
                    end
                    if ni1 == Nx
                        label1[2] = -1
                        label1[4] = -1
                    end
                    if nj2 == 1
                        label2[1] = -1
                        label2[2] = -1
                    end
                    if nj2 == Ny
                        label2[3] = -1
                        label2[4] = -1
                    end
                    if ni2 == 1
                        label2[1] = -1
                        label2[3] = -1
                    end
                    if ni2 == Nx
                        label2[2] = -1
                        label2[4] = -1
                    end

                    # coordinates
                    @. xx1 = -Lx + hx * (ni1 - 0.5 + x1 / 2)
                    @. yy1 = -Ly + hy * (nj1 - 0.5 + y1 / 2)
                    @. xx2 = -Lx + hx * (ni2 - 0.5 + x2 / 2)
                    @. yy2 = -Ly + hy * (nj2 - 0.5 + y2 / 2)
                    #@. r = 1 / sqrt((xx1 - xx2)^2 + (yy1 - yy2)^2)
                    @. r = V(xx1 - xx2, yy1 - yy2)

                    for k1 = 1:4, k2 = 1:4, l1 = 1:4, l2 = 1:4
                        m1 = label1[k1]
                        m2 = label2[k2]
                        n1 = label1[l1]
                        n2 = label2[l2]
                        if m1 > 0 && m2 > 0 && n1 > 0 && n2 > 0
                            push!(I, (m2 - 1) * dof + m1)
                            push!(J, (n2 - 1) * dof + n1)
                            @views f1 = fw[:, k1+(k2-1)*4]
                            @views f2 = fw[:, l1+(l2-1)*4]
                            push!(vals, dot(r, broadcast!(*, ff, f1, f2)))
                            #mat[m1,m2,n1,n2] = mat[m1,m2,n1,n2] + dot(r,broadcast!(*,ff,f1,f2))
                        end
                    end
                end
            end
        end
    end

    return sparse(I, J, vals)
end
twoB_V_2d_sp(L::T1, N::T2, V::Function, nx1::T2, ny1::T2, nx2::T2, ny2::T2) where {T1<:AbstractFloat,T2<:Signed} = twoB_V_2d_sp(L, L, N, N, V, nx1, ny1, nx2, ny2)

# generate 1-body and 2-body matrices
function matrix_1_2body_2d(Lx::T1, Ly::T1, Nx::T2, Ny::T2, vext::Function, vee::Function, nx1::Int64, ny1::Int64, nx2::Int64, ny2::Int64) where {T1<:AbstractFloat,T2<:Signed}
    # construct the 1-body matrix for -1/2⋅Δ, external potential and overlap
    AΔ = oneB_lap_2d(Lx, Ly, Nx, Ny)
    AV = oneB_V_2d(Lx, Ly, Nx, Ny, vext, nx1, nx2)
    C = oneB_over_2d(Lx, Ly, Nx, Ny)
    # construct the 2-body matrix for electron-electron interaction v_ee
    B1 = twoB_V_2d_sp(Lx, Ly, Nx, Ny, vee, nx1, ny1, nx2, ny2)
    B2 = twoB_V_2d_sp(Lx, Ly, Nx, Ny, vee, nx2, ny2, nx1, ny1)
    B = (B1 + B2) / 2
    return AΔ, AV, C, B
end
matrix_1_2body_2d(L::Float64, N::Int64, vext::Function, vee::Function, nx1::Int64, ny1::Int64, nx2::Int64, ny2::Int64) = matrix_1_2body_2d(L, L, N, N, vext, vee, nx1, ny1, nx2, ny2)
