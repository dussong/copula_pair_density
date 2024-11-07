using Interpolations
export Interp2D

# interpolation for 2d pictures
function Interp2D(x, y, data, factor)

    IC = CubicSplineInterpolation((axes(data, 1), axes(data, 2)), data)

    xx = LinRange(x[1], x[end], length(x) * factor)
    yy = LinRange(y[1], y[end], length(y) * factor)
    finerx = LinRange(firstindex(data, 1), lastindex(data, 1), size(data, 1) * factor)
    finery = LinRange(firstindex(data, 2), lastindex(data, 2), size(data, 2) * factor)
    nx = length(finerx)
    ny = length(finery)

    data_interp = Array{Float64}(undef, nx, ny)
    for i ∈ 1:nx, j ∈ 1:ny
        data_interp[i, j] = IC(finerx[i], finery[j])
    end

    return xx, yy, data_interp

end