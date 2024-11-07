
export inv_power

function cg_solve(ne::Int, A::SparseMatrixCSC{Float64,Int64},x::Array{Float64,1},
         f::Array{Float64,1})
    # sovle the liner system A*x = f using a cg iteration method
    # note that A shall be symmetric

    tol = 1.0e-8
    max_iter = 200;

    res = f - A * x;
    rr1 = 1.
    rr0 = 1.
    p = copy(res)
    ap = copy(res)
    for k = 1:max_iter
        e = norm(res)
        if e < tol
            break;
        end

        beta = rr1/rr0;
        @. p = res + beta * p;

        rr0 = dot(res,res)#res' * res;
        mul!(ap,A,p)
        pap = dot(p,ap)#p' * ap;
        rf = rr0/pap;

        @. x = x + rf * p;
        @. res = res - rf * ap;
        rr1 = dot(res,res)#res' * res;
    end

    return x
end


function  inv_power(ne::Int, A::SparseMatrixCSC{Float64,Int64},
          B::SparseMatrixCSC{Float64,Int64};max_iter = 300, tol = 1.0e-8)
    # solve the generalized eigenvalue problem Ax=λBx
    # since we want the lowest eigenvalue, a inverse power method is used

    e = 1.0;
    # give an initial vector for the iteration

    n = A.n
    vec = ones(n);
    x = ones(n);
    y = ones(n);
    val = 10.0;

    # start power iteration
    for k = 1:max_iter
        if e < tol
            println("step = $(k), res = $(e), eig = $(1.0/val) \n")
            break;
        end
        # for a  power method , we shall do  y = A * vec;
        # here we imply y = A^{-}B * x for a inverse power method
        @. x = 1.
        mul!(y,B,vec)
        y = cg_solve(ne, A, x, y);

        e = abs(norm(vec) - val);
        val = norm(vec);
        @. vec = y/val;

        k % 10 == 0 && (println("step = $(k), res = $(e), eig = $(1.0/val) \n"))
    end

    vec = vec/norm(vec);
    val = 1.0/val;      # remember it is a inverse method
    res = e;        # return the error

    return val, vec, res
end
