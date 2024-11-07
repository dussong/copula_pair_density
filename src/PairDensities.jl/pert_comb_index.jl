export seq2num, num2seq, num2seq!, seq2num_ns, num2seq_ns, num_ns2s, num_s2ns


function seq2num(m::Int64,n::Int64,seq::Vector{Int64})
    # Map [combination sequence] to [sequence number] of all combinations of C(m,n)

    num = seq[n]-seq[n-1]

    for j = 1:seq[1]-1
        num += binomial(m-j,n-1)
    end

    for i = 1:n-2
        for j = (seq[i]+1):(seq[i+1]-1)
            num += binomial(m-j,n-i-1)
        end
    end

    return num;
end

function num2seq(m::Int64,n::Int64,num::Int64)
    # Map [sequence number] of a combination of C(m,n) to [combination sequence]

    seq = collect(1:n);
    firstPoint = 1;
    for i = 1:n-1
        s1 = 0

        if num == 1
            if i > 1
                seq[i:n] = collect(seq[i-1]+1:seq[i-1]+n-i+1)
            end
            break;
        end

        for j = firstPoint:m
            s2 = s1 + binomial(m-j,n-i)
            if num<=s2 && num>s1
                seq[i] = j
                break;
            end
            s1 = s2
        end

        num -= s1
        firstPoint = seq[i] + 1
    end

    seq[n] = seq[n-1] + num

    return seq;
end

function num2seq!(seq::Vector{Int64},m::Int64,n::Int64,num::Int64)
    # Map [sequence number] of a combination of C(m,n) to [combination sequence]

    for i = 1:n
        seq[i] = i
    end
    firstPoint = 1;
    for i = 1:n-1
        s1 = 0

        if num == 1
            if i > 1
                seq[i:n] = collect(seq[i-1]+1:seq[i-1]+n-i+1)
            end
            break;
        end

        for j = firstPoint:m
            s2 = s1 + binomial(m-j,n-i)
            if num<=s2 && num>s1
                seq[i] = j
                break;
            end
            s1 = s2
        end

        num -= s1
        firstPoint = seq[i] + 1
    end

    seq[n] = seq[n-1] + num

    return seq;
end

function seq2num_ns(m::Int64,n::Int64,seq::Vector{Int64})

    num = seq[1]
    for i = 2 : n
        num += (seq[i]-1) * m^(i-1)
    end

    return num;
end

function num2seq_ns(m::Int64,n::Int64,num::Int64)
    # Map [sequence number] of a combination of m^n to [combination sequence]

    seq = zeros(Int,n);
    num = num - 1
    #=
    for i = 1 : n
        seq[i] = num % m + 1
        num = (num - seq[i] + 1)/m
    end
    =#
    for i = n:-1:2
        seq[i] = fld(num, m^(i-1)) + 1
        num = num - (seq[i]-1) * m^(i-1)
    end
    seq[1] = num + 1

    return seq;
end

num_s2ns(m::Int64, n::Int64, num::Int64) = seq2num_ns(m, n, num2seq(m, n, num))

num_ns2s(m::Int64, n::Int64, num::Int64) = seq2num(m, n, num2seq_ns(m, n, num))


    
