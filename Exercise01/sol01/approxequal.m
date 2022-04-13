function out = approxequal( x, y, eps )
    % several options:
    % Default dimension for sum and max is 1, i.e. along the columns
    out = all(abs(x-y) < eps) % or
    out = sum(abs(x-y) >= eps) == 0 % or
    out = max(abs(x-y)) < eps % or
    out = max( (x-y) .* (x-y) ) < eps*eps
end

