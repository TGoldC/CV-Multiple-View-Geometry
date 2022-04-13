function result = addprimes( s, e )
    % Store all numbers between s and e in vector
    z = s:e;
    
    % isprime(z) returns a logical mask for which numbers are prime
    % numbers
    % z(isprime(z)) uses this mask to index z
    result = sum(z(isprime(z)));
end

