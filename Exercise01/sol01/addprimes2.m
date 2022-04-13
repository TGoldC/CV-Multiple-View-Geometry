function result = addprimes2( s, e )
    % Store all numbers between s and e in vector
    z = s:e;
    
    % isprime(z) returns a logical mask for which numbers are prime numbers
    % and which are not
    % isprime(z) .* z interprets the logical mask as 0s and 1s to set all
    % nonprimes to zero
    result = sum(isprime(z) .* z);
end

