% solve for the best fitting linear function of several variables 
% through a set of points using the Matlab function svd
  

%% create the data

% generate data set consisting of m samples of each of 4 variables
m = 3;
d1 = rand(m,1);
d2 = rand(m,1);
d3 = rand(m,1);
d4 = 4*d1 - 3*d2 + 2*d3 - 1;

% introduce small errors into the data
% (the 'rand' function produces numbers between 0 and 1)
eps = 1.e-4;
d1 = d1 .* (1 + eps * rand(m,1));
d2 = d2 .* (1 + eps * rand(m,1));
d3 = d3 .* (1 + eps * rand(m,1));
d4 = d4 .* (1 + eps * rand(m,1));

%% find the coefficients x solving the system: x1*d1 + x2*d2 + x3*d3 + x4*d4 = 1;

% define A and compute the svd
D = [d1 d2 d3 d4];
[U, S, V] = svd(D);

% construct b and S^+
S_plus = S';
S_plus(S_plus ~= 0) = S_plus(S_plus ~= 0) .^ -1;
b = ones(m,1);

% solve the least squares problem using the pseudo inverse D_plus = V * S_plus * U'
D_plus = V * S_plus * U';
D_plus_matlab = pinv(D);
disp(abs(D_plus - D_plus_matlab)) % allow to compare our pseudo inverse and the one using matlab pinv function

x = D_plus * b;
disp(x)


%% When m = 3, display the graph of the norm of x_lambda and the error according to lambda to check if the pseudo inverse solution has the minimal norm among all solutions.
if (m==3)
    
    v = null(D); % get a vector of the kernel of D
    norm(v)
    lambda = -100:1:100;
    nb_values = length(lambda);
    values_norm = zeros(nb_values,1);
    values_error = zeros(nb_values,1);
    
    for i = 1:nb_values
        x_lambda = x + lambda(i)*v;
        values_norm(i) = norm(x_lambda);
        values_error(i) = norm(D*x_lambda - b)^2;
    end
    
    %Display the graph, we notice that all x_lambda have the same error, and for lambda = 0, we have the smallest
    %norm possible, which was expected.
    figure(1)
    plot(lambda,values_norm,'b-',lambda,values_error,'r-');
    
end

%% Notes:

% In Matlab you would usually solve a linear system with the built-in solver:
% x = D\b;
