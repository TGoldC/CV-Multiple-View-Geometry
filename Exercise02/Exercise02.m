%% 
%
% File:     Exercise02_XinCheng.m
% Author:   Xin Cheng
% Date:     01.05.2021
% Comment:  This File is the solution of exercise 02 in Computer Vision Course.
%
%

%% Creat some data
m = 4;
D = rand(m,4);
D(:,4) = 1;
d4 = D * [4 -3 2 -1].';
D(:,4) = d4;

% introduce small noise
eps = 1.e-4;
D = (1+eps) * rand(m,4);
D(:,4) = 1;
d4 = D * [4 -3 2 -1].';
D(:,4) = d4;

%% Find the coefficients x solving the system Dx = b
[U,S,V] = svd(D);
b = ones(4,1);

S_inv = S';
S_inv(S_inv ~= 0) = S_inv(S_inv ~= 0) .^ -1;
D_MPpi = V * S_inv * U.';
x_svd = D_MPpi * b;

D_pinv = pinv(D);
x_pinv = D_pinv * b;
% solutions x from svd and directly from function pinv are same [4 -3 2 -1].'

%% Repeat the two previous questions, by setting m to a higher value. How is the precision impacted?
m = 10;
% introduce small noise
eps = 1.e-4;
D = (1+eps) * rand(m,4);
D(:,4) = 1;
d4 = D * [4 -3 2 -1].';
D(:,4) = d4;

[U,S,V] = svd(D);
b = ones(m,1);

S_inv = S';
S_inv(S_inv ~= 0) = S_inv(S_inv ~= 0) .^ -1;
D_MPpi = V * S_inv * U.';
x_svd = D_MPpi * b;

D_pinv = pinv(D);
x_pinv = D_pinv * b;
% good precise.

%% We assume in the following that m = 3, hence we have infinitely many solutions
m = 3;
% introduce small noise
eps = 1.e-4;
D = (1+eps) * rand(m,4);
D(:,4) = 1;
d4 = D * [4 -3 2 -1].';
D(:,4) = d4;

[U,S,V] = svd(D);
b = ones(m,1);

S_inv = S';
S_inv(S_inv ~= 0) = S_inv(S_inv ~= 0) .^ -1;
D_MPpi = V * S_inv * U.';
x_svd = D_MPpi * b;

D_pinv = pinv(D);
x_pinv = D_pinv * b;
% there are many different solutions.

%% Use the function null to get a vector v belongs to kernel(D).
v = null(D);
lambda = -100:1:100;
norm_val = zeros(length(lambda),1);
error_val = zeros(length(lambda),1);

for i = 1:length(lambda)
    x_lambda = x_pinv + lambda(i) * v;
    norm_val(i) = norm(x_lambda);
    error_val(i) = norm(D * x_lambda - b)^2;
end
figure
plot(lambda, norm_val)
%hold on
figure
plot(lambda, error_val)
%hold off
grid on
xlabel('Lambda')
ylabel('Norm of x & Error')
%legend('norm of x','associated error')

x_pinv_norm = norm(x_pinv);
