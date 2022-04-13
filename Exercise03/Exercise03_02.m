%% 
%
% File:     Exercise03_XinCheng.m
% Author:   Xin Cheng
% Date:     08.05.2021
% Comment:  This File is the solution of the second exercise in exercise 03 in Computer Vision Course.
%
%
%% setup

% try large and small vector (to test Taylor expansion for small angles)

w1 = rand(3,1);       % 这两个是正常的例子
v1 = rand(3,1);

w2 = w1/norm(w1) * 1e-6;  % 这两个是无穷小的例子
v2 = v1/norm(v1) * 1e-6;

xi1 = [v1;w1];
xi2 = [v2;w2];


%% SO(3)

% compute exponential
R1 = Exp_SO3(w1);         % 大值和小值 用公式 exp[w_hat] = I + sin(|w|)/|w| * w_hat + (1-cos(|w|))/|w| * w_hat^2 算出来的R
R2 = Exp_SO3(w2);

% compute logarithm to get back to the original value
w1_b = Log_SO3(R1);       % 在算出来的R上，往回计算，得出 两个w的back
w2_b = Log_SO3(R2,0);

% compare original w with log(exp(w)) --> compute relative difference
% we expect small values (1e-10 or smaller)
test_exp_log_so3_1 = norm(w1 - w1_b) / norm(w1);       % 期望得到的是：w1和w1_back差不多大，相对误差很小
test_exp_log_so3_2 = norm(w2 - w2_b) / norm(w2);

% Compare to Matlab's generic expm function --> compute relative difference
% we expect small values (1e-10 or smaller)
test_exp_so3_1 = norm(R1 - expm(hat_SO3(w1))) / norm(w1);
test_exp_so3_2 = norm(R2 - expm(hat_SO3(w2))) / norm(w2);
test_log_so3_1 = norm(w1_b - vee_SO3(logm(R1))) / norm(w1);
test_log_so3_2 = norm(w2_b - vee_SO3(logm(R2))) / norm(w2);

%% SE(3)

% compute exponential
T1 = Exp_SE3(xi1);
T2 = Exp_SE3(xi2);

% compute logarithm to get back to the original value
xi1_b = Log_SE3(T1);
xi2_b = Log_SE3(T2);

% compare original w with log(exp(w)) --> compute relative difference
% we expect small values (1e-10 or smaller)
test_exp_log_se3_1 = norm(xi1 - xi1_b) / norm(xi1);
test_exp_log_se3_2 = norm(xi2 - xi2_b) / norm(xi2);

% Compare to Matlab's generic expm function --> compute relative difference
% we expect small values (1e-10 or smaller)
test_exp_se3_1 = norm(T1 - expm(hat_SE3(xi1))) / norm(xi1);
test_exp_se3_2 = norm(T2 - expm(hat_SE3(xi2))) / norm(xi2);
test_log_se3_1 = norm(xi1_b - vee_SE3(logm(T1))) / norm(xi1);
test_log_se3_2 = norm(xi2_b - vee_SE3(logm(T2))) / norm(xi2);


%% difine some basic functions
function S = hat_SO3(w)                   % w-->w_hat
    S = [  0   -w(3)  w(2)
          w(3)   0   -w(1)
         -w(2)  w(1)   0  ];
end
 
function w = vee_SO3(S)                   % % w_hat-->w  逆运算
    % check S is skew symmetric
    assert(norm(S+S') < 1e-10)
    
    w = [S(3,2); S(1,3); S(2,1)];
end

function Xi = hat_SE3(xi)                 % 见18页
    v = xi(1:3);
    w = xi(4:6);
    Xi = zeros(4);
    Xi(1:3,1:3) = hat_SO3(w);
    Xi(1:3,4) = v;
end

function xi = vee_SE3(Xi)
    S = Xi(1:3,1:3);
    v = Xi(1:3,4);
    
    % check last row is 0
    assert(norm(Xi(4,:)) < 1e-10);
    
    % this checks if S is skew symmetric
    w = vee_SO3(S);
    
    xi = [v; w];
end
%% define Expoential and logarithm functions
function R = Exp_SO3(w, epsilon)                                           % Exponential for so(3) --> SO(3)
                                                                           % epsilon 就是一个阈，当w的norm小于epsilon时，视为 无穷小。这时候 泰勒展开就变化了
    if ~exist('epsilon','var')
        % set default value for epsilon
        epsilon = 1e-5;
    end
    S = hat_SO3(w);
    theta_2 = w'*w;
    theta = sqrt(theta_2);                                                 % theta 就是 ||w||
    % Use Taylor expansion to avoid numerical instabilities for small theta
    % At least a check for theta != 0 would be needed.
    if theta <= epsilon
        % http://www.wolframalpha.com/input/?i=sin(theta)%2Ftheta          % 泰勒展开见连接 
        A = 1 - theta_2/6;
        % https://www.wolframalpha.com/input/?i=(1+-+cos(theta))%2Ftheta%5E2
        B = 0.5 - theta_2/24;
    else
        A = sin(theta)/theta;                                              % 这里的A 是 sin(||w||)/||w||
        B = (1 - cos(theta))/theta_2;
    end
    % Rodrigues' formula
    R = eye(3) + A*S + B*S*S;                                              % exp[w_hat] = I + sin(|w|)/|w| * w_hat + (1-cos(|w|))/|w| * w_hat^2
end

function w = Log_SO3(R, epsilon)
    if ~exist('epsilon','var')
        % set default value for epsilon
        epsilon = 1e-5;
    end
    theta = acos((trace(R)-1)/2);
    % Use Taylor expansion to avoid numerical instabilities for small theta
    % At least a check for theta != 0 would be needed.
    if theta <= epsilon
        % http://www.wolframalpha.com/input/?i=theta+%2F+sin(theta)
        A = 1 + (theta^2)/6;
    else
        A = theta / sin(theta);
    end
    w = 0.5 * A * vee_SO3(R-R');
end

function T = Exp_SE3(xi, epsilon)
    if ~exist('epsilon','var')
        % set default value for epsilon
        epsilon = 1e-5;
    end
    v = xi(1:3);
    w = xi(4:6);
    S = hat_SO3(w);
    S_2 = S*S;
    theta_2 = w'*w;
    theta = sqrt(theta_2);
    % Use Taylor expansion to avoid numerical instabilities for small theta
    % At least a check for theta != 0 would be needed.
    if theta <= epsilon
        % http://www.wolframalpha.com/input/?i=sin(theta)%2Ftheta
        A = 1 - theta_2/6;
        % https://www.wolframalpha.com/input/?i=(1+-+cos(theta))%2Ftheta%5E2
        B = 0.5 - theta_2/24;
        % http://www.wolframalpha.com/input/?i=(1+-+sin(theta)%2Ftheta)+%2F+theta%5E2
        C = 1/6 - theta_2/120;
    else
        A = sin(theta)/theta;
        B = (1 - cos(theta))/theta_2;
        C = (1 - A) / theta_2;
    end
    R = eye(3) + A*S + B*S_2;  % Rodrigues' formula
    V = eye(3) + B*S + C*S_2;
    T = eye(4);
    T(1:3,1:3) = R;
    T(1:3,4) = V*v;
end

function xi = Log_SE3(T, epsilon)
    if ~exist('epsilon','var')
        % set default value for epsilon
        epsilon = 1e-5;
    end
    R = T(1:3,1:3);
    t = T(1:3, 4);
    assert(norm(T(4,:) - [0 0 0 1]) < 1e-10); % 断言 是一个 skew-system
    theta = acos((trace(R)-1)/2);       % theta就是||w||
    theta_2 = theta^2;                  % theta_2就是|w|^2
    % Use Taylor expansion to avoid numerical instabilities for small theta
    % At least a check for theta != 0 would be needed.
    if theta <= epsilon
        % http://www.wolframalpha.com/input/?i=theta+%2F+sin(theta)
        A = 1 + (theta_2)/6;
        % http://www.wolframalpha.com/input/?i=(1+-+(sin(theta)%2Ftheta)%2F(2*(1+-+cos(theta))%2Ftheta%5E2))%2Ftheta%5E2
        D = 1/12 - theta_2/720;
    else
        A = theta / sin(theta);
        A_temp = sin(theta)/theta;
        B_temp = (1 - cos(theta))/theta_2;
        D = (1 - A_temp/(2*B_temp))/theta_2;
    end
    S = 0.5 * A * (R-R');
    w = vee_SO3(S);
    S_2 = S*S;
    % Note: It is not obvious to derive the closed-form solution of V, and
    % one possibility is to use Matlab's general matrix inverse inv(V)
    % to compute it (but the closed-form solution is probably faster).
    V_inv = eye(3) - 0.5*S + D*S_2;
    v = V_inv * t;
    xi = [v; w];
end
