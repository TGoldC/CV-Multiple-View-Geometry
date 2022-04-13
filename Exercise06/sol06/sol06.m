% Notes:
% - You will not see that rank(chi)==8 for real data, as the noise destroys
%   the nullspace. However, the rank will be 8 with perfect simulated data.
% - The algorithm is not robust if the correspondences are inaccurate. For
%   the given images, a set of predefined correspondences is given, for
%   which the algorithm should work. To hand-select points, it may be more
%   robust to use another image provided by matlab (see below).
% - If you hand-select points, do it in the same order for both images,
%   such that corresponding points have the same index

close all;
clear all;

%% obtain correspondences

% Method for selecting corresponding point pairs. (un)comment as needed
method = "predefined"; % use predefined correspondences 表示事先定义好 两个图中corresponding的点的坐标了
%method = "select"; % select correspondences by hand   表示人工选择 两个图中corresponding的点的坐标了
%method = "simulate"; % simulate perfect correspondences (for debugging)

% Choose to use the images given with the exercise, or alternative ones
% provided by matlab. (un)comment as needed
% (no effect if method = "simulate")
images = "given";
%images = "matlab";

if strcmp(method, "simulate")           % 字符串比较，相同出1
    % Random ground-truth pose
    R_gt = expm(hat(rand(3,1)*0.1));
    T_gt = rand(3,1);
    
    % camera intrinsics
    K1 = eye(3);
    K2 = eye(3);
    
    % generate random 3D points
    nPoints = 12;
    % for corresponding pixel coordinates
    x1 = zeros(nPoints,1);
    y1 = zeros(nPoints,1);
    x2 = zeros(nPoints,1);
    y2 = zeros(nPoints,1);
    % for ground-truth 3D points
    X1_gt = zeros(3, nPoints);
    n = 0;
    while n < nPoints
        X1 = (rand(3, 1) - 0.5) * 20;
        X2 = R_gt*X1 + T_gt;
        % only accept if they are in front of both cameras
        if X1(3) > 0.1 && X2(3) > 0.1
            n = n+1;
            xy1 = K1 * X1 / X1(3);
            x1(n) = xy1(1);
            y1(n) = xy1(2);
            xy2 = K2 * X2 / X2(3);
            x2(n) = xy2(1);
            y2(n) = xy2(2);
            X1_gt(:, n) = X1;
        end
    end
    
 % use real images
else
    if strcmp(images, "matlab") % 当image选择为 matlab时，有matlab自带的vision工具来生成images
        imageDir = fullfile(toolboxdir('vision'), 'visiondata','upToScaleReconstructionImages');
        % fullfile来组合文件的路径；toolboxdir的工具箱的根文件夹的绝对路径
        images = imageDatastore(imageDir);
        % imds = imageDatastore(location) 根据 location 指定的图像数据集合创建一个数据存储 imds。
        I1 = readimage(images, 1);
        % img = readimage(imds,I) 从数据存储 imds 读取第 I 个图像文件并返回图像数据 img
        I2 = readimage(images, 2);
        load upToScaleReconstructionCameraParameters.mat
        image1 = undistortImage(I1, cameraParams);
        image2 = undistortImage(I2, cameraParams);
        K1 = cameraParams.IntrinsicMatrix';
        K2 = K1;
        if strcmp(method, "predefined")
            [x1,y1,x2,y2] = getpoints_predefined_matlabimage();
        end
    elseif strcmp(images, "given")      % 当设置为 given，就是用两个文件夹里已给的图

        % Read input images:
        image1 = double(imread('batinria0.tif'));
        image2 = double(imread('batinria1.tif'));


        % Intrinsic camera parameter matrices
        K1 = [844.310547 0 243.413315; 0 1202.508301 281.529236; 0 0 1];
        K2 = [852.721008 0 252.021805; 0 1215.657349 288.587189; 0 0 1];
        
        if strcmp(method, "predefined")
            [x1,y1,x2,y2] = getpoints_predefined_givenimage();
        end
    end

    % choose corresponding points by hand
    if strcmp(method, "select")
        nPoints = 12; %or 8
        [x1,y1,x2,y2] = getpoints(image1,image2,nPoints); % 创建的函数，在两个图中，人工选取 12个corresponding的点

    % use predefined points
    elseif strcmp(method, "predefined") 
        nPoints = length(x1);
        % Show the selected points 标出predefined 的 corresponding的点
        imshow(uint8(image1))
        hold on
        plot(x1, y1, 'r+')
        hold off
        figure
        imshow(uint8(image2))
        hold on
        plot(x2, y2, 'r+')
        hold off
    end
end
    

%% Pose and structure reconstruction

% Transform image coordinates with inverse camera matrices:
xy1 = K1 \ [x1'; y1'; ones(1,nPoints)]; % x1和y1都是 12*1 的列向量;等式右边是一个 3*12的横向量；K1 * xy1 = [3*12];最后得到的xy1 是一个 3 * 12的数组
xy2 = K2 \ [x2'; y2'; ones(1,nPoints)];
% 给的x1和y1 是在各自的image coordinate坐标系下的坐标，所以为啥要 在用K1和K2转化一下呢
x1 = xy1(1,:);
y1 = xy1(2,:);
x2 = xy2(1,:);
y2 = xy2(2,:);

% Compute constraint matrix chi:  见第5张第14页 八点算法
chi = zeros(nPoints,9);
for i = 1:nPoints
    chi(i,:) = kron([x1(i) y1(i) 1],[x2(i) y2(i) 1]); % 见第5章P11，chi就是 花体X；
end

rank_chi = rank(chi);

% Find minimizer for |chi*E|:
[UChi,DChi,VChi] = svd(chi);

% Unstacked ninth column of V:
E = reshape(VChi(:,9),3,3); % 使得|chi*E|最小的E，是V的第九列

% SVD of E
[U,D,V] = svd(E);
if det(U) < 0
    U = -U;
end
if det(V) < 0
    V = -V;
end

% Project E onto essential space (replace eigenvalues):
D(1,1) = 1;       % 把sigularvalues改成 1 1 0
D(2,2) = 1;
D(3,3) = 0;

% Final essential matrix:
E = U * D * V'; % 得到最后的essential matrix

% Recover R and T from the essential matrix E:
% (Compare Slides)
Rz1 = [0 1 0; -1 0 0; 0 0 1]'; % 给出Rz是 绕Z轴旋转 +-90° 的旋转矩阵
Rz2 = [0 -1 0; 1 0 0; 0 0 1]';
R1 = U * Rz1' * V';
R2 = U * Rz2' * V';
T_hat1 = U * Rz1 * D * U';
T_hat2 = U * Rz2 * D * U';

% 得到R和T

% Translation belonging to T_hat
T1 = [ -T_hat1(2,3); T_hat1(1,3); -T_hat1(1,2) ];
T2 = [ -T_hat2(2,3); T_hat2(1,3); -T_hat2(1,2) ];


% Compute scene reconstruction and correct combination of R and T:
n_success = 0; % 讨论4中情况的成功与否，这个变量表示成功的次数
[success, X1, ~] = reconstruction(R1,T1,x1,y1,x2,y2,nPoints);
if success
    n_success = n_success + 1;
    R_result = R1;
    T_result = T1;
    X1_result = X1;
end
[success, X1, ~] = reconstruction(R1,T2,x1,y1,x2,y2,nPoints);
if success
    n_success = n_success + 1;
    R_result = R1;
    T_result = T2;
    X1_result = X1;
end
[success, X1, ~] = reconstruction(R2,T1,x1,y1,x2,y2,nPoints);
if success
    n_success = n_success + 1;
    R_result = R2;
    T_result = T1;
    X1_result = X1;
end
[success, X1, ~] = reconstruction(R2,T2,x1,y1,x2,y2,nPoints);
if success
    n_success = n_success + 1;
    R_result = R2;
    T_result = T2;
    X1_result = X1;
end

switch n_success
    case 1
        disp("Success! Found one valid solution.")
    case 0
        disp("No valid solution found.")
    otherwise
        disp("Multiple valid solutions found.")
end

% for simulated correspondences, compare result to ground truth
if strcmp(method, "simulate")
    scale_factor = norm(T_gt) / norm(T_result);
    T_scaled = T_result * scale_factor;
    X1_scaled = X1_result * scale_factor;
    error_pose = norm(logm([R_result, T_scaled; 0,0,0,1]^-1 * [R_gt, T_gt; 0,0,0,1]))
    error_points = norm(X1_gt - X1_scaled)
end
    

%% function for structure reconstruction

% success specifies whether all depth values are positive
function [success, X1, x2] = reconstruction(R,T,x1,y1,x2,y2,nPoints)
% 见第5章18页
% Structure reconstruction matrix M:
M = zeros(3*nPoints, nPoints + 1);    % M是 36*13
for i = 1:nPoints
   x2_hat = hat([x2(i) y2(i) 1]);
   
   M(3*i-2 : 3*i, i) = x2_hat*R*[x1(i); y1(i); 1];
   M(3*i-2 : 3*i, nPoints+1) = x2_hat*T;
end

% Get depth values (eigenvector to the smallest eigenvalue of M'M):
[V,D] = eig(M' * M); % 返回特征值的对角矩阵 D 和矩阵 V，其列是对应的右特征向量

lambda1 = V(1:nPoints, 1); % V的（~，1）代表的是最小的eigenvector对应的 lambda
gamma  = V(nPoints + 1, 1); % 设||T||为1，gamma就是scale for T

% Gamma has to be positive. If it is negative, use (-lambda1, -gamma), as
% it also is a solution for the system solved by (lambda1, gamma)
if gamma < 0
    gamma = -gamma;
    lambda1 = -lambda1;
end

% make lambda1 have the same scale as T
lambda1 = lambda1 / gamma;  % lambda1是12*1,每个数表示 对于12个点的 分别的depth

% generate 3D points
X1 = [x1; y1; ones(1, nPoints)] .* repmat(lambda1', 3, 1); % 把lambda1' 复制3*1 = 3次，然后按3*1来平铺；
% X1是12*3的数组，一共12个点，每一行是这个点的在 3D坐标，第三列是z坐标，也就是等于lambda1
X2 = R * X1 + repmat(T, 1, nPoints); % 经过R和T之后的X2的坐标

lambda2 = X2(3,:); % X2的z坐标，即为lambda2

% Determine if we have the correct combination of R and T:
n_positive_depth1 = sum(lambda1 >= 0) % lambda1中有几个为 正，表示depth，一定要是正的
n_positive_depth2 = sum(lambda2 >= 0) % lambda2中有几个为 正
if n_positive_depth1==nPoints && n_positive_depth2==nPoints % 若lambda1和lambda2中 正的距离 的个数都等于 12个点，那么这一组R和T 是合格的
    
    % show results
    R
    T
    X1
    
    figure
    
    plot3(X1(1,:), X1(2,:), X1(3,:), 'b+') % 画出 X1点
    
    hold on
    
    plotCamera('Location',[0 0 0],'Orientation',eye(3),'Opacity',0, 'Size', 0.2, 'Color', [1 0 0]) % red 画出两个 camera
    plotCamera('Location', -R'*T,'Orientation',R,'Opacity',0, 'Size', 0.2, 'Color', [0 1 0]) % green
    
    axis equal
    grid on
    xlabel('x')
    ylabel('y')
    zlabel('z')
    
    hold off
    
    success = true;
else
    success = false;
end

end


%% Helper functions

% ================
% Hat-function
function A = hat(v)
    A = [0 -v(3) v(2) ; v(3) 0 -v(1) ; -v(2) v(1) 0];
end



% ================
% function getpoints
function [x1,y1,x2,y2] = getpoints(image1,image2,nPoints)

x1 = zeros(nPoints,1);
y1 = zeros(nPoints,1);
x2 = zeros(nPoints,1);
y2 = zeros(nPoints,1);

% Click points in image1:
% Can be done without for-loop: ginput(nPoints)
figure; imshow(uint8(image1));
hold on;
for i = 1:nPoints
    [x,y] = ginput(1); % ginput自带函数，鼠标点到的点，可以自动获取其坐标，每个loop读取一个数
    x1(i) = double(x);
    y1(i) = double(y);
    plot(x, y, 'r+');
end
hold off;


% Click points in image2:
figure; imshow(uint8(image2));
hold on;
for i = 1:nPoints
    [x,y] = ginput(1);
    x2(i) = double(x);
    y2(i) = double(y);
    plot(x, y, 'r+');
end
hold off;

end



% ================
% predefined points for the given images
function [x1,y1,x2,y2] = getpoints_predefined_givenimage()

x1 = [
   10.0000
   92.0000
    8.0000
   92.0000
  289.0000
  354.0000
  289.0000
  353.0000
   69.0000
  294.0000
   44.0000
  336.0000
  ];

y1 = [ 
  232.0000
  230.0000
  334.0000
  333.0000
  230.0000
  278.0000
  340.0000
  332.0000
   90.0000
  149.0000
  475.0000
  433.0000
    ];
 
x2 = [
  123.0000
  203.0000
  123.0000
  202.0000
  397.0000
  472.0000
  398.0000
  472.0000
  182.0000
  401.0000
  148.0000
  447.0000
    ];

y2 = [ 
  239.0000
  237.0000
  338.0000
  338.0000
  236.0000
  286.0000
  348.0000
  341.0000
   99.0000
  153.0000
  471.0000
  445.0000
    ];

end

% ================
% predefined points for the images provided by matlab
function [x1,y1,x2,y2] = getpoints_predefined_matlabimage()

x1 = 1.0e+03 * [
    0.1692
    0.5515
    0.4525
    0.4165
    1.4241
    1.4345
    1.6699
    1.6744
    1.3011
    1.1542
    0.9938
    1.1647
];

y1 = [
  887.1881
  932.1662
  467.3931
  110.5673
  146.5498
  392.4297
  392.4297
  677.2906
  834.7138
  770.2452
  554.3507
  434.4092
];

x2 = 1.0e+03 * [
    0.2651
    0.6249
    0.5680
    0.5365
    1.5440
    1.5545
    1.8154
    1.8019
    1.2951
    1.1452
    0.9953
    1.1647
];

y2 = [
  869.1969
  921.6713
  473.3902
  136.0549
  143.5512
  395.4283
  390.9305
  690.7840
  849.7064
  782.2394
  564.8455
  449.4019
];

end
