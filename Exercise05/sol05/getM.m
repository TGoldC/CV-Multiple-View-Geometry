function [M11, M12, M22] = getM(I, sigma)
% Compute structure tensor
% I 是整个image的每个pixel的brightness   480*640
% sigma 是 Gaussian分布的 方差

% spatial gradient using central differences
Ix = 0.5*(I(:,[2:end end]) - I(:,[1 1:end-1])); 
% 前一部分是：每一行，从第2个数到最后一个数 拼接最后一个数；
% 后一部分是：每一行，第一个数 拼接 从第一个数到倒数第二个数；
% *0.5表现为：计算 某一个pixel的Ix，用其左右两边brightness的差 除以2 得到；最左(右)边，用其右(左)边的数减去本身 除以2
Iy = 0.5*(I([2:end end],:) - I([1 1:end-1],:));

% Gaussian kernel
k = ceil(4*sigma+1); % 朝正无穷大方向取整
G = fspecial('gaussian', k, sigma); % fspecial函数是滤波函数，
% 高斯半径(sigma)对曲线形状的影响,sigma越小,曲线越高越尖,sigma越大,曲线越低越平缓。因此高斯半径越小,则模糊越小,高斯半径越大,则模糊程度越大

M11 = conv2(Ix .* Ix, G, 'same');
% 每个pixel对应一个window，算M的时候，也是 一个pixel对应一个M，即一个M11
% 所以最后得到的M11 一定是和 I的大小相同
% 用 same的 paddig可以实现 M11和I的size相同；
M12 = conv2(Ix .* Iy, G, 'same');
M22 = conv2(Iy .* Iy, G, 'same');

% Note: We do not need to compute M21 sparately, since M is symmetrical and
% therefore M21 == M12

end
