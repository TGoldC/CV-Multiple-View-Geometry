function [M11, M12, M22, q1, q2] = getMq(I1, I2, sigma)
% Compute structure tensor

% spatial gradient using central differences
% use I1 here since we compute flow for I1
Ix = 0.5*(I1(:,[2:end end]) - I1(:,[1 1:end-1])); 
% 前一部分是：每一行，从第2个数到最后一个数 拼接最后一个数；
% 后一部分是：每一行，第一个数 拼接 从第一个数到倒数第二个数；
% *0.5表现为：计算 某一个pixel的Ix，用其左右两边brightness的差 除以2 得到；最左(右)边，用其右(左)边的数减去本身 除以2
Iy = 0.5*(I1([2:end end],:) - I1([1 1:end-1],:));

% temporal gradient with forward differences
It = I2 - I1;

% gaussian kernel
k = ceil(4*sigma+1);
G = fspecial('gaussian', k, sigma);

M11 = conv2(Ix .* Ix, G, 'same');
M12 = conv2(Ix .* Iy, G, 'same');
M22 = conv2(Iy .* Iy, G, 'same');

q1 = conv2(Ix .* It, G, 'same');
q2 = conv2(Iy .* It, G, 'same');

end
