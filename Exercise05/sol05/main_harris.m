%% cleanup
clear
close all

%% load data and set parameters
I = imreadbw('img1.png');
% imshow(I)
%I = imreadbw('small.png');
sigma = 2;
kappa = 0.05;
theta = 1e-7;

%% compute corners and visualize 

[score, points] = getHarrisCorners(I, sigma, kappa, theta);

figure(2);
subplot(121);
drawPts(I, points); % 是一个自己写的函数，为了画图
axis image; % 能够同时实现紧凑以及xy比例一致两个功能

subplot(122);
imagesc(sign(score) .* abs(score).^(1/4));
colormap(gca, 'jet'); % gca表示当前坐标区或图；colormap设置当前图，为彩色；jet表示是颜色图
axis image;
colorbar; % 显示颜色栏


% sigma 越大，图片越糊，识别到 corners 变少
