%% cleanup
clear
close all

%% load data and set parameters
I1 = imreadbw('img1.png');
I2 = imreadbw('img2.png');
%I1 = imreadbw('small.png');
%I2 = imreadbw('small2.png');
sigma = 2;
kappa = 0.05;
theta = 1e-7;

%% compute flow and visualize

[vx, vy] = getFlow(I1, I2, sigma);

% min/max for equal color scale of by velocity plots
velmin = min(min(vx(:)), min(vy(:))); % 速度的最小值
velmax = max(max(vx(:)), max(vy(:))); % 速度的最大值

figure(1)

subplot(221)
imagesc(vx)
colormap(gca, 'jet')
colorbar
caxis([velmin velmax])
axis image
title('vx')

subplot(222)
imagesc(vy)
colormap(gca, 'jet')
colorbar
caxis([velmin velmax])
axis image
title('vy')

subplot(223)
imagesc(I1)
colormap(gca, 'gray')
axis image
hold on
% plot arrows with true scale
quiver(vx, vy, 0) % 在等距点上绘制箭头，箭头的定向分量由 vx 和 vy 指定，0表示禁用自动缩放
% 第三张图里的所有的箭头，标示的是速度的方向
title('flow')


% Bonus: Compute flow at image corners

[vx_corners, vy_corners] = getSparseFlow(I1, I2, sigma, kappa, theta);

display_scale = 10;

subplot(224)
imagesc(I1)
colormap(gca, 'gray')
axis image
hold on
% plot arrows with true scale
quiver(display_scale*vx_corners, display_scale*vy_corners, 0)
title(sprintf('flow corners (%dx)', display_scale));