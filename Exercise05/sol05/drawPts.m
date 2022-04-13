function [  ] = drawPts( img, pts )
hold off
imagesc(img);   % imagesc(C) 将数组 C 中的数据显示为一个图像，该图像使用颜色图中的全部颜色。C 的每个元素指定图像的一个像素的颜色。
% imagesc(A)将矩阵A中的元素数值按大小转化为不同颜色，并在坐标轴对应位置处以这种颜色染色。
colormap gray
hold on
plot(pts(:,1), pts(:,2), 'yo','LineWidth',3)
axis equal
end

