%% 
%
% File:     Exercise01_XinCheng.m
% Author:   Xin Cheng
% Date:     17.04.2021
% Comment:  This File is the solution of exercise 01 in Computer Vision Course.
%
%
%% Basic image processing
%
% (a) Download lena.png, provided in ex01.zip.
% (b) Load the image into the workspace.
% (c) Determine the size of the image and show the image.
% (d) Convert the image to gray scale and determine the maximum and the minimum value of
% the image.
% (e) Apply a gaussian smoothing filter and save the output image.
% (f) Show 1) the original image, 2) the gray scale image and 3) the filtered image in one figure
% and give the figures appropriate titles.
% (g) Compare the gray scale image and the filtered image for different values of the smoothing.
%
image = imread('C:\Users\15051\Documents\MATLAB\Computer Vision Matlab\ex01\lena.png');
%size = size(image);
figure
imshow(image);
title('Lena(RGB)')

image_gray = rgb2gray(image); % convert to gray scale
figure
imshow(image_gray);
title('Lena(Gray)')
min1 = min(min(min(image))); % 二维数据3*4 运用一次min，得到1*4，即每一列的最小值，再运用一次min得到全局最小值。
max1 = max(max(max(image)));

image_blur = imgaussfilt(image,2); % Apply a gaussian smoothing filter
figure
imshow(image_blur);
title('Lena(filtered with \sigma = 2)')
Folder = 'C:\Users\15051\Documents\MATLAB\Computer Vision Matlab\ex01';  % save the output image.
File   = 'Lena(blur).png';
imwrite(image_blur, fullfile(Folder, File));

figure
subplot(1,3,1), imshow(image),title('Lena(RGB)')
subplot(1,3,2), imshow(image_gray),title('Lena(Gray)')
subplot(1,3,3), imshow(image_blur),title('Lena(blur)')

image_blur1 = imgaussfilt(image,2); % Apply a gaussian smoothing filter
image_blur2 = imgaussfilt(image,4); % Apply a gaussian smoothing filter
image_blur3 = imgaussfilt(image,6); % Apply a gaussian smoothing filter
figure
subplot(2,2,1),imshow(image_gray),title('Lena(gray)')
subplot(2,2,2),imshow(image_blur1),title('Lena(filtered with \sigma = 2)')
subplot(2,2,3),imshow(image_blur2),title('Lena(filtered with \sigma = 4)')
subplot(2,2,4),imshow(image_blur3),title('Lena(filtered with \sigma = 6)')

%% Basic operations
A = [2 2 0; 0 8 3];
b = [5; 15];
x = A\b;    % solve Ax = b
B = A;
A(1,2) = 4;

c = 0;
for i = -4:4:4
    c = c + i* A'* b;
end
disp(c)

result1 = A.*B; % coresponding element multiplication
result2 = A'*B; % transpose of A matrix multiplication with B

%% Write a function approxequal
% Write a function approxequal(x; y; Epsilon) checking if two vectors x and y are almost equal, i.e.
% if arbitrary i |xi - yi| <= Epsilon
% The output should be logical 1 or 0.
% If the input consists of two matrices, your function should compare the columns of the matrices
% if they are almost equal. In this case, the output should be a vector with logical values 1 or 0.

% Test
x = [1 2; 3 4];
y = [1.01 2.02; 3.03 4.04];
output = approxequal(x,y,0.1);

% 
% function [output] = approxequal(x,y,Epsilon)
% 
% if Epsilon <0
%     error('Error: Input Epsilon must be larger than zero.')
% end
% 
% if size(x) ~= size(y)
%     error('Error: The sizes of input x and y should be same.')
% end
% 
% z = abs(x - y);
% maxdiff = max(z);
% Size = size(maxdiff);
% output = zeros(Size,'logical');
% 
% for i = 1:Size(2)
%     if maxdiff(i) - Epsilon <= 1e-6        % 设一个很小的精度，只要小于这个就行。因为matlab中， maxdiff 可能显示的是0.1000，但是实际上是 0.10000...1
%         output(i) = 1;
%     end
% end
% end


%% Write a function addprimes
% Write a function addprimes(s; e) returning the sum of all prime(素数) numbers between and including
% s and e. Use the Matlab-function isprime.

% Test

s = 6.5;
e = -0.5;
Sum = addprimes(s,e);


function [SumOfprimes] = addprimes(s,e)

if s > e
    Start_int = ceil(e);
    End_int = floor(s);
else
    Start_int = ceil(s);
    End_int = floor(e);
end

if Start_int <= 0
    Start_int = 1; 
    if End_int <= 0
        error('Error: The provided Interval is negative and does not contain primes')
    end
end

SumOfprimes = 0;
for i = Start_int:End_int
    Isprime = isprime(i);
    if Isprime == 1
        SumOfprimes = SumOfprimes + i;
    end
end
end
