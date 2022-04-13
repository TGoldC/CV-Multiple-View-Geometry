%% Multiple View Geometry 2020, Exercise Sheet 1
% Prof. Dr. Florian Bernard, Florian Hofherr, Tarun Yenamandra

%% Exercise 1

%% (b)
I = imread('lena.png');

%% (c)
[rows, cols, channels] = size(I)
figure
imshow(I) , axis image

%% (d)
I_gray = rgb2gray(I);

min_I = min(I_gray(:))
max_I = max(I_gray(:))

%% (e)
I_filt = imgaussfilt(I_gray);

imwrite(I_filt, 'lena_gauss.png');

% More general filters can be obtained by using fspecial() and conv2()

%% (f)
subplot(131), imshow(I),      title('Original Lena')
subplot(132), imshow(I_gray), title('Grayscale Lena')
subplot(133), imshow(I_filt), title('Smoothed Lena')

%% (g)
sigma = 3;
I_filt = imgaussfilt(I_gray, sigma);

subplot(131), imshow(I),      title('Original Lena')
subplot(132), imshow(I_gray), title('Grayscale Lena')
subplot(133), imshow(I_filt), title('Smoothed Lena')

%% Exercise 2

%% (a)
A = [2 2 0; 0 8 3]
b = [5; 15]

x = A\b         % \: Universal. Solve system of equations and least squares

%% (b)
B = A

%% (c)
A(1, 2) = 4     % Indexing starts from 1!

%% (d)
c = 0;
for i = -4:4:4          % start:step:stop
    c = c + i*A'*b;     % "Broadcasting" in the fist iteration
end
c

%% (e)
A.*B % is an element-wise multiplication, requires matrices of equal size
A'*B % is the matrix multiplication, requires number of columns of first factor = number of rows of second one

%% Exercise 3

approxequal([1,2;3,4], [1,2;3.1,3.5], 0.5)

%% Exercise 4

addprimes(1,10)
addprimes2(1,10)