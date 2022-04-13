%% 
%
% File:     Exercise03_XinCheng.m
% Author:   Xin Cheng
% Date:     08.05.2021
% Comment:  This File is the solution of the first exercise in exercise 03 in Computer Vision Course.
%
%
%% load model
[V,F] = openOFF('model.off', '');

%% close figure
close all;

%% display model again (possibly with changed vertex positions V)
plot_figure(V,F)

%% rotate and translate

figure
% Rotate the model first 45 degrees around the x-axis and then 120 degrees around the z-axis.
V1 = transformation(V,[deg2rad(45) 0 0]',[0 0 0]');
V1 = transformation(V,[0 0 deg2rad(120)]',[0 0 0]');
plot_figure(V1,F) % 函数定义见下

figure
% Rotate the model first 120 degrees around the z-axis and then 45 degrees around the x-axis.
V2 = transformation(V,[0 0 deg2rad(120)]',[0 0 0]');
V2 = transformation(V,[deg2rad(45) 0 0]',[0 0 0]');
plot_figure(V2,F)

% translate
figure
V3 = transformation(V,  [0 0 0]', [0.5 0.2 0.1]');
plot_figure(V3, F)

%% define a function for figure
function plot_figure(V,F)
    C = 0.3*ones(size(V,1),3);
    patch('Vertices', V, 'Faces', F, 'FaceVertexCData', C);
    axis equal;
    shading interp;
    camlight right;
    camlight left;
end

