function V = transformation(V,rotation_angles, translation)
    % rotate the vertices V around their center with given rotation_angles(a vector) around 
    % axes x, y and z, then translate by translation(also a vector)

    center = mean(V)';
    no_translation = zeros(3,1);
    
    T = SE3(eye(3), translation) * ...
        SE3(eye(3), center) * ... 
        SE3(rotate_along_x(rotation_angles(3)), no_translation) * ...
        SE3(rotate_along_y(rotation_angles(2)), no_translation) * ...
        SE3(rotate_along_z(rotation_angles(1)), no_translation) * ...
        SE3(eye(3), -center);
    % 从下往上，依次为 平移到原点---绕x轴旋转---绕y轴旋转---绕z轴旋转---平移回中心---进行要求的平移转换
    
    V_hom = [V ones(size(V,1),1)]'; % V是19105*3的矩阵，homogeneous坐标系下，必须要V成为 4*19105
    V_hom = T * V_hom;
    
    V = V_hom(1:3,:)';  %最后V只去 4*19105 的前三行
end


function T = SE3(R, t) % 把rotation和translation转化成SE（3）的形式
    T = eye(4);
    T(1:3, 1:3) = R;
    T(1:3,4) = t;
end

function R = rotate_along_x(theta)
    R = [1      0       0;
         0 cos(theta) -sin(theta);
         0 sin(theta)  cos(theta)];
end

function R = rotate_along_y(theta)
    R = [ cos(theta) 0 sin(theta);
               0 1      0;
         -sin(theta) 0 cos(theta)];
end

function R = rotate_along_z(theta)
    R = [cos(theta) -sin(theta) 0;
         sin(theta)  cos(theta) 0;
              0       0 1];
end