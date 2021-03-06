function [vx, vy] = getFlow(I1, I2, sigma)

[M11, M12, M22, q1, q2] = getMq(I1, I2, sigma);

vx = (M22.*q1 - M12.*q2) ./ (M12.^2 - M11.*M22);  % 推导过程见 Exercise05 第一题c
vy = (M11.*q2 - M12.*q1) ./ (M12.^2 - M11.*M22);

end

