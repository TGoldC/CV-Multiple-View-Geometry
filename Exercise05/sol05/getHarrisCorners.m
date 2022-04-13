function [score, points] = getHarrisCorners(I, sigma, kappa, theta)

[M11, M12, M22] = getM(I, sigma);

% compute score using det and trace
detM = M11 .* M22 - M12.^2; 
% M11等都是一个480*640的矩阵，算出来的M也是一个480*640的矩阵，里面的每一项表示
% 对于I中的每个pixel的 M的determination
traceM = M11 + M22; % traceM的大小也是480*640
score = detM - kappa * traceM.^2; % 见讲义第4章 Page16，也是480*640

% display score (for debugging)
%imagesc(sign(score) .* abs(score).^(1/4));
%colorbar;

% you can disable non-maximum suppression (for debugging)
max_only = 1;  % c小问中，不仅要找score大于theta的，还要找score是local maximum的那些pixels，即该点的
% 这个max_only就是为了 设定 要不要考虑local maximum的这个条件

% padded image for easier non-max suppression check
score_pad = -inf * ones(size(I) + 2); % ones(size(I) + 2)得到的是 482*642的 全部都是1 的矩阵;* -inf 变成都是负无穷的；上下个加一行
score_pad(2:end-1, 2:end-1) = score; % score_pad应该 周围一圈为-inf，中间480*640为 上面算的score

% find corners
[y, x] = find( score > theta ...
             & ( ~max_only ...                             % 当max_only 为0时，&后面肯定为1，也就是不管它是不是local maximum;为1时，检测后面是不是满足local maximum
               | ( score > score_pad(1:end-2, 2:end-1) ... % 每个pixel的score 要比它上面的pixel大 这里是c小问的答案
                 & score > score_pad(3:end  , 2:end-1) ... % 每个pixel的score 要比它右边的pixel大
                 & score > score_pad(2:end-1, 1:end-2) ... % 每个pixel的score 要比它左边的pixel大
                 & score > score_pad(2:end-1, 3:end))));   % 每个pixel的score 要比它下面的pixel大
% image的行index是 画图的纵坐标；列index是画图的横坐标
points = [x y]; % image coordinate中的坐标系是建在 左上角
% points指的是所有 被检测到是corners的坐标
end