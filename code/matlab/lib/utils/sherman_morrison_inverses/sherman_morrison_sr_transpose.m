% @Author: amishkin
% @Date:   2018-05-13
% @Email:  aaron.mishkin@riken.jp
% @Last modified by:   aaronmishkin
% @Last modified time: 18-08-16

function [p] = sherman_morrison_sr_transpose(diagonal, R, Del, x, varargin)
    [~, K] = size(R);

    % Obtain the optional vector signs
    if size(varargin) > 0
        signs = varargin{1};
    else
        signs = ones(1,K);
    end

    p = x;
    p = p ./ sqrt(diagonal);

    for j = 1:K
        c = 1 - sqrt(1 / (1 + (signs(j) * Del(j))));
        vp = (R(:, j) .* (R(:, j)' * p)) .* c;
        p = (p - (signs(j) .* (vp ./ (signs(j) * Del(j)))));
    end

end
