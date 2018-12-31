% @Author: amishkin
% @Date:   2018-05-13
% @Email:  aaron.mishkin@riken.jp
% @Last modified by:   aaronmishkin
% @Last modified time: 18-08-16



function [p] = sherman_morrison_inv(A, diagonal, D, Del, x, varargin)
    [~, K] = size(D);

    % Obtain the optional vector signs
    if size(varargin) > 0
        signs = varargin{1};
    else
        signs = ones(1,K);
    end

    p = x ./ diagonal;
    for j = 1:K
        c = (A(:, j)' * p) / (1 + signs(j) * Del(j));
        p = (p - (signs(j) .* (D(:, j) .* c)));
    end

end
