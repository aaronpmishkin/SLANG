% @Author: amishkin
% @Date:   2018-05-13
% @Email:  aaron.mishkin@riken.jp
% @Last modified by:   aaronmishkin
% @Last modified time: 18-08-16


function [D, Del, R] = sherman_morrison_memory(A, diagonal, varargin)
    [l, J] = size(A);
    % Delete old memory.
    D = zeros(l, 0);
    Del = zeros(0);
    R = zeros(l, 0);

    % Obtain the optional vector signs
    if size(varargin) > 0
        signs = varargin{1};
    else
        signs = ones(1,J);
    end

    for i = 1:J
        d_new = sherman_morrison_inv(A, diagonal, D, Del, A(:,i), signs);
        del_new = d_new' * A(:,i);

        r_new = sherman_morrison_sr_transpose(diagonal, R, Del, A(:,i), signs);

        D(:, i) = d_new;
        Del(i) = del_new;
        R(:, i) = r_new;
    end
end
