function y = fft2adj_rectangular_new(x, ind, m,n)

%if  strcmp(class(x) , 'double')
y = zeros([m n]);
% elseif strcmp(class(x), 'gpuArray')
%    y = zeros(m, n,'gpuArray');
% end

y(ind) = x;
y = (ifft2(fftshift(y)));

