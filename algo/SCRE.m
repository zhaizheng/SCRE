
function [steps, data_move] = SCRE(data, data_move, opts)

    % each column of 'data' is a point, which assumed to be drawn from some
    % manifold with some noise added.
    % each column of 'data_move' is the outlier set, which we want to move
    % onto the underlying manifold
    
    steps = 0;
    n = size(data_move,2);
    [max_iter, epsilon, stepsize, sigma, neighbor_size, d] = getopts(opts);
    
    for i = 1:n
        for k = 1:max_iter
            direction = Update_Direction(data_move(:,i), sigma, data, d, neighbor_size);
            data_move(:,i) = data_move(:,i)+ stepsize*direction;
            if norm(direction) < epsilon
                break;
            end
        end
        steps = steps+k/n;
    end 
end


function g = Update_Direction(x, sigma, data, d, neighbor_size)

    s = neighbor_size;
    sq_distance = sum((data - x).^2,1);
    [~, ind] = sort(sq_distance);
    
    c = zeros(size(x));
    sum_r = 0;

    for i = ind(1:s)
        r = exp(-norm(x - data(:,i)).^2/(sigma^2));
        c = c + r*data(:,i);
        sum_r = sum_r + r;
    end
    c = c/sum_r;
    B = zeros(size(x,1));
    for i = ind(1:s)
        r = exp(-norm(x - data(:,i)).^2/(sigma^2));
        B = B + r*(data(:,i)-c)*(data(:,i)-c)';
    end
    [V,~,~] = svd(B);
    P = eye(size(x, 1))-V(:,1:d)*V(:,1:d)';
    g = P*(c - x);
end


function [maxiter, epsilon, stepsize, sigma, neighbor_size, d] = getopts(opts)

    epsilon = 1e-10;  maxiter = 10000;  stepsize = 0.1; neighbor_size = 20;
    d = 1;
   
    if isfield(opts, 'maxiter'); maxiter = opts.maxiter; end
    if isfield(opts, 'epsilon'); epsilon = opts.epsilon; end
    if isfield(opts, 'stepsize'); stepsize = opts.stepsize; end
    if isfield(opts, 'h'); sigma = opts.h; end
    if isfield(opts, 'neighbor_size'); neighbor_size = opts.neighbor_size; end
    if isfield(opts, 'd'); d = opts.d; end
end