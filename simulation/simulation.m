figure(1);
sample_size = 300; % in the paper, we let it to be 300;
D = 2; d = 1; % One-dimensional circle
%D = 3; d = 2; % Two-dimensional sphere
sigma = 0.4; neigh_size = 20;

Ridge_Computation(sample_size, D, d, sigma, neigh_size);
result = Compare_Algorithms(sample_size, D, d);
format_print_to_latex(result);


function Ridge_Computation(sample_size, D, d, sigma, neigh_size)
    [X1, X2, ~] = generate_sphere(0.08, D, sample_size, sample_size);
    [~, result_scre]  = Algorithm(X2, X1, sigma, 1, d);
    [~, result_lscre] = Algorithm(X2, X1, sigma, 2, d, neigh_size);
    
    subplot(2,2,1)
    plot(X2(1,:), X2(2,:), 'd');
    N2 = normalize(result_scre);
    hold on
    plot(result_scre(1,:), result_scre(2,:), 'b.');
    hold on
    plot(N2(1,:), N2(2,:), 'r.');
    title('SCRE')
    set(gca,'FontSize',16);
    
    subplot(2,2,2)
    plot(X2(1,:), X2(2,:), 'd');
    N3 = normalize(result_lscre);
    hold on
    plot(result_lscre(1,:), result_lscre(2,:), 'b.');
    hold on
    plot(N3(1,:), N3(2,:), 'r.');
    title('{\it{l}}-SCRE')
    set(gca,'FontSize',16);
end


function result = Compare_Algorithms(sample_size, D, d)
    [X1, X2, ~] = generate_sphere(0.04, D, sample_size, sample_size);
    sigma = 0.1:0.1:0.9; neigh_size = 20;
    
    result = zeros(8, length(sigma));
    for i = 1:length(sigma)
        [~, result_SCRE]   = Algorithm(X2, X1, sigma(i), 1, d);
        [~, result_lSCRE]  = Algorithm(X2, X1, sigma(i), 2, d, neigh_size);
        [~, result_mfit1]  = Algorithm(X2, X1, sigma(i), 3, d);
        [~, result_mfit2]  = Algorithm(X2, X1, sigma(i), 4, d);
        Ridge = {result_SCRE, result_lSCRE, result_mfit1, result_mfit2};
        for j = 1:4
            result(j,i) = average_distance(Ridge{j}, bsxfun(@rdivide, Ridge{j}, sqrt(sum(Ridge{j}.^2,1))));
            result(j+4,i) = max_distance(Ridge{j}, bsxfun(@rdivide, Ridge{j}, sqrt(sum(Ridge{j}.^2,1))));
        end
    end
    marker1 = {'-*','-o','-->','--d'};
    
    subplot(2,2,3)
    for k = 1:4
        semilogy(sigma,result(k,:),marker1{k},'linewidth',1.2); 
        hold on; 
    end
    legend('SCRE','{\it{l}}-SCRE','MFIT-i', 'MFIT-ii')
    xlabel('h')
    title('Average Margin')
    set(gca,'FontSize',16);
    
    subplot(2,2,4)
    for k = 1:4
        semilogy(sigma,result(4+k,:),marker1{k},'linewidth',1.2); 
        hold on; 
    end  
    legend('SCRE','{\it{l}}-SCRE','MFIT-i', 'MFIT-ii')
    xlabel('h')
    title('Hausdorff')
    set(gca,'FontSize',16);
end


function out = normalize(in)
    out = in*diag(1./sqrt(sum(in.^2)));
end


function ave = average_distance(data1, data2)
    n = size(data1,2);
    sum_d = sum(sqrt(sum((data1-data2).^2, 1)),2);
    ave = sum_d/n;
end


function max_d = max_distance(data1, data2)
    max_d = max(sqrt(sum((data1-data2).^2, 1)));
end


function [data_ini, samples, X] = generate_sphere(sigma, D, NumSample, NumIni)
    samples = randn(D, NumSample);
    samples = samples*diag(1./sqrt(sum(samples.^2)))+ sigma*randn(D, NumSample);
    
    data_ini = randn(D, NumIni);
    X = data_ini*diag(1./sqrt(sum(data_ini.^2)));
    data_ini = X + 0.5*sqrt(sigma)/sqrt(D)*(2*rand(D, NumIni)-1);
end


function [steps, data_move] = Algorithm(data, data_move, sigma, algo, d, neig_size)
    % As different algorithms corresponding to different directions for
    % updating, for fairness of comparison, we use a uniform framework for
    % the four different algorithms.
    epsilion = 1e-10;
    max_iter = 10000;
    steps = 0;
    n = size(data_move,2);
    step = 0.5;
    for i = 1:n
        for k = 1:max_iter
            if algo == 1
                direction = SCRE(data_move(:,i), sigma, data, d, step);
            elseif algo == 2
                direction = l_SCRE(data_move(:,i), sigma, data, d, step, neig_size); 
            elseif algo == 3
                direction = Mfit_1(data_move(:,i), data, sigma, 2, d, step);
            elseif algo == 4
                direction = Mfit_2(data_move(:,i), data, sigma, 2, d, step);
            end
            data_move(:,i) = data_move(:,i)+ direction;
            if norm(direction) < epsilion
                break;
            end
        end
        steps = steps+k/n;
    end
end


function direction = Mfit_1(x, data, r, beta, dim, step)
    d = size(data, 1);
    n = size(data, 2);
    d2 = 1 - sum((data-x).^2, 1)./(r^2);
    d2(d2<0) = 0;
    alpha_tilde = d2.^beta;
    alpha_sum = sum(alpha_tilde(:));
    if alpha_sum >0 
        alpha = alpha_tilde./alpha_sum;
    else
        alpha = alpha_tilde;
    end
    cx = sum(data.* alpha, 2);
    Ns = zeros(d);
    for i = 1:n
        if d2(i)>0
            ns = normal_space(data, i, r, dim);
            Ns = Ns+ns*alpha(i);
        end    
    end
    [U,~,~] = svd(Ns);
    P = U(:,1:d-dim)*U(:,1:d-dim)';
    direction = step*P*(cx - x);
end



function direction = Mfit_2(x, data, r, beta, dim, step)
    d = size(data, 1);
    n = size(data, 2);
    d2 = 1 - sum((data-x).^2, 1)./(r^2);
    d2(d2<0) = 0;
    alpha_tilde = d2.^beta;
    alpha_sum = sum(alpha_tilde(:));
    if alpha_sum >0 
        alpha = alpha_tilde./alpha_sum;
    else
        alpha = alpha_tilde;
    end
    Ns = zeros(d);
    c_vec = zeros(d,1);
    for i = 1:n
        if d2(i)>0
            ns = normal_space(data, i, r, dim);
            Ns = Ns+ns*alpha(i);
            c_vec = c_vec+alpha(i)*ns*(data(:,i)-x);
        end    
    end
    [U,~,~] = svd(Ns);
    P = U(:,1:d-dim)*U(:,1:d-dim)';
    direction = step*P*c_vec;
end




function g = SCRE(x, sigma, data, d, step)
    c = zeros(size(x));
    sum_r = 0;
    for i = 1:size(data,2)
        r = exp(-norm(x - data(:,i)).^2/(sigma^2));
        c = c + r*data(:,i);
        sum_r = sum_r + r;
    end
    c = c/sum_r;
    B = zeros(size(x,1));
    for i = 1:size(data,2)
        r = exp(-norm(x - data(:,i)).^2/(sigma^2));
        B = B + r*(data(:,i)-x)*(data(:,i)-x)';
    end
    [V,~,~] = svd(B);
    P = eye(size(x, 1))-V(:,1:d)*V(:,1:d)';
    %g = 2*sum_r/size(data,2)/sigma^2*P*(c - x);
    g = step*P*(c - x);
end


function g = l_SCRE(x, sigma, data, d, step, neig_size)

    sq_distance = sum((data - x).^2,1);
    [~, ind] = sort(sq_distance);
    
    c = zeros(size(x));
    sum_r = 0;
    for i = ind(1:neig_size)
        r = exp(-norm(x - data(:,i)).^2/(sigma^2));
        c = c + r*data(:,i);
        sum_r = sum_r + r;
    end
    c = c/sum_r;
    B = zeros(size(x,1));
    for i = ind(1:neig_size)
        r = exp(-norm(x - data(:,i)).^2/(sigma^2));
        B = B + r*(data(:,i)-c)*(data(:,i)-c)';
    end
    [V,~,~] = svd(B);
    P = eye(size(x, 1))-V(:,1:d)*V(:,1:d)';
    g = step*P*(c - x);
end


function P = normal_space(data, i, r, dim)

    % normal space is used repeatedly in Mfit_1 and Mfit_2 by averaging the
    % normal space in each neighborhood
    ds = sum((data-data(:,i)).^2, 1);
    ds(ds > r^2) = 0;
    n = size(data, 2);
    d = size(data, 1);
    indicator = zeros([1,n]);
    indicator(ds>0) = 1;
    select = (data-data(:,i)).*indicator;
    cor = select*select';
    [U,~,~] = svd(cor);
    P = eye(d)-U(:,1:dim)*U(:,1:dim)';     
end
