[clean, noisy] = cryo_em_data(0);
Data_c = zeros(64*64, 200);
sample_size = 200;

sigma = 1;
for i = 1:sample_size
    Data_c(:,i) = reshape(clean(i,:,:),[64*64,1]);
end

[U,~,~] = svd(Data_c);
D = 10;
V = U(:,1:D);
Data_o = U(:,1:D)'*Data_c;
Data_n = zeros(D, sample_size);

for i = 1:sample_size
    Data_n(:,i) = Data_o(:,i)+sigma*randn(D,1);
end

X1 = Data_n;
X2 = Data_n;

norm_sq = sum(Data_n.^2,1);
D = ones(sample_size,1)*norm_sq+norm_sq'*ones(1, sample_size)-2*Data_n'*Data_n;
sigma_std = sqrt(mean(D(:)));

d = 5; s = 100; 
%%
figure
step1(Data_o, X1, X2, d, sigma_std, V, s);
%result = step2(X1, X2, d, Data_o, sigma_std, s);
%%
figure
h = (0.1:0.1:0.9)*sigma_std;
%result = step2(X1, X2, d, Data_o, sigma_std, s);
draw(result, h)

function draw(result, h)
    marker1 = {'r-.*','b-.o','k-.>','m-.d'};
    subplot(1,2,1)
    for k = 1:4
        semilogy(h,result(k,:),marker1{k},'linewidth',1.2); 
        hold on; 
    end
    %semilogy(sigma, result(2,:),'--o','linewidth',1.2); %hold on; semilogy(sigma, result(3,:),'--s');
    legend('SCRE','{\it{l}}-SCRE','MFIT-i', 'MFIT-ii')
    xlabel('h')
    title('Average Margin')
    axis([min(h), max(h), 1.5, 4])
    set(gca,'FontSize',16);
    
    %subplot('position',[0.53 0.14 0.45 0.82])
    subplot(1,2,2)
    for k = 1:4
        semilogy(h,result(4+k,:),marker1{k},'linewidth',1.2); 
        hold on; 
    end  
    %semilogy(sigma, result(6,:),'--o','linewidth',1.2); %hold on; semilogy(sigma, result(6,:),'--s')
    legend('SCRE','{\it{l}}-SCRE','MFIT-i', 'MFIT-ii')
    xlabel('h')
    title('Hausdorff')
    axis([min(h), max(h), 3, 8])
    set(gca,'FontSize',16);
end

function step1(X0, X1, X2, d, sigma_std, V, s)
    h = sigma_std*0.3;
    [~, data6]  = algorithm(X2, X1, h, 1, d, s);
    [~, data6n] = algorithm(X2, X1, h, 7, d, s);
%    [~, data6nn] = algorithm(X2, X1, sigma, 8);
    %subplot('position',[0.04 0.1 0.45 0.82]);
    i = 1;
    subplot('position', [0.02*i+0.22*(i-1) 0.12 0.23 0.80]);
    show_im(3, 3, V*X0);
    title('Pure Signal')
    axis off
    i = 2;
    subplot('position', [0.02*i+0.22*(i-1) 0.12 0.23 0.80]);
    show_im(3, 3, V*X1);
    title('Noisy')
    axis off
    i = 3;
    subplot('position', [0.02*i+0.22*(i-1) 0.12 0.23 0.80]);
    show_im(3, 3, V*data6);
    title('SCRE')
    axis off
    %set(gca,'FontSize',16);
    
    %subplot('position',[0.53 0.1 0.45 0.82])
    i = 4;
    subplot('position', [0.02*i+0.22*(i-1) 0.12 0.23 0.80]);
    show_im(3, 3, V*data6n);
    title('{\it{l}}-SCRE')
    axis off
    %set(gca,'FontSize',16);
end


function show_im(m, n, data2)
    im = zeros(64*m,64*n);
    for i = 1:m
        for j = 1:n
            im((i-1)*64+1:i*64,(j-1)*64+1:j*64) = reshape(data2(:,(i-1)*n+j),[64,64]);
        end
    end
    image(im*200);
end

function result = step2(X1, X2, d, Data_o, sigma_std, s)
%    [X1, X2, ~] = generate_sphere(0.04, D, sample_size, sample_size);
    h = (0.1:0.1:0.9)*sigma_std;
    result = zeros(8, length(h));
    %s = 30;
    for i = 1:length(h)
        [~, data6]     = algorithm(X2, X1, h(i), 1, d, s);
        [~, data6n]    = algorithm(X2, X1, h(i), 7, d, s);
        [~, data6nn]   = algorithm(X2, X1, h(i), 8, d, s);
        [~, data6nnn]  = algorithm(X2, X1, h(i), 9, d, s);
        result(1,i) = average_distance(data6, Data_o);
        result(2,i) = average_distance(data6n, Data_o);
        result(3,i) = average_distance(data6nn, Data_o);
        result(4,i) = average_distance(data6nnn, Data_o);
        result(5,i) = max_distance(data6, Data_o);
        result(6,i) = max_distance(data6n, Data_o);
        result(7,i) = max_distance(data6nn, Data_o);
        result(8,i) = max_distance(data6nnn, Data_o);
    end
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


function [steps, data_move] = algorithm(data, data_move, sigma, algo, d, s)
    epsion = 1e-10;
    max_iter = 10000;
    steps = 0;
    n = size(data_move,2);
    step = 0.5;
    for i = 1:n
        for k = 1:max_iter
            if algo == 1
                direction = Hc1(data_move(:,i), sigma, data, d, step);
            elseif algo == 2
                direction = Hc2(data_move(:,i), sigma, data, d, step);
            elseif algo == 3
                direction = Hc3(data_move(:,i), sigma, data, d, step);
            elseif algo == 4
                direction = Hc44(data_move(:,i), sigma, data, d, step);
            elseif algo == 5
                direction = Hc55(data_move(:,i), sigma, data, d, step);
            elseif algo == 6
                direction = Hc66(data_move(:,i), sigma, data, d, step);
            elseif algo == 7
                direction = Hc77(data_move(:,i), sigma, data, d, step, s);
                %direction = xia(data_move(:,i), data, sigma, 3, 2);     
            elseif algo == 8
                direction = xia(data_move(:,i), data, sigma, 2, d, step);
            elseif algo == 9
                direction = mfit(data_move(:,i), data, sigma, 2, d, step);
            end
            data_move(:,i) = data_move(:,i)+ direction;
            if norm(direction) < epsion
                break;
            end
        end
        steps = steps+k/n;
        %plot(track(1,:), track(2,:),'r-','Linewidth',1)
        %hold on
        %fprintf('iteration i=%d, iter=%d, error=%f\n', i, k, norm(direction));
    end
    
end

function direction = mfit(x, data, r, beta, dim, step)
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


function direction = xia(x, data, r, beta, dim, step)
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

function P = normal_space(data, i, r, dim)
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


function g = Hc1(x, sigma, data, d, step)
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


function g = Hc77(x, sigma, data, d, step, s)

    sq_distance = sum((data - x).^2,1);
    [~, ind] = sort(sq_distance);
    
    c = zeros(size(x));
    sum_r = 0;
    %for i = 1:size(data,2)
    %s = 20;
    for i = ind(1:s)
        r = exp(-norm(x - data(:,i)).^2/(sigma^2));
        c = c + r*data(:,i);
        sum_r = sum_r + r;
    end
    c = c/sum_r;
    B = zeros(size(x,1));
%    for i = 1:size(data,2)
    for i = ind(1:s)
        r = exp(-norm(x - data(:,i)).^2/(sigma^2));
        B = B + r*(data(:,i)-c)*(data(:,i)-c)';
    end
    [V,~,~] = svd(B);
    P = eye(size(x, 1))-V(:,1:d)*V(:,1:d)';
    %g = 2*sum_r/size(data,2)/sigma^2*P*(c - x);
    g = step*P*(c - x);
end
