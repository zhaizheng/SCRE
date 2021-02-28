addpath('../algo/');

% D: the dimension of the ambient space;
% d: the dimension of the manifold;
% h: the bandwidth in SCRE;
% sample_size: the number of samples;
% sigma: the variance of data generated;
% maxiter: the maximum iteration in SCRE;
% neighbor_size: the number of neighbors which affect SCRE

sample_size = 200;
D = 2;
sigma = 0.08;
[X1, X2, ~] = generate_sphere(sigma, D, sample_size, sample_size);

opts.h=0.4; opts.maxiter = 10000;  opts.neighbor_size = 20; opts.d=1;
[steps, result] = SCRE(X2, X1, opts);

plot(X2(1,:),X2(2,:),'d');
hold on
plot(result(1,:),result(2,:),'b.')


function [data_ini, samples, X] = generate_sphere(sigma, D, NumSample, NumIni)
    samples = randn(D, NumSample);
    samples = samples*diag(1./sqrt(sum(samples.^2)))+ sigma*randn(D, NumSample);
    
    data_ini = randn(D, NumIni);
    X = data_ini*diag(1./sqrt(sum(data_ini.^2)));
    data_ini = X + 0.5*sqrt(sigma)/sqrt(D)*(2*rand(D,NumIni)-1);
end

