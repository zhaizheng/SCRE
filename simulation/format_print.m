function format_print_to_latex(result)
    [m,n] = size(result);
    for i = 1:m
        for j = 1:n
            fprintf('%s %.4f ','&',result(i,j));
        end
        fprintf('%s','\\');
        fprintf('\n')
    end
end