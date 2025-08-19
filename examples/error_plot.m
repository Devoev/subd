clear

%% Parse data
data = csvread("errs.csv", 1, 0);
n_dof_sqrt = sqrt(data(:,1));
err_l2 = data(:,2);

%% Calculate approximate convergence rate
x = log(n_dof_sqrt);
y = log(err_l2);
q =  (y(end) - y(end-1)) / (x(end) - x(end-1));
fprintf('Approximated order of convergence: q = %.6f', -q);

%% Plot
figure()
loglog(n_dof_sqrt, err_l2, 'x-')
hold on
fplot(@(n) n.^(-4), [n_dof_sqrt(1) n_dof_sqrt(end)], 'r--')
xlabel("Square root of number of DOFs")
ylabel("L^2 error")
legend("Data", "Expected rate in L^2 (p + 1 = 4)")
pause()
