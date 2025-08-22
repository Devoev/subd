clear

%% Parse data
data = csvread("errs.csv", 1, 0);
n_dof_sqrt = sqrt(data(:,1));
err_l2 = data(:,2);
err_h1 = data(:,3);

%% Calculate approximate convergence rate
x = log(n_dof_sqrt);

%% L2
y = log(err_l2);
q =  (y(end) - y(end-1)) / (x(end) - x(end-1));
fprintf('Approximated order of convergence in L2: q = %.6f \n', -q);

%% H1
y = log(err_h1);
q =  (y(end) - y(end-1)) / (x(end) - x(end-1));
fprintf('Approximated order of convergence in H1: q = %.6f \n', -q);

%% Plot
figure()
loglog(n_dof_sqrt, err_l2, 'x-')
hold on
loglog(n_dof_sqrt, err_h1, 'x-')
fplot(@(n) n.^(-4), [n_dof_sqrt(1) n_dof_sqrt(end)], 'b--')
fplot(@(n) n.^(-3), [n_dof_sqrt(1) n_dof_sqrt(end)], 'r--')
xlabel("Square root of number of DOFs")
ylabel("Error")
legend("L2 error", "H1 error", "Expected rate in L^2 (p + 1 = 4)", "Expected rate in H^1 (p = 3)")
pause()
