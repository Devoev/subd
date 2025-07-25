data = csvread("err_catmull_clark.csv", 1, 0);
n_dof = data(:,1);
err_l2 = data(:,2);

figure()
loglog(n_dof, err_l2, 'x-')
hold on
fplot(@(n) n.^(-4/2), [n_dof(1) n_dof(end)], 'r--')
xlabel("Number of DOFs")
ylabel("L^2 error")
legend("Data", "Expected rate in L^2 (p + 1 = 4)")
pause()
