%% regular

N = 30;
tau = 100;
dg = 0.01;
gstart = 0.6;

f = @(x, a, c) 1./(1 + exp(-a.*(x-c)));

t_b = 20;
t_t = 150;

data = zeros(N, size(gstart:dg:1, 2));

for n = 1:N
    e_a = (n + 1) / 2;
    e_a2 = ((n * n) + 1) / 2;
    idg = 1;
    for g = gstart:dg:1
        s_b = 0;
        s_t = 0;
        for i = 1:tau
            s_b = s_b + (g^(i - 1)) * f(i, 0.1, t_b);
            s_t = s_t + (g^(i - 1)) * f(i, 0.1, t_t);
        end
        
        c_eq = (2 * e_a * (1 - (s_t / s_b))) / (e_a2 - e_a);
        
        data(n, idg) = c_eq;
        idg = idg + 1;
    end
end


hold on;
surf(gstart:dg:1, 1:N, data, 'FaceColor', [0.8 0.27 0.15]);
ylabel('N', 'FontSize', 12);
xlabel('\gamma', 'FontSize', 18);
zlabel('C_{eq}', 'FontSize', 12);
zlim([0 5]);

%% EVC

N = 30;
tau = 100;
dg = 0.01;
gstart = 0.6;

f = @(x, a, c) 1./(1 + exp(-a.*(x-c)));
eta = @(x) 0.1 * x^(2);

t_b = 20;
t_t = 40;

data = zeros(N, size(gstart:dg:1, 2));

for n = 1:N
    e_a = (n + 1) / 2;
    e_a2 = ((n * n) + 1) / 2;
    
    idg = 1;
    for g = gstart:dg:1
        s_b = 0;
        s_t = 0;
        mu = 0;
        for i = 1:tau
            s_b = s_b + (g^(i - 1)) * f(i, 0.1, t_b);
            s_t = s_t + (g^(i - 1)) * f(i, 0.1, t_t);
            mu = mu + (g^(i - 1));
        end
        
        e_n_a = (0.1 + (0.1 * n^2)) / 2;
        sub = mu * (e_n_a - eta(1) * e_a);
        num = (e_a * s_t) - sub;
        numnum = 2 * (e_a - (num / s_b));
        c_eq = numnum / (e_a2 - e_a);
               
        data(n, idg) = c_eq;
        idg = idg + 1;
    end
end

hold on;
surf(gstart:dg:1, 1:N, data)
ylabel('N', 'FontSize', 12);
xlabel('\gamma', 'FontSize', 18);
zlabel('C_{eq}', 'FontSize', 12);
zlim([0 5]);

%%

x = 0:0.01:30;

figure;
plot(x, 0.01 .* x.^2);
hold on;
plot(x, 0.01 * x);
xlim([0 1]);
ylim([0 0.01]);