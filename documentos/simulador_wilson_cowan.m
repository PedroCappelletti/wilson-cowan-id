clearvars; close all; clc


%% =========================================================================
%  SECCIÓN 1: PARÁMETROS DEL SISTEMA (modificar aquí)
% =========================================================================
te = 1; ti = 2; %ctes de tiempo para cada población neuronal
wEE = 6.4;  % pesos sinápticos (en particular generan oscilaciones sostenidas, ciclo límite)
wEI = 4.8;  
wIE = 6; 
wII = 1.2; 
ae = 1.2; ai = 1; %ganancia sigmoideas
thetae = 2.8; thetai = 4; %umbral sigmoideas
ki = 1/(1+exp(ai*thetai)); %ctes sigmoideas
ke = 1/(1+exp(ae*thetae));
%noise_seed = 42; % Semilla para ruido blanco reproducible
I0  = 0.0;    % Nivel inhibitorio inicial [rate]
E0  = 0.0;    % Nivel excitatorio inicial [rate]

t0  = 0.0;    % Tiempo inicial [s]
tf  = 600.0;   % Tiempo final [s]

%% =========================================================================
%  SECCIÓN 2: GENERACIÓN SEÑALES DE ENTRADA Pstim, Qstim
%  Se pre-genera en una grilla fina y luego se interpola en el ODE
% =========================================================================
N_pre  = 10000;
t_pre  = linspace(t0, tf, N_pre);
dt_pre = t_pre(2) - t_pre(1);

%rng(noise_seed);   % reproducibilidad del ruido

%%%% pulsos cuadrados, modificar a voluntad el tipo de entrada %%%%%%
t_box_on  = 100.0; % Inicio pulso box [s]
t_box_off = 400.0; % Fin    pulso box [s]

Pstim  =  0.8 * double(t_pre >= t_box_on & t_pre < t_box_off);
Qstim  =  0.6 * double(t_pre >= t_box_on + 100 & t_pre < t_box_off +100);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Función de interpolación (usada dentro del ODE)
Psi = @(t) interp1(t_pre, Pstim, t, 'linear', 0);
Qsi = @(t) interp1(t_pre, Qstim, t, 'linear', 0);

%% =========================================================================
%  SECCIÓN 3: DEFINICIÓN DEL SISTEMA DE EDOs (dos entradas, una salida)
%  Estado: X = [I; E]
%  dotI = (1/ti)*(-I+1./(1+exp(-ai*(wIE*E-wII*I+Qsi-thetai)))-ki);  
%  dotE = (1/te)*(-E+1./(1+exp(-ae*(wEE*E-wEI*I+Psi-thetae)))-ke);
%     y = E-I;
% =========================================================================

msd_ode = @(t, X) [ ...
    (1/ti)*(-X(1)+1./(1+exp(-ai*(wIE*X(2)-wII*X(1)+Qsi(t)-thetai)))-ki); 
    (1/te)*(-X(2)+1./(1+exp(-ae*(wEE*X(2)-wEI*X(1)+Psi(t)-thetae)))-ke)];

%% =========================================================================
%  SECCIÓN 4: INTEGRACIÓN NUMÉRICA
% =========================================================================
rel_tol=1e-3; abs_tol=1e-6;
ode_opts = odeset('RelTol', rel_tol, 'AbsTol', abs_tol);

[t_sol, X_sol] = ode45(msd_ode, [t0 tf], [I0; E0], ode_opts);

I_sol = X_sol(:,1);   % tasa inhibitoria
E_sol = X_sol(:,2);   % tasa excitatoria
y_sol = E_sol-I_sol;  % potencial

%% =========================================================================
%  SECCIÓN 5: VISUALIZACIÓN
% =========================================================================
figure
tiledlayout(3,1), nexttile, 
plot(t_sol,Qsi(t_sol),t_sol,Psi(t_sol),'LineWidth',1.3), grid,ylabel('[xA]')
ylim([-0.1 1]) 
legend('Q input to I population','P input to E population','Location','northeast'),title('Inputs')
nexttile
plot(t_sol,I_sol,t_sol,E_sol,'LineWidth',1.3), grid, title('States'), ylim([-0.1 1]) 
legend('I: Inhibitory activity','E: Excitatory activity'), ylabel('[rate]')
nexttile
plot(t_sol,E_sol-I_sol,'LineWidth',1.3), grid
ylabel('[xV]'), title('Output'), xlabel('time(s)')
set(gcf,"Position",[ 21    96   716   665])