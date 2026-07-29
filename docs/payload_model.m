syms alpha_x  alpha_y  real
syms alpha_dot_x alpha_dot_y real
syms alpha_ddot_x alpha_ddot_y real

Q = [alpha_x alpha_dot_x alpha_ddot_x
     alpha_y alpha_dot_y alpha_ddot_y];

Ry = [ cos(alpha_x) 0 -sin(alpha_x)
       0            1  0
       sin(alpha_x) 0  cos(alpha_x)];
Rx = [1 0             0
      0 cos(alpha_y) -sin(alpha_y)
      0 sin(alpha_y)  cos(alpha_y)];
downI = [0; 0; -1];

qI = simplify(Ry*Rx*downI);
%%
function df = ddt(f, Q)
%DDT  Total time derivative of a symbolic expression via the chain rule.
%
%   df = ddt(f, Q) returns df/dt for f expressed in the generalized
%   coordinates and their derivatives listed in Q.
%
%   Q is an n-by-(m+1) symbolic array whose columns are successive time
%   derivatives of the n coordinates:
%
%       Q = [ q1  dq1  ddq1
%             q2  dq2  ddq2 ]
%
%   The last column is treated as the highest derivative present (an input,
%   not differentiated further), so with the 3-column Q above you can take
%   two derivatives. Add a column to go one order deeper.
%
%   Works on scalars, vectors, and matrices (shape is preserved).
%
%   Example:
%       syms a1 a2 da1 da2 dda1 dda2 real
%       Q = [a1 da1 dda1; a2 da2 dda2];
%       q     = [sin(a1)*cos(a2); sin(a2); -cos(a1)*cos(a2)];
%       qdot  = ddt(q,    Q);
%       qddot = ddt(qdot, Q);

x  = Q(:, 1:end-1);      % variables f is allowed to depend on
xd = Q(:, 2:end);        % their derivatives, same column ordering

sz = size(f);
df = reshape( jacobian(f(:), x(:)) * xd(:), sz );
end

function g = keepOrder(f, smallVars, n)
%KEEPORDER  Truncate a symbolic expression to order n in a set of small variables.
%
%   g = keepOrder(f, smallVars)      keeps terms up to first order (n = 1)
%   g = keepOrder(f, smallVars, n)   keeps terms up to order n
%
%   Every variable in smallVars is treated as O(eps). Terms of total degree
%   greater than n across the whole set are discarded -- so with n = 1,
%   alpha_x^2, alpha_x*alpha_y, and alpha_x*dalpha_y all drop out together.
%
%   Expansion is about zero, which is the swing equilibrium.
%   Works on scalars, vectors, and matrices (shape preserved).
%
%   Example:
%       small = [alpha_x alpha_y dalpha_x dalpha_y ddalpha_x ddalpha_y];
%       qI_lin = keepOrder(qI, small);          % -> [alpha_x; alpha_y; -1]

if nargin < 3, n = 1; end

bk = sym('bookkeeping_eps_', 'real');   % unlikely to collide with your states
v  = smallVars(:).';

fe = subs(f, v, bk*v);

g = sym(zeros(size(f)));
for k = 1:numel(fe)
    g(k) = subs( taylor(fe(k), bk, 'Order', n+1), bk, 1 );
end
g = simplify(g);
end


%%
qIdot = ddt(qI, Q) %[output:49d94e0b]
qIddot = ddt(qIdot, Q) %[output:0b0525ac]
%%
syms m_P m_D L g0 T real
syms sddD1 sddD2 sddD3 real
syms fx fy fz real

s_ddot_D = [sddD1; sddD2; sddD3];
f_vec    = [fx; fy; fz];
g_vec    = [0; 0; g0];

qIdot  = ddt(qI,    Q);
qIddot = ddt(qIdot, Q);

a_P = s_ddot_D + L*qIddot;

eq1 = m_P*a_P      == -T*qI - m_P*g_vec;
eq2 = m_D*s_ddot_D ==  f_vec - T*qI - m_D*g_vec;

vars = [alpha_ddot_x; alpha_ddot_y; T; s_ddot_D];
[A, b] = equationsToMatrix([eq1; eq2], vars);
sol = simplify(A\b);

alpha_ddot_x = sol(1) %[output:2a267da3]
alpha_ddot_y = sol(2) %[output:6749be13]
T            = sol(3) %[output:6fd60506]
%%
syms Jxx Jyy Jzz       positive

%% Drone states
syms px py pz          real
syms vx vy vz          real
syms phi theta psi     real
syms p q r          real

%% Inputs
syms C_Sigma n1 n2 n3  real

%% Shorthand
sp = sin(phi);   cp = cos(phi);
st = sin(theta); ct = cos(theta);
sy = sin(psi);   cy = cos(psi);

T_EB = [ct*cy,  sp*st*cy - cp*sy,  cp*st*cy + sp*sy;
        ct*sy,  sp*st*sy + cp*cy,  cp*st*sy - sp*cy;
        -st,    sp*ct,             cp*ct];

% Thrust acts along body +z, so this is C_Sigma times the third column.
F_I = T_EB*[0; 0; C_Sigma];

% T = (mP/(mD + mP))*(mD*L*(qIdot.'*qIdot) - qI.'*F_I);

%% Translational dynamics
a_D = F_I/mD + (T*qI)/mD - g0*[0; 0; 1];

%% Euler kinematics
tt = st/ct;
phi_dot   = p + (sp*q + cp*r)*tt;
theta_dot = cp*q - sp*r;
psi_dot   = (sp*q + cp*r)/ct;

%% Attitude dynamics (diagonal inertia)
wx_dot = (n1 - (Jyy - Jzz)*q*r)/Jxx;
wy_dot = (n2 - (Jzz - Jxx)*p*r)/Jyy;
wz_dot = (n3 - (Jxx - Jyy)*p*q)/Jzz;

%% Drone-side state derivative (states 1..12)
xdot_drone = [ vx; vy; vz; %[output:group:2c86f5f3] %[output:1da5a850]
               a_D; %[output:1da5a850]
               phi_dot; theta_dot; psi_dot; %[output:1da5a850]
               wx_dot;  wy_dot;    wz_dot ] %[output:group:2c86f5f3] %[output:1da5a850]
xdot_payload = [alpha_dot_x; alpha_dot_y; alpha_ddot_x; alpha_ddot_y] %[output:115704a0]

xs_drone = [px; py; pz; vx; vy; vz; phi; theta; psi; p; q; r];
xs_payload = [alpha_x; alpha_y; alpha_dot_x; alpha_dot_y];
us       = [C_Sigma; n1; n2; n3];

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":43.3}
%---
%[output:49d94e0b]
%   data: {"dataType":"symbolic","outputData":{"name":"qIdot","value":"\\left(\\begin{array}{c}\n{\\dot{\\alpha} }_x \\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)-{\\dot{\\alpha} }_y \\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\\\\\n{\\dot{\\alpha} }_y \\,\\cos \\left(\\alpha_y \\right)\\\\\n{\\dot{\\alpha} }_x \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)+{\\dot{\\alpha} }_y \\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\n\\end{array}\\right)"}}
%---
%[output:0b0525ac]
%   data: {"dataType":"symbolic","outputData":{"name":"qIddot","value":"\\left(\\begin{array}{c}\n{\\ddot{\\alpha} }_x \\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)-{\\dot{\\alpha} }_y \\,{\\left({\\dot{\\alpha} }_x \\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)+{\\dot{\\alpha} }_y \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)\\right)}-{\\dot{\\alpha} }_x \\,{\\left({\\dot{\\alpha} }_x \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)+{\\dot{\\alpha} }_y \\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\\right)}-{\\ddot{\\alpha} }_y \\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\\\\\n{\\ddot{\\alpha} }_y \\,\\cos \\left(\\alpha_y \\right)-{{\\dot{\\alpha} }_y }^2 \\,\\sin \\left(\\alpha_y \\right)\\\\\n{\\dot{\\alpha} }_x \\,{\\left({\\dot{\\alpha} }_x \\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)-{\\dot{\\alpha} }_y \\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\\right)}+{\\dot{\\alpha} }_y \\,{\\left({\\dot{\\alpha} }_y \\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)-{\\dot{\\alpha} }_x \\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\\right)}+{\\ddot{\\alpha} }_x \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)+{\\ddot{\\alpha} }_y \\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\n\\end{array}\\right)"}}
%---
%[output:2a267da3]
%   data: {"dataType":"symbolic","outputData":{"name":"alpha_ddot_x","value":"-\\frac{\\mathrm{fx}\\,\\cos \\left(\\alpha_x \\right)+\\mathrm{fz}\\,\\sin \\left(\\alpha_x \\right)-2\\,L\\,{\\dot{\\alpha} }_x \\,{\\dot{\\alpha} }_y \\,m_D \\,\\sin \\left(\\alpha_y \\right)}{L\\,m_D \\,\\cos \\left(\\alpha_y \\right)}"}}
%---
%[output:6749be13]
%   data: {"dataType":"symbolic","outputData":{"name":"alpha_ddot_y","value":"-\\frac{L\\,m_D \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_y \\right)\\,{{\\dot{\\alpha} }_x }^2 +\\mathrm{fy}\\,\\cos \\left(\\alpha_y \\right)+\\mathrm{fz}\\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)-\\mathrm{fx}\\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)}{L\\,m_D }"}}
%---
%[output:6fd60506]
%   data: {"dataType":"symbolic","outputData":{"name":"T","value":"\\frac{m_P \\,{\\left(\\mathrm{fz}\\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)-\\mathrm{fy}\\,\\sin \\left(\\alpha_y \\right)-\\mathrm{fx}\\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)+L\\,{{\\dot{\\alpha} }_y }^2 \\,m_D +L\\,{{\\dot{\\alpha} }_x }^2 \\,m_D \\,{\\cos \\left(\\alpha_y \\right)}^2 \\right)}}{m_D -m_P }"}}
%---
%[output:1da5a850]
%   data: {"dataType":"symbolic","outputData":{"name":"xdot_drone","value":"\\begin{array}{l}\n\\left(\\begin{array}{c}\n\\mathrm{vx}\\\\\n\\mathrm{vy}\\\\\n\\mathrm{vz}\\\\\n\\frac{C_{\\Sigma } \\,{\\left(\\sin \\left(\\phi \\right)\\,\\sin \\left(\\psi \\right)+\\cos \\left(\\phi \\right)\\,\\cos \\left(\\psi \\right)\\,\\sin \\left(\\theta \\right)\\right)}}{\\mathrm{mD}}+\\frac{m_P \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)\\,\\sigma_1 }{\\mathrm{mD}\\,{\\left(m_D -m_P \\right)}}\\\\\n\\frac{m_P \\,\\sin \\left(\\alpha_y \\right)\\,\\sigma_1 }{\\mathrm{mD}\\,{\\left(m_D -m_P \\right)}}-\\frac{C_{\\Sigma } \\,{\\left(\\cos \\left(\\psi \\right)\\,\\sin \\left(\\phi \\right)-\\cos \\left(\\phi \\right)\\,\\sin \\left(\\psi \\right)\\,\\sin \\left(\\theta \\right)\\right)}}{\\mathrm{mD}}\\\\\n\\frac{C_{\\Sigma } \\,\\cos \\left(\\phi \\right)\\,\\cos \\left(\\theta \\right)}{\\mathrm{mD}}-g_0 -\\frac{m_P \\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)\\,\\sigma_1 }{\\mathrm{mD}\\,{\\left(m_D -m_P \\right)}}\\\\\np+\\frac{\\sin \\left(\\theta \\right)\\,\\sigma_2 }{\\cos \\left(\\theta \\right)}\\\\\nq\\,\\cos \\left(\\phi \\right)-r\\,\\sin \\left(\\phi \\right)\\\\\n\\frac{\\sigma_2 }{\\cos \\left(\\theta \\right)}\\\\\n\\frac{n_1 -q\\,r\\,{\\left(\\mathrm{Jyy}-\\mathrm{Jzz}\\right)}}{\\mathrm{Jxx}}\\\\\n\\frac{n_2 +p\\,r\\,{\\left(\\mathrm{Jxx}-\\mathrm{Jzz}\\right)}}{\\mathrm{Jyy}}\\\\\n\\frac{n_3 -p\\,q\\,{\\left(\\mathrm{Jxx}-\\mathrm{Jyy}\\right)}}{\\mathrm{Jzz}}\n\\end{array}\\right)\\\\\n\\mathrm{}\\\\\n\\textrm{where}\\\\\n\\mathrm{}\\\\\n\\;\\;\\sigma_1 =\\mathrm{fz}\\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)-\\mathrm{fy}\\,\\sin \\left(\\alpha_y \\right)-\\mathrm{fx}\\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)+L\\,{{\\dot{\\alpha} }_y }^2 \\,m_D +L\\,{{\\dot{\\alpha} }_x }^2 \\,m_D \\,{\\cos \\left(\\alpha_y \\right)}^2 \\\\\n\\mathrm{}\\\\\n\\;\\;\\sigma_2 =r\\,\\cos \\left(\\phi \\right)+q\\,\\sin \\left(\\phi \\right)\n\\end{array}"}}
%---
%[output:115704a0]
%   data: {"dataType":"symbolic","outputData":{"name":"xdot_payload","value":"\\left(\\begin{array}{c}\n{\\dot{\\alpha} }_x \\\\\n{\\dot{\\alpha} }_y \\\\\n-\\frac{\\mathrm{fx}\\,\\cos \\left(\\alpha_x \\right)+\\mathrm{fz}\\,\\sin \\left(\\alpha_x \\right)-2\\,L\\,{\\dot{\\alpha} }_x \\,{\\dot{\\alpha} }_y \\,m_D \\,\\sin \\left(\\alpha_y \\right)}{L\\,m_D \\,\\cos \\left(\\alpha_y \\right)}\\\\\n-\\frac{L\\,m_D \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_y \\right)\\,{{\\dot{\\alpha} }_x }^2 +\\mathrm{fy}\\,\\cos \\left(\\alpha_y \\right)+\\mathrm{fz}\\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)-\\mathrm{fx}\\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)}{L\\,m_D }\n\\end{array}\\right)"}}
%---
