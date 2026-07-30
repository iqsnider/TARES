syms alpha_x  alpha_y
syms alpha_dot_x alpha_dot_y
syms alpha_ddot_x alpha_ddot_y

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
% computing ddt chain rules

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
syms m_P m_D L g0 T
syms sddD1 sddD2 sddD3
syms fx fy fz

s_ddot_D = [sddD1; sddD2; sddD3]; % uncoordinated, but cancels out later
f_vec  = [fx; fy; fz]; % inertial frame (computed by transforming C_Sigma from body to inertial)
g_vec  = [0; 0; g0]; % inertial frame

a_P = s_ddot_D + L*qIddot;

eq1 = m_P*a_P == T*qI - m_P*g_vec;
eq2 = m_D*s_ddot_D == f_vec - T*qI - m_D*g_vec;

vars = [alpha_ddot_x; alpha_ddot_y; T; s_ddot_D];
[A, b] = equationsToMatrix([eq1; eq2], vars);
sol = simplify(A\b);

alpha_ddot_x = sol(1) %[output:3f67bf19]
alpha_ddot_y = sol(2) %[output:19313db0]
T = sol(3) %[output:184b6d58]
%%
syms Jxx Jyy Jzz

% Drone states
syms px py pz
syms vx vy vz
syms phi theta psi
syms p q r

% Inputs
syms C_Sigma n1 n2 n3

% Shorthand
sp = sin(phi); cp = cos(phi);
st = sin(theta); ct = cos(theta);
sy = sin(psi); cy = cos(psi);

T_EB = [ct*cy,  sp*st*cy - cp*sy,  cp*st*cy + sp*sy;
        ct*sy,  sp*st*sy + cp*cy,  cp*st*sy - sp*cy;
        -st, sp*ct, cp*ct];

% Thrust acts along body +z, so this is C_Sigma times the third column.
F_I = T_EB*[0; 0; C_Sigma];

% T = (mP/(mD + mP))*(mD*L*(qIdot.'*qIdot) - qI.'*F_I);

% Translational dynamics
a_D = F_I/m_D - (T*qI)/m_D - g0*[0; 0; 1];

% Euler kinematics
tt = st/ct;
phi_dot   = p + (sp*q + cp*r)*tt;
theta_dot = cp*q - sp*r;
psi_dot   = (sp*q + cp*r)/ct;

% Attitude dynamics
wx_dot = (n1 - (Jyy - Jzz)*q*r)/Jxx;
wy_dot = (n2 - (Jzz - Jxx)*p*r)/Jyy;
wz_dot = (n3 - (Jxx - Jyy)*p*q)/Jzz;

% Drone-side state derivative
xdot_drone = [ vx; vy; vz; %[output:group:2c86f5f3] %[output:2a48970a]
               a_D; %[output:2a48970a]
               phi_dot; theta_dot; psi_dot; %[output:2a48970a]
               wx_dot;  wy_dot;    wz_dot ] %[output:group:2c86f5f3] %[output:2a48970a]
xdot_payload = [alpha_dot_x; alpha_dot_y; alpha_ddot_x; alpha_ddot_y] %[output:934c0bc2]

xs_drone = [px; py; pz; vx; vy; vz; phi; theta; psi; p; q; r];
xs_payload = [alpha_x; alpha_y; alpha_dot_x; alpha_dot_y];
us       = [C_Sigma; n1; n2; n3];
%%
% EKF derivation
syms psi_p
xi_I = [alpha_x %[output:group:6868ac45] %[output:5fe9a1cc]
    alpha_y %[output:5fe9a1cc]
    alpha_dot_x %[output:5fe9a1cc]
    alpha_dot_y %[output:5fe9a1cc]
    psi_p] %[output:group:6868ac45] %[output:5fe9a1cc]

xi_dot_I = [alpha_dot_x %[output:group:8ca6abf8] %[output:77ef35c4]
    alpha_dot_y %[output:77ef35c4]
    alpha_ddot_x %[output:77ef35c4]
    alpha_ddot_y %[output:77ef35c4]
    0] %[output:group:8ca6abf8] %[output:77ef35c4]

F = jacobian(xi_dot_I, xi_I) %[output:1aab32ec]


% Measurement scalars
syms o_j X_j Y_j Z_j f_u f_v u_0 v_0 L_m
l_B    = symmatrix('l_B', [3 1]);
t_BC_B = symmatrix('t_BC_B', [3 1]);

nI = qI %[output:6a3f5115]
mI = [cos(psi_p) %[output:group:9813c335] %[output:180a574d]
    sin(psi_p) %[output:180a574d]
    0] %[output:group:9813c335] %[output:180a574d]

C_IB = symmatrix('C_IB', [3 3]);
C_CB = symmatrix('C_CB', [3 3]);
C_BI = C_IB.';

pI_j = L_m*nI - o_j*mI;
pC_j = C_CB*(l_B + C_BI*symmatrix(pI_j) - t_BC_B);

XYZ = pC_j;

h_j = [f_u*(X_j/Z_j) + u_0
    f_v*(Y_j/Z_j) + v_0];

dh_dpC  = jacobian(h_j, [X_j; Y_j; Z_j]);
dpI_dxi = jacobian(pI_j, xi_I);

H_j = symmatrix(dh_dpC) * C_CB*C_BI * symmatrix(dpI_dxi) %[output:15d3e504]

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":51.6}
%---
%[output:49d94e0b]
%   data: {"dataType":"symbolic","outputData":{"name":"qIdot","value":"\\left(\\begin{array}{c}\n{\\dot{\\alpha} }_x \\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)-{\\dot{\\alpha} }_y \\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\\\\\n{\\dot{\\alpha} }_y \\,\\cos \\left(\\alpha_y \\right)\\\\\n{\\dot{\\alpha} }_x \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)+{\\dot{\\alpha} }_y \\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\n\\end{array}\\right)"}}
%---
%[output:0b0525ac]
%   data: {"dataType":"symbolic","outputData":{"name":"qIddot","value":"\\left(\\begin{array}{c}\n{\\ddot{\\alpha} }_x \\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)-{\\dot{\\alpha} }_y \\,{\\left({\\dot{\\alpha} }_x \\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)+{\\dot{\\alpha} }_y \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)\\right)}-{\\dot{\\alpha} }_x \\,{\\left({\\dot{\\alpha} }_x \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)+{\\dot{\\alpha} }_y \\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\\right)}-{\\ddot{\\alpha} }_y \\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\\\\\n{\\ddot{\\alpha} }_y \\,\\cos \\left(\\alpha_y \\right)-{{\\dot{\\alpha} }_y }^2 \\,\\sin \\left(\\alpha_y \\right)\\\\\n{\\dot{\\alpha} }_x \\,{\\left({\\dot{\\alpha} }_x \\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)-{\\dot{\\alpha} }_y \\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\\right)}+{\\dot{\\alpha} }_y \\,{\\left({\\dot{\\alpha} }_y \\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)-{\\dot{\\alpha} }_x \\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\\right)}+{\\ddot{\\alpha} }_x \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)+{\\ddot{\\alpha} }_y \\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)\n\\end{array}\\right)"}}
%---
%[output:3f67bf19]
%   data: {"dataType":"symbolic","outputData":{"name":"alpha_ddot_x","value":"-\\frac{\\mathrm{fx}\\,\\cos \\left(\\alpha_x \\right)+\\mathrm{fz}\\,\\sin \\left(\\alpha_x \\right)-2\\,L\\,{\\dot{\\alpha} }_x \\,{\\dot{\\alpha} }_y \\,m_D \\,\\sin \\left(\\alpha_y \\right)}{L\\,m_D \\,\\cos \\left(\\alpha_y \\right)}"}}
%---
%[output:19313db0]
%   data: {"dataType":"symbolic","outputData":{"name":"alpha_ddot_y","value":"-\\frac{L\\,m_D \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_y \\right)\\,{{\\dot{\\alpha} }_x }^2 +\\mathrm{fy}\\,\\cos \\left(\\alpha_y \\right)+\\mathrm{fz}\\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)-\\mathrm{fx}\\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)}{L\\,m_D }"}}
%---
%[output:184b6d58]
%   data: {"dataType":"symbolic","outputData":{"name":"T","value":"-\\frac{m_P \\,{\\left(\\mathrm{fz}\\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)-\\mathrm{fy}\\,\\sin \\left(\\alpha_y \\right)-\\mathrm{fx}\\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)+L\\,{{\\dot{\\alpha} }_y }^2 \\,m_D +L\\,{{\\dot{\\alpha} }_x }^2 \\,m_D \\,{\\cos \\left(\\alpha_y \\right)}^2 \\right)}}{m_D +m_P }"}}
%---
%[output:2a48970a]
%   data: {"dataType":"symbolic","outputData":{"name":"xdot_drone","value":"\\begin{array}{l}\n\\left(\\begin{array}{c}\n\\mathrm{vx}\\\\\n\\mathrm{vy}\\\\\n\\mathrm{vz}\\\\\n\\frac{C_{\\Sigma } \\,{\\left(\\sin \\left(\\phi \\right)\\,\\sin \\left(\\psi \\right)+\\cos \\left(\\phi \\right)\\,\\cos \\left(\\psi \\right)\\,\\sin \\left(\\theta \\right)\\right)}}{m_D }+\\frac{m_P \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)\\,\\sigma_1 }{m_D \\,{\\left(m_D +m_P \\right)}}\\\\\n\\frac{m_P \\,\\sin \\left(\\alpha_y \\right)\\,\\sigma_1 }{m_D \\,{\\left(m_D +m_P \\right)}}-\\frac{C_{\\Sigma } \\,{\\left(\\cos \\left(\\psi \\right)\\,\\sin \\left(\\phi \\right)-\\cos \\left(\\phi \\right)\\,\\sin \\left(\\psi \\right)\\,\\sin \\left(\\theta \\right)\\right)}}{m_D }\\\\\n\\frac{C_{\\Sigma } \\,\\cos \\left(\\phi \\right)\\,\\cos \\left(\\theta \\right)}{m_D }-g_0 -\\frac{m_P \\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)\\,\\sigma_1 }{m_D \\,{\\left(m_D +m_P \\right)}}\\\\\np+\\frac{\\sin \\left(\\theta \\right)\\,\\sigma_2 }{\\cos \\left(\\theta \\right)}\\\\\nq\\,\\cos \\left(\\phi \\right)-r\\,\\sin \\left(\\phi \\right)\\\\\n\\frac{\\sigma_2 }{\\cos \\left(\\theta \\right)}\\\\\n\\frac{n_1 -q\\,r\\,{\\left(\\mathrm{Jyy}-\\mathrm{Jzz}\\right)}}{\\mathrm{Jxx}}\\\\\n\\frac{n_2 +p\\,r\\,{\\left(\\mathrm{Jxx}-\\mathrm{Jzz}\\right)}}{\\mathrm{Jyy}}\\\\\n\\frac{n_3 -p\\,q\\,{\\left(\\mathrm{Jxx}-\\mathrm{Jyy}\\right)}}{\\mathrm{Jzz}}\n\\end{array}\\right)\\\\\n\\mathrm{}\\\\\n\\textrm{where}\\\\\n\\mathrm{}\\\\\n\\;\\;\\sigma_1 =\\mathrm{fz}\\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)-\\mathrm{fy}\\,\\sin \\left(\\alpha_y \\right)-\\mathrm{fx}\\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)+L\\,{{\\dot{\\alpha} }_y }^2 \\,m_D +L\\,{{\\dot{\\alpha} }_x }^2 \\,m_D \\,{\\cos \\left(\\alpha_y \\right)}^2 \\\\\n\\mathrm{}\\\\\n\\;\\;\\sigma_2 =r\\,\\cos \\left(\\phi \\right)+q\\,\\sin \\left(\\phi \\right)\n\\end{array}"}}
%---
%[output:934c0bc2]
%   data: {"dataType":"symbolic","outputData":{"name":"xdot_payload","value":"\\left(\\begin{array}{c}\n{\\dot{\\alpha} }_x \\\\\n{\\dot{\\alpha} }_y \\\\\n-\\frac{\\mathrm{fx}\\,\\cos \\left(\\alpha_x \\right)+\\mathrm{fz}\\,\\sin \\left(\\alpha_x \\right)-2\\,L\\,{\\dot{\\alpha} }_x \\,{\\dot{\\alpha} }_y \\,m_D \\,\\sin \\left(\\alpha_y \\right)}{L\\,m_D \\,\\cos \\left(\\alpha_y \\right)}\\\\\n-\\frac{L\\,m_D \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_y \\right)\\,{{\\dot{\\alpha} }_x }^2 +\\mathrm{fy}\\,\\cos \\left(\\alpha_y \\right)+\\mathrm{fz}\\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)-\\mathrm{fx}\\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)}{L\\,m_D }\n\\end{array}\\right)"}}
%---
%[output:5fe9a1cc]
%   data: {"dataType":"symbolic","outputData":{"name":"xi_I","value":"\\left(\\begin{array}{c}\n\\alpha_x \\\\\n\\alpha_y \\\\\n{\\dot{\\alpha} }_x \\\\\n{\\dot{\\alpha} }_y \\\\\n\\psi_p \n\\end{array}\\right)"}}
%---
%[output:77ef35c4]
%   data: {"dataType":"symbolic","outputData":{"name":"xi_dot_I","value":"\\left(\\begin{array}{c}\n{\\dot{\\alpha} }_x \\\\\n{\\dot{\\alpha} }_y \\\\\n-\\frac{\\mathrm{fx}\\,\\cos \\left(\\alpha_x \\right)+\\mathrm{fz}\\,\\sin \\left(\\alpha_x \\right)-2\\,L\\,{\\dot{\\alpha} }_x \\,{\\dot{\\alpha} }_y \\,m_D \\,\\sin \\left(\\alpha_y \\right)}{L\\,m_D \\,\\cos \\left(\\alpha_y \\right)}\\\\\n-\\frac{L\\,m_D \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_y \\right)\\,{{\\dot{\\alpha} }_x }^2 +\\mathrm{fy}\\,\\cos \\left(\\alpha_y \\right)+\\mathrm{fz}\\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)-\\mathrm{fx}\\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)}{L\\,m_D }\\\\\n0\n\\end{array}\\right)"}}
%---
%[output:1aab32ec]
%   data: {"dataType":"symbolic","outputData":{"name":"F","value":"\\left(\\begin{array}{ccccc}\n0 & 0 & 1 & 0 & 0\\\\\n0 & 0 & 0 & 1 & 0\\\\\n-\\frac{\\mathrm{fz}\\,\\cos \\left(\\alpha_x \\right)-\\mathrm{fx}\\,\\sin \\left(\\alpha_x \\right)}{L\\,m_D \\,\\cos \\left(\\alpha_y \\right)} & 2\\,{\\dot{\\alpha} }_x \\,{\\dot{\\alpha} }_y -\\frac{\\sin \\left(\\alpha_y \\right)\\,{\\left(\\mathrm{fx}\\,\\cos \\left(\\alpha_x \\right)+\\mathrm{fz}\\,\\sin \\left(\\alpha_x \\right)-2\\,L\\,{\\dot{\\alpha} }_x \\,{\\dot{\\alpha} }_y \\,m_D \\,\\sin \\left(\\alpha_y \\right)\\right)}}{L\\,m_D \\,{\\cos \\left(\\alpha_y \\right)}^2 } & \\frac{2\\,{\\dot{\\alpha} }_y \\,\\sin \\left(\\alpha_y \\right)}{\\cos \\left(\\alpha_y \\right)} & \\frac{2\\,{\\dot{\\alpha} }_x \\,\\sin \\left(\\alpha_y \\right)}{\\cos \\left(\\alpha_y \\right)} & 0\\\\\n\\frac{\\mathrm{fx}\\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)+\\mathrm{fz}\\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right)}{L\\,m_D } & \\frac{\\mathrm{fy}\\,\\sin \\left(\\alpha_y \\right)-\\mathrm{fz}\\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)+\\mathrm{fx}\\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)+L\\,{{\\dot{\\alpha} }_x }^2 \\,m_D \\,{\\sin \\left(\\alpha_y \\right)}^2 -L\\,{{\\dot{\\alpha} }_x }^2 \\,m_D \\,{\\cos \\left(\\alpha_y \\right)}^2 }{L\\,m_D } & -2\\,{\\dot{\\alpha} }_x \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_y \\right) & 0 & 0\\\\\n0 & 0 & 0 & 0 & 0\n\\end{array}\\right)"}}
%---
%[output:6a3f5115]
%   data: {"dataType":"symbolic","outputData":{"name":"nI","value":"\\left(\\begin{array}{c}\n\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right)\\\\\n\\sin \\left(\\alpha_y \\right)\\\\\n-\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right)\n\\end{array}\\right)"}}
%---
%[output:180a574d]
%   data: {"dataType":"symbolic","outputData":{"name":"mI","value":"\\left(\\begin{array}{c}\n\\cos \\left(\\psi_p \\right)\\\\\n\\sin \\left(\\psi_p \\right)\\\\\n0\n\\end{array}\\right)"}}
%---
%[output:15d3e504]
%   data: {"dataType":"symbolic","outputData":{"name":"H_j","value":"\\begin{array}{l}\n{\\mathbf{\\Sigma }}_{\\mathbf{1}} \\,{\\bm{C}}_{\\textrm{CB}} \\,{{\\bm{C}}_{\\textrm{IB}} }^{\\textrm{T}} \\,{\\mathbf{\\Sigma }}_{\\mathbf{2}} \\\\\n\\mathrm{}\\\\\n\\textrm{where}\\\\\n\\mathrm{}\\\\\n\\;\\;{\\mathbf{\\Sigma }}_{\\mathbf{1}} =\\left(\\begin{array}{ccc}\n\\frac{f_u }{Z_j } & 0 & -\\frac{X_j \\,f_u }{{Z_j }^2 }\\\\\n0 & \\frac{f_v }{Z_j } & -\\frac{Y_j \\,f_v }{{Z_j }^2 }\n\\end{array}\\right)\\\\\n\\mathrm{}\\\\\n\\;\\;{\\mathbf{\\Sigma }}_{\\mathbf{2}} =\\left(\\begin{array}{ccccc}\nL_m \\,\\cos \\left(\\alpha_x \\right)\\,\\cos \\left(\\alpha_y \\right) & -L_m \\,\\sin \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right) & 0 & 0 & o_j \\,\\sin \\left(\\psi_p \\right)\\\\\n0 & L_m \\,\\cos \\left(\\alpha_y \\right) & 0 & 0 & -o_j \\,\\cos \\left(\\psi_p \\right)\\\\\nL_m \\,\\cos \\left(\\alpha_y \\right)\\,\\sin \\left(\\alpha_x \\right) & L_m \\,\\cos \\left(\\alpha_x \\right)\\,\\sin \\left(\\alpha_y \\right) & 0 & 0 & 0\n\\end{array}\\right)\n\\end{array}"}}
%---
