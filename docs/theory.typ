#import "@preview/cetz:0.3.4": canvas, draw, matrix
#import draw: *

#set math.equation(numbering: "(1)")
#set page(margin: 50pt)
#set text(size: 11pt)

#let defs(body) = block(
  inset: (left: 1.3em), above: 0.55em, below: 1.0em,
  text(size: 8.8pt, fill: rgb("#333333"), body))
#let darkgreen = rgb("#008000")

#let mp = $m_P$
#let ap = $underline(a)_P$
#let gu = $underline(g)$
#let qu = $underline(q)$
#let md = $m_D$
#let ad = $underline(a)_D$
#let fu = $underline(f)$
#let ddt(y)  = $(dif #y) / (dif t)$
#let omegaBE = $underline(omega)^(B E)$
#let OmegaBE = $underline(Omega)^(B E)$
#let JBB = $underline(J)_B^B$
#let nB = $underline(n)_B$
#let sp = $underline(s)_P$
#let sd = $underline(s)_D$
#let ddotsd = $dot.double(underline(s))_D$
#let ddotsp = $dot.double(underline(s))_P$
#let dotq = $dot(underline(q))$
#let omegau = $underline(omega)$
#let skew = $underline(K)$
#let ddotq = $dot.double(underline(q))$
#let Omegau = $underline(Omega)$

// ekf variables
#let atantwo = math.op("atan2")

#let argmin = math.op("arg min", limits: true)
#let Lm = $L_"m"$
#let sdet = $sigma_"det"$
#let sth = $sigma_theta$
#let ax = $alpha_x$
#let ay = $alpha_y$
#let dax = $dot(alpha)_x$
#let day = $dot(alpha)_y$
#let psip = $psi_P$
#let xihat = $hat(xi)$


#block(
  stroke: 1pt + red, 
  radius: 4pt, 
  inset: 10pt,
  fill: gray.lighten(80%)
)[
= UAV Swing Payload Model
#v(0.5cm)
  $ mp ap = T qu - mp gu $
$ md ad = fu - T qu - md gu $
$ underline(ddt("")omega^(B E)) = (JBB)^(-1)[nB - OmegaBE JBB omegaBE] $
$ sp = sd - L qu $

// $ ap = ddotsp $
// $ ad = ddotsd $
// $ dotq = skew(omegau) qu $
// $ ddotq = skew(dot(omegau))qu + skew(omegau)skew(omegau) qu \ 
// = dot(Omegau)qu - norm(omega)^2 qu $
// === tension derivation
// $ ad - ap = L ddotq \
// = L(dot(Omegau) qu - norm(omegau)||^2 qu) \
// = (1/md) fu - T qu (1/md + 1/mp) $
//
// $ arrow.double.r T = frac(fu/md - L (dot(Omegau)qu - norm(omegau)^2 qu), qu(1/md + 1/mp)) $

=== Swing angles
#v(0.5cm)
#align(center)[
#canvas({
  // Set up the transformation matrix
  ortho(x: -70deg, y: 0deg, z: -110deg, {
    let axis-style = (stroke: black, mark: (end: ">", fill: black, scale: 0.5))
    let red-style = (stroke: red + 1.5pt, mark: (end: ">", fill: red, scale: 0.5))
    let blue-style = (stroke: blue + 1.5pt, mark: (end: ">", fill: blue, scale: 0.5))
    let green-style = (stroke: green + 1.5pt, mark: (end: ">", fill: green, scale: 0.5))

    // Coordinate axes arrows
    line((0, 0, 0), (3, 0, 0), ..axis-style)
    line((0, 0, 0), (0, 3, 0), ..axis-style)
    line((0, 0, 0), (0, 0, 2), ..axis-style)
    line((0, 0, 0), (0, 0, -3), stroke: (dash: "dashed"))

    // Position content on specific 3D axes points
    content((3.5, 0, 0), [$underline(1)^I$])
    content((0, 3.5, 0), [$underline(2)^I$])
    content((0, 0, 2.5), [$underline(3)^I$])

    // vectors of interest
    line((0, 0, 0), (2, 0, -2), ..red-style)
    line((0, 0, 0), (0, 2, -2), ..blue-style)
    line((0, 0, 0), (2, 2, -2), ..green-style)
    content((2.3, 2.3, -2.3), text(fill: green)[$[underline(q)]^I$])

    // shaded planes
    line((0, 0, 0), (3, 0, 0), (3, 0, -3), (0, 0, -3),
         close: true,
         fill: red.transparentize(90%),
         stroke: none)

    line((0, 0, 0), (0, 3, 0), (0, 3, -3), (0, 0, -3),
         close: true,
         fill: blue.transparentize(90%),
         stroke: none)

    // arcs
    on-xz({
          arc((0, 0), start: -90deg, stop: -45deg, radius: 1.2,
              anchor: "origin",
              stroke: red,
              mark: (end: ">", fill: red, scale: .5))
          content((0.6, -1.4), text(fill: red)[$alpha_x$])
        })

    on-yz({
          arc((0, 0), start: 180deg, stop: 135deg, radius: 1.2,
              anchor: "origin",
              stroke: blue,
              mark: (end: ">", fill: blue, scale: 0.5))
          content((-1.4, 0.6), text(fill: blue)[$alpha_y$])
        })


  })
})
]
$ [qu]^I = mat(cos alpha_x, 0, -sin alpha_x; 0, 1, 0; sin alpha_x, 0, cos alpha_x) mat(1, 0, 0; 0, cos alpha_y, -sin alpha_y; 0, sin alpha_y, cos alpha_y) [-underline(3)^I]^I = vec(sin alpha_x cos alpha_y, sin alpha_y, -cos alpha_x cos alpha_y) approx vec(alpha_x, alpha_y, -1) $
]

#block(
  stroke: 1pt + darkgreen, 
  radius: 4pt, 
  inset: 10pt,
  fill: gray.lighten(80%)
)[

== ArUco Marker Tracking EKF

=== Frames


$
[dot.c]^I : "ENU" quad (+x "East", +y "North", +z "Up")\
[dot.c]^B : "body" quad (+x "nose", +y "portside", +z "up")\
[dot.c]^C : "camera" quad (+x "right", +y "up", +z "optical axis")
$




=== State

$
[xi]^I = vec(alpha_x, alpha_y, dot(alpha)_x, dot(alpha)_y, psi_P) in RR^5
$

$
s_x = sin ax, quad c_x = cos ax, quad s_y = sin ay, quad c_y = cos ay
$

#defs[
  $ax, ay$: tether swing angles, tipping toward $+x^I$ (East) and $+y^I$ (North) $quad$ [rad]\
  $dax, day$: swing rates $quad$ [rad/s]\
  $psip$: payload yaw: marker-row direction $quad$ [rad]\
]

]

#block(
  stroke: 1pt + darkgreen, 
  radius: 4pt, 
  inset: 10pt,
  fill: gray.lighten(80%)
)[
=== Input

$
[T]^(I B) = mat(
  c_theta c_psi, s_phi s_theta c_psi - c_phi s_psi, c_phi s_theta c_psi + s_phi s_psi;
  c_theta s_psi, s_phi s_theta s_psi + c_phi c_psi, c_phi s_theta s_psi - s_phi c_psi;
  -s_theta,      s_phi c_theta,                     c_phi c_theta
)
$

#defs[Euler angles: $quad$
  $phi$: roll $quad$
  $theta$: pitch $quad$
  $psi$: yaw
]

$ [ad]^I = vec(a_1, a_2, a_3) = [T]^(I B) [ad]^B $


#defs[
  $[ad]^I$: acceleration inputs from the drone $quad$ [m/ s$""^2$]
]


=== Process model

$
[dot(xi)]^I = f(xi, a) = vec(
  dax,
  day,
  - (c_x a_1 + s_x a_3) / (L c_y) + 2 s_y / c_y dax day,
  - (c_y a_2 + s_y (c_x a_3 - s_x a_1)) / L - s_y c_y dax^2,
  0
)
 approx vec(dot(alpha)_x, dot(alpha)_y, -1/L (a_1 + alpha_x norm(g)), -1/L (a_2 + alpha_y norm(g)), 0) $

#defs[
$L$: drone tether-pivot to payload CG [m]\
]

$ F = mat(0, 0, 1, 0, 0;
          0, 0, 0, 1, 0;
          -norm(g)/L, 0,0,0,0;
          0, -norm(g)/L, 0,0,0;
        0,0,0,0,0) $


#defs[
  $F in RR^(5 times 5)$: process Jacobian (continuous-time)
]
]
#block(
  stroke: 1pt + darkgreen, 
  inset: 10pt, 
  radius: 4pt, 
  fill: gray.lighten(80%)
)[
=== Prediction
Jacobian to discrete-time conversion w/ Euler integration.
$
xihat_(k+1)^- = xihat_k^+ + Delta t thin f(xihat_k^+, a_s)
$
$
Phi_k = I_5 + F Delta t, quad quad
Q = "diag"(0, 0, q_x, q_y, q_psi) Delta t
$
$
P_(k+1)^- = Phi_k P_k^+ Phi_k^top + Q
$
#defs[
  $Delta t$: time step [s]\
  $xihat_k^-, xihat_k^+$: estimate before \& after the update at step $k$\
  $P_k^-, P_k^+ in RR^(5 times 5)$: covariance before \& after\
  $Phi_k$: discrete time Jacobian\
  $q$: process-noise intensity
]
]

#v(0.5cm)
#text(size: 9pt)[
Shelving Jacobian to discrete-time conversion w/ RK4 for now.
$
k_1 &= f(xihat_k^+, a_s), quad
k_2 = f(xihat_k^+ + (Delta t) / 2 k_1, a_s)\
k_3 &= f(xihat_k^+ + (Delta t) / 2 k_2, a_s), quad
k_4 = f(xihat_k^+ + Delta t thin k_3, a_s)
$

$
xihat_(k+1)^- = xihat_k^+ + (Delta t) / 6 (k_1 + 2 k_2 + 2 k_3 + k_4)
$

$
Phi_k = I_5 + F Delta t + 1/2 (F Delta t)^2, quad quad
Q = "diag"(0, 0, q, q, q_psi) Delta t
$

$
P_(k+1)^- = Phi_k P_k^+ Phi_k^top + Q
$

#defs[
  $Delta t$: time step [s]\
  $xihat_k^-, xihat_k^+$: estimate before \& after the update at step $k$\
  $P_k^-, P_k^+ in RR^(5 times 5)$: covariance before \& after\
  $Phi_k$: discrete time Jacobian\
  $q$: process-noise intensity
]
]

#let VK = $V_k$
#let mC = $[underline(m)]^C$
#let ell = $underline(ell)$
#let tBC = $underline(t)_(B C)$

#block(
  stroke: 1pt + darkgreen, 
  radius: 4pt, 
  inset: 10pt,
  fill: gray.lighten(80%)
)[
== Measurement
=== Pre-processing logic

For every $j in VK$, `solvePnP` returns the marker pose in the camera frame,

$
[underline(t)_j]^C, quad [T]^(C M_j) quad quad j in VK
$

#defs[
$VK$: marker IDs detected in frame $k$\
$[underline(t)_j]^C$: marker $j$ center in the camera frame [m]\
$[T]^(C M_j)$: transformation to the camera frame from the marker local frame $M_j$
]

The first column of $[T]^(C M_j)$ is the marker $x$-axis, which runs along
the marker row,

$
[underline(m)_j]^C = [T]^(C M_j) [underline(1)^(M_j)]^(M_j)
$

Each marker gives its own estimate of the board center; average over the
markers actually detected in the frame,

$
[underline(p)_"ctr"]^C = 1/m_k sum_(j in VK) ( [underline(t)_j]^C + o_j [underline(m)_j]^C )
$

#defs[
  $[underline(p)_"ctr"]$: payload center in the camera frame \
  $m_k$: \# of markers in the frame\
  $o_j$: marker offset from board center along the row, $+$left [m]\
  ]

Shift the origin from the camera optical center to the tether pivot,

$
[underline(p)]^C = [underline(p)_"ctr"]^C + [T]^(C B) ( [tBC]^B - [ell]^B )
$

#defs[
  $[underline(p)]^C$: payload center after the camera offset is corrected, camera frame [m]\
  $[tBC]^B$: camera optical center in the body frame [m]\
  $[ell]^B$: drone tether pivot in the body frame [m]\
]

Average the payload yaw and transform into the inertial frame,

$
[underline(m)]^I = [T]^(I B) [T]^(B C) ( 1/m_k sum_(j in VK) [underline(m)_j]^C )\
z_(psi_P) = atantwo(m_2^I, m_1^I)
$

=== Measurment model

After some pre-processing from pixels to camera frame coordinates, we measure,

  $ z_k = vec(frac(p_x^C, norm(underline(p))), frac(p_y^C, norm(underline(p))), z_(psi_P)) + epsilon_k $

#defs[
    $underline(p)$: payload center position\
    $z_(psi_p)$: measured payload yaw\
    $epsilon_k$: measurement noise
]

=== Prediction of measurement
Measurement predicted from swing angles,

  $ h(xi) = vec(q_x^C, q_y^C, psi_P) quad quad quad [qu]^C = [T]^(C B)[T]^(B I) vec(alpha_x, alpha_y, -1) $

$
H = frac(partial h, partial xi) = vec(
  mat(1,0,0; 0,1,0) [T]^(C B) [T]^(B I) mat(1,0,0,0,0; 0,1,0,0,0; 0,0,0,0,0),
  mat(0, 0, 0, 0, 1)
)
$

]

#let sx = $sigma_x$
#let sy = $sigma_y$
#let spsi = $sigma_psi$
#let wrap = math.op("wrap")

#block(
  stroke: 1pt + darkgreen, 
  radius: 4pt, 
  inset: 10pt,
  fill: gray.lighten(80%)
)[
== Update

=== Innovation

$ y_k = z_k - h_k (hat(xi)_k^-) $

$
R_k = "diag"(sx^2, sy^2, spsi^2) in RR^(3 times 3)
$

$
S_k = H_k P_k^- H_k^top + R_k in RR^(3 times 3)
$

#defs[
    $y_k$: innovation\
    $R_k$: measurement noise\
    $S_K$: innovation covariance
] 


=== Kalman gain
$
K_k = P_k^- H_k^top S_k^(-1) in RR^(5 times 3)
$

#defs[
  $K_k in RR^(5 times 3)$: Kalman gain\
]
$
hat(xi)_k^+ = hat(xi)_k^- + K_k y_k, quad quad
$

$
P_k^+ = (I_5 - K_k H_k) P_k^-
$


If no markers detected:
$
xihat_k^+ = xihat_k^-, quad P_k^+ = P_k^-
$

]

#block(
  stroke: 1pt + purple, 
  radius: 4pt, 
  inset: 10pt,
  fill: gray.lighten(80%)
)[
= Control
=== State-Space
  $ dot(x) = A x + B u $
where,
  $ x = vec(s_1^I, s_2^I, s_3^I, v_1^I, v_2^I, v_3^I, alpha_x^I, alpha_y^I, dot(alpha)_x^I, dot(alpha)_y^I), quad u = vec(a_1^I, a_2^I, a_3^I) $

  $ "drone states" cases(dot(s)_1^I = v_1^I,
  dot(s)_2^I = v_2^I ,
  dot(s)_3^I = v_3^I ,
  dot(v)_1^I = a_1^I ,
  dot(v)_2^I = a_2^I ,
  dot(v)_3^I = 0) $

  $ "payload states" cases(dot(alpha)_x^I = dot(alpha)_x^I ,
  dot(alpha)_y^I = dot(alpha)_y^I ,
  dot.double(alpha)_x^I = -g/L dot(alpha)_x^I - 1/L a_1^I ,
  dot.double(alpha)_y^I = -g/L dot(alpha)_y^I - 1/L a_2^I) $

  $ arrow.r.double A, B $

=== Equilibrium

  $ x^* = vec([underline(s)_(P L) (t)]^I + vec(0,0,L), [underline(v)_(P L)]^I, 0 ) $


]

#block(
  stroke: 1pt + purple, 
  radius: 4pt, 
  inset: 10pt,
  fill: gray.lighten(80%)
)[
== LQI Controller

=== Control Law
  $ u = -mat(K_x, K_I) vec(e, integral e)\ e = x - x^* $

  $ y_"error" = C e $
  $ integral e[k + 1] = integral e[k] + y_"error" Delta t $

=== Output
  $ y = C x $

  $ y = [underline(s)_(P L)]^I\
  = vec(s_1^I - L alpha_x^I, s_2^I - L alpha_y^I, s_3^I - L) $

  $ arrow.r.double C $

=== Gain matrix
  $ overline(K) = mat(K_x, K_I) $

=== Augmented State-space

  $ dot(z) = overline(A) z + overline(B) u $
  $ z = vec(x, integral [underline(s)_(P L)]^I ) $
  $ arrow.r.double overline(A), overline(B) $

=== Cost minimization

Solve for the optimal $overline(K)$,

  $ J = integral_0^oo ( vec(e, integral e)^top overline(Q) vec(e, integral e)  + u^top R u ) dif t $

State weighting matrix,
  $ overline(Q) = mat(C^top W C, 0_(10 times 3); 0_(3 times 10), W_"int") in RR^(13 times 13) $
  $ W = mat(w_x,0,0;0,w_y,0;0,0,w_z) $
  $ W_"int" = mat(w_("int",x),0,0;0,w_("int", y),0;0,0,w_("int",z)) $

Control weighting matrix,
  $ R = ("tuning const") times mat(1,0,0;0,1,0;0,0,1) $

Solving for the gain matrix,
  $ overline(A)^top P + P overline(A) - P overline(B) R^(-1) overline(B)^top P + overline(Q) = 0 arrow.r.double P $
  $ overline(K) = R^(-1) overline(B)^top P $

]
