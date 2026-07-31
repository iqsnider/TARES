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


#block(
  stroke: 1pt + darkgreen, 
  radius: 4pt, 
  inset: 10pt,
  fill: gray.lighten(80%)
)[
=== Geometry

$
[underline(m)]^I = vec(cos psip, sin psip, 0), quad "assuming no roll or pitch"
$


#defs[
  $[underline(m)]^I$: unit vector along the marker row \
]

=== Measurement model, $j in V_k$

$
[p_j]^I = Lm [qu]^I - o_j [underline(m)]^I
$

#defs[
  $V_k$: marker IDs detected in frame $k$\
  $Lm$: drone tether-pivot to marker-board CG\
  $o_j$: marker offset from center along the row, +left\
  $[p_j]^I$: position of marker j
]

$
[p_j]^C = vec(X_j, Y_j, Z_j) = [C]^(C B) ([ell]^B + [C]^(B I) [p_j]^I - [t_(B C)]^B)
$

#defs[
  $[t_(B C)]^B$: camera optical center (offset due to science stick) [m]\
  $[ell]^B$: tether pivot point offset [m]\
]

$
h_j (xi) = vec(f_u X_j \/ Z_j + u_0, f_v Y_j \/ Z_j + v_0)
$

#defs[
  $f_u, f_v, u_0, v_0$: intrinsics  [px]\
  $h_j$: predicted pixel in image plane  
]

$
H_j = (partial h_j (xi)) / (partial [p_j]^C) thin
      (partial [p_j]^C) / (partial [p_j]^I) thin
      (partial [p_j]^I) / (partial [xi]^I)
$

$
H_j = mat(
  f_u / Z_j, 0, -(f_u X_j) / Z_j^2;
  0, f_v / Z_j, -(f_v Y_j) / Z_j^2
) thin [C]^(C B) thin [C]^(B I) thin
(Lm (partial [hat(n)]^I) / (partial xi) - o_j (partial [hat(m)]^I) / (partial xi))
$

#defs[
  $H_j in RR^(2 times 5)$: Jacobian of $h_j$ w.r.t. $xi$, by chain rule\
  $[ell]^B, [t_(B C)]^B$ are constant
]

]

=== Update

$
z^((j)) = h_j (xi) + epsilon^((j))
$

#defs[
  $m_k in {0,1,2,3}$: number of markers spotted in a given frame\
  $z^((j))$: pixel logged for marker $j$ [px]\
  $epsilon^((j))$: measurement noise
  
]

$
z_k = vec(z^((1)), dots.v, z^((m_k))), quad
h_k = vec(h_1, dots.v, h_(m_k)), quad
H_k = vec(H_1, dots.v, H_(m_k)), quad
R_k = (sdet^2 + f_u^2 sth^2) I_(2 m_k)
$

$
z_k = h_k (xi) + epsilon_k, quad quad epsilon_k tilde.op cal(N)(0, R_k)
$

$
y_k = z_k - h_k, quad
S_k = H_k P_k^- H_k^top + R_k, quad
K_k = P_k^- H_k^top S_k^(-1)
$

$
xihat_k^+ = xihat_k^- + K_k y_k
$

$
P_k^+ = (I_5 - K_k H_k) P_k^-
$

#defs[
  $y_k$: innovation $quad$ $S_k$: innovation covariance $quad$ $K_k in RR^(5 times 2 m_k)$: Kalman gain\
  $sdet$: marker detection noise [px] $quad$ $sth$: attitude 1#sym.sigma [rad]\
]

If no markers detected:
$
m_k = 0 quad ==> quad xihat_k^+ = xihat_k^-, quad P_k^+ = P_k^-
$





