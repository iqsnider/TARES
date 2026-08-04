#import "@preview/slydst:0.1.4": *
#show: slides.with(
  layout: "large",
  ratio: 4/3,
)
#let cropped(path, top: 0%, bottom: 0%) = layout(size => {
  let img = image(path, width: size.width)
  let h = measure(img).height
  box(height: h - h * (top + bottom), clip: true, move(dy: -h * top, img))
})

#set text(size: 8pt)
#set page(margin: (bottom: 0cm))
== Payload trajectory control progress
- Collected data from camera tracking a payload tethered beneath the drone
- Used the data to design an Extended Kalman Filter (EKF) for estimating the payload pose
#columns(2, gutter: 4pt)[
#figure(
  cropped("curated_figures/ekf_post_hoc_on_lqr_no_payload_control.png", top: -10%, bottom: 8%),
  caption: [Drone tracking a trajectory with the payload as an unmodeled disturbance. EKF estimates are post-hoc; payload control is not yet closed-loop.],
)
#figure(
  cropped("curated_figures/simulated_ekf_testing_drone_recovering_from_perturbation.png", top: -10%, bottom: 8%),
  caption: [Simulation with EKF and payload control closed-loop: recovery from an initial perturbation, then trajectory tracking.],
)
]
