# Double Pendulum – Chaotic System Simulation

This task simulates and animates the motion of a **double pendulum**, a well-known example of deterministic chaos in classical mechanics.

## What it is

A double pendulum consists of one pendulum attached to the end of another. While the system is simple in structure, it exhibits highly sensitive dependence on initial conditions, making it a classic example of chaotic behavior.

## Physical Model

* **Variables**:

    * `theta1`, `theta2`: angles of pendulum arms
    * `theta1_dot`, `theta2_dot`: angular velocities
* **Parameters**:

    * `l1`, `l2`: lengths of the pendulum arms
    * `m1`, `m2`: masses
    * `g`: gravitational acceleration

## Dynamics

* The system uses Newtonian equations of motion derived from Lagrangian mechanics.
* At each time step:

    * Angular accelerations are computed based on current angles and velocities
    * Angular velocities are updated using Euler integration
    * Positions are recalculated for visualization

## Parameters and Features

* Trail fading using alpha blending
* Trail length based on number of frames
* Independent pendulums with different configurations are simulated concurrently

## Results

The animation produces:

* Synchronized and then diverging pendulum paths
* Clear chaotic behavior — small differences in starting angles lead to vastly different outcomes

### Example Output

![Double Pendulum Animation](../results/double_pendulum.gif)

## Insights

* Demonstrates sensitivity to initial conditions — key property of chaotic systems
* Useful for visual intuition on deterministic chaos
* Can be extended to include damping or external forces
