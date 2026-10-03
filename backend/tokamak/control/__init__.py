"""Closed-loop co-simulation of a tokamak and a neural plasma controller.

Four blocks, run in lock-step at 1 kHz:

1. :mod:`.simulator` -- physical simulator: Grad-Shafranov equilibrium,
   coil / vessel / plasma circuit equations, massless force balance.
2. :mod:`.sensors`   -- FIFO delay (1-3 ms), Gaussian noise, integrator
   drift, normalisation.
3. :mod:`.policy`    -- actor-critic networks and PPO (numpy).
4. :mod:`.safety`    -- deterministic QP projection of the coil voltages.

:mod:`.env` wires blocks 1, 2 and 4 behind ``reset`` / ``step`` with the
reward and the early-termination rules; :mod:`.loop` runs any controller
through the cycle and defines the I/O contract; :mod:`.baseline` is a
classical model-based controller for comparison and as a teacher;
:mod:`.train` is the training entry point.
"""
