use na::RealField;

use crate::object::Fluid;

/// Structure responsible for regulating the timestep length of the simulation.
pub struct TimestepManager<N: RealField> {
    cfl_coeff: N,
    min_num_substeps: u32,
    max_num_substeps: u32,
    dt: N,
    inv_dt: N,
    total_step_size: N,
    remaining_time: N,
    particle_radius: N,
}

impl<N: RealField> TimestepManager<N> {
    /// Initialize a new timestep manager with default parameters.
    pub fn new(particle_radius: N) -> Self {
        Self {
            cfl_coeff: na::convert(0.4),
            min_num_substeps: 1,
            max_num_substeps: 10,
            particle_radius,
            dt: N::zero(),
            inv_dt: N::zero(),
            total_step_size: N::zero(),
            remaining_time: N::zero(),
        }
    }

    fn max_substep(&self, fluids: &[Fluid<N>]) -> N {
        let mut max_sq_vel = N::zero();
        for (v, a) in fluids
            .iter()
            .flat_map(|f| f.velocities.iter().zip(f.accelerations.iter()))
        {
            max_sq_vel = max_sq_vel.max((v + a * self.remaining_time).norm_squared());
        }

        self.particle_radius * na::convert(2.0) / max_sq_vel.sqrt() * self.cfl_coeff
    }

    pub fn reset(&mut self, total_step_size: N) {
        self.total_step_size = total_step_size;
        self.remaining_time = total_step_size;
    }

    #[inline]
    pub fn is_done(&self) -> bool {
        self.remaining_time <= N::default_epsilon()
    }

    #[inline]
    pub fn dt(&self) -> N {
        self.dt
    }

    #[inline]
    pub fn inv_dt(&self) -> N {
        self.inv_dt
    }

    #[inline]
    pub fn advance(&mut self, fluids: &[Fluid<N>]) {
        self.remaining_time -= self.dt;
        let substep = self.compute_substep(fluids);
        self.dt = substep;
        self.inv_dt = if substep.is_zero() {
            N::zero()
        } else {
            N::one() / substep
        };
    }

    fn compute_substep(&self, fluids: &[Fluid<N>]) -> N {
        return self.total_step_size;
        let min_substep = self.total_step_size / na::convert(self.max_num_substeps as f64);
        let max_substep = self.total_step_size / na::convert(self.min_num_substeps as f64);
        let computed_substep = self.max_substep(fluids);
        na::clamp(computed_substep, min_substep, max_substep)
    }
}
