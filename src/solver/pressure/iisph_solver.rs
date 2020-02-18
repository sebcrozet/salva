use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField};

use crate::geometry::{ContactManager, ParticlesContacts};
use crate::kernel::{CubicSplineKernel, Kernel};
use crate::math::Vector;
use crate::object::{Boundary, Fluid};
use crate::solver::PressureSolver;

/// A Position Based Fluid solver.
pub struct IISPHSolver<
    N: RealField,
    KernelDensity: Kernel = CubicSplineKernel,
    KernelGradient: Kernel = CubicSplineKernel,
> {
    min_pressure_iter: usize,
    max_pressure_iter: usize,
    max_density_error: N,
    omega: N,
    densities: Vec<Vec<N>>,
    aii: Vec<Vec<N>>,
    dii: Vec<Vec<Vector<N>>>,
    dij_pjl: Vec<Vec<Vector<N>>>,
    pressures: Vec<Vec<N>>,
    next_pressures: Vec<Vec<N>>,
    predicted_densities: Vec<Vec<N>>,
    boundaries_volumes: Vec<Vec<N>>,
    velocity_changes: Vec<Vec<Vector<N>>>,
    phantoms: PhantomData<(KernelDensity, KernelGradient)>,
}

impl<N, KernelDensity, KernelGradient> IISPHSolver<N, KernelDensity, KernelGradient>
where
    N: RealField,
    KernelDensity: Kernel,
    KernelGradient: Kernel,
{
    /// Initialize a new Position Based Fluid solver.
    pub fn new() -> Self {
        Self {
            min_pressure_iter: 1,
            max_pressure_iter: 50,
            max_density_error: na::convert(0.05),
            omega: na::convert(0.5),
            densities: Vec::new(),
            dii: Vec::new(),
            aii: Vec::new(),
            dij_pjl: Vec::new(),
            pressures: Vec::new(),
            next_pressures: Vec::new(),
            predicted_densities: Vec::new(),
            boundaries_volumes: Vec::new(),
            velocity_changes: Vec::new(),
            phantoms: PhantomData,
        }
    }

    fn update_fluid_contacts(
        &mut self,
        _dt: N,
        kernel_radius: N,
        fluid_fluid_contacts: &mut [ParticlesContacts<N>],
        fluid_boundary_contacts: &mut [ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        let _velocity_changes = &self.velocity_changes;
        for contacts in fluid_fluid_contacts.iter_mut() {
            par_iter_mut!(contacts.contacts_mut()).for_each(|c| {
                let fluid1 = &fluids[c.i_model];
                let fluid2 = &fluids[c.j_model];
                let pi = fluid1.positions[c.i];
                let pj = fluid2.positions[c.j];

                c.weight = KernelDensity::points_apply(&pi, &pj, kernel_radius);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, kernel_radius);
            })
        }

        for contacts in fluid_boundary_contacts.iter_mut() {
            par_iter_mut!(contacts.contacts_mut()).for_each(|c| {
                let fluid1 = &fluids[c.i_model];
                let bound2 = &boundaries[c.j_model];

                let pi = fluid1.positions[c.i];
                let pj = bound2.positions[c.j];

                c.weight = KernelDensity::points_apply(&pi, &pj, kernel_radius);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, kernel_radius);
            })
        }
    }

    fn update_boundary_contacts(
        &mut self,
        kernel_radius: N,
        boundary_boundary_contacts: &mut [ParticlesContacts<N>],
        boundaries: &[Boundary<N>],
    ) {
        for contacts in boundary_boundary_contacts.iter_mut() {
            par_iter_mut!(contacts.contacts_mut()).for_each(|c| {
                let bound1 = &boundaries[c.i_model];
                let bound2 = &boundaries[c.j_model];

                let pi = bound1.positions[c.i];
                let pj = bound2.positions[c.j];

                c.weight = KernelDensity::points_apply(&pi, &pj, kernel_radius);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, kernel_radius);
            })
        }
    }

    fn compute_boundary_volumes(
        &mut self,
        boundary_boundary_contacts: &[ParticlesContacts<N>],
        boundaries: &[Boundary<N>],
    ) {
        for boundary_id in 0..boundaries.len() {
            par_iter_mut!(self.boundaries_volumes[boundary_id])
                .enumerate()
                .for_each(|(i, volume)| {
                    let mut denominator = N::zero();

                    for c in boundary_boundary_contacts[boundary_id].particle_contacts(i) {
                        denominator += c.weight;
                    }

                    assert!(!denominator.is_zero());
                    *volume = N::one() / denominator;
                })
        }
    }

    fn compute_densities(
        &mut self,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) {
        let boundaries_volumes = &self.boundaries_volumes;

        for fluid_id in 0..fluids.len() {
            par_iter_mut!(self.densities[fluid_id])
                .enumerate()
                .for_each(|(i, density)| {
                    *density = N::zero();

                    for c in fluid_fluid_contacts[fluid_id].particle_contacts(i) {
                        *density += fluids[c.j_model].particle_mass(c.j) * c.weight;
                    }

                    for c in fluid_boundary_contacts[fluid_id].particle_contacts(i) {
                        *density += boundaries_volumes[c.j_model][c.j]
                            * fluids[c.i_model].density0
                            * c.weight;
                    }

                    assert!(!density.is_zero());
                })
        }
    }

    fn compute_predicted_densities(
        &mut self,
        dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) {
        let boundaries_volumes = &self.boundaries_volumes;
        let velocity_changes = &self.velocity_changes;
        let densities = &self.densities;
        let _max_error = N::zero();

        for fluid_id in 0..fluids.len() {
            let _it = par_iter_mut!(self.predicted_densities[fluid_id])
                .enumerate()
                .for_each(|(i, predicted_density)| {
                    let fluid_i = &fluids[fluid_id];
                    let mut delta = N::zero();

                    for c in fluid_fluid_contacts[fluid_id].particle_contacts(i) {
                        let fluid_j = &fluids[c.j_model];
                        let vi = fluid_i.velocities[c.i] + velocity_changes[c.i_model][c.i];
                        let vj = fluid_j.velocities[c.j] + velocity_changes[c.j_model][c.j];

                        delta += fluids[c.j_model].particle_mass(c.j) * (vi - vj).dot(&c.gradient);
                    }

                    for c in fluid_boundary_contacts[fluid_id].particle_contacts(i) {
                        let vi = fluid_i.velocities[c.i] + velocity_changes[c.i_model][c.i];
                        // FIXME: take the velocity of j too?

                        delta += boundaries_volumes[c.j_model][c.j]
                            * fluid_i.density0
                            * vi.dot(&c.gradient);
                    }

                    *predicted_density = densities[fluid_id][i] + delta * dt;
                    assert!(!predicted_density.is_zero());
                });
        }
    }

    fn compute_dii(
        &mut self,
        dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) {
        let boundaries_volumes = &self.boundaries_volumes;

        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let dii = &mut self.dii[fluid_id];
            let fluid_i = &fluids[fluid_id];
            let densities = &self.densities;

            par_iter_mut!(dii).enumerate().for_each(|(i, dii)| {
                dii.fill(N::zero());

                let rhoi = densities[fluid_id][i];
                let factor = -dt * dt / (rhoi * rhoi);

                for c in fluid_fluid_contacts.particle_contacts(i) {
                    let mj = fluids[c.j_model].particle_mass(c.j);
                    *dii += c.gradient * (mj * factor);
                }

                for c in fluid_boundary_contacts.particle_contacts(i) {
                    let mj = boundaries_volumes[c.j_model][c.j] * fluid_i.density0;
                    *dii += c.gradient * (mj * factor);
                }
            })
        }
    }

    fn compute_aii(
        &mut self,
        dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) {
        let boundaries_volumes = &self.boundaries_volumes;

        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let aii = &mut self.aii[fluid_id];
            let dii = &self.dii[fluid_id];
            let fluid_i = &fluids[fluid_id];
            let densities = &self.densities;

            par_iter_mut!(aii).enumerate().for_each(|(i, aii)| {
                *aii = N::zero();
                let rhoi = densities[fluid_id][i];
                let mi = fluids[fluid_id].particle_mass(i);
                let factor = dt * dt * mi / (rhoi * rhoi);

                for c in fluid_fluid_contacts.particle_contacts(i) {
                    let mj = fluids[c.j_model].particle_mass(c.j);
                    let dji = c.gradient * factor;
                    *aii += mj * (dii[c.i] - dji).dot(&c.gradient);
                }

                for c in fluid_boundary_contacts.particle_contacts(i) {
                    let mj = boundaries_volumes[c.j_model][c.j] * fluid_i.density0;
                    let dji = c.gradient * factor;
                    *aii += mj * (dii[c.i] - dji).dot(&c.gradient);
                }
            })
        }
    }

    fn compute_dij_pjl(
        &mut self,
        dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) {
        let _boundaries_volumes = &self.boundaries_volumes;

        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let _fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let dij_pjl = &mut self.dij_pjl[fluid_id];
            let _fluid_i = &fluids[fluid_id];
            let densities = &self.densities;
            let pressures = &self.pressures;

            par_iter_mut!(dij_pjl).enumerate().for_each(|(i, dij_pjl)| {
                dij_pjl.fill(N::zero());

                for c in fluid_fluid_contacts.particle_contacts(i) {
                    let rhoj = densities[c.j_model][c.j];
                    let mj = fluids[c.j_model].particle_mass(c.j);
                    let p_jl = pressures[c.j_model][c.j];
                    *dij_pjl += c.gradient * (-mj * p_jl / (rhoj * rhoj));
                }

                *dij_pjl *= dt * dt;
            })
        }
    }

    fn compute_next_pressures(
        &mut self,
        dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) -> N {
        let boundaries_volumes = &self.boundaries_volumes;
        let mut max_error = N::zero();

        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let next_pressures = &mut self.next_pressures[fluid_id];
            let pressures = &self.pressures;
            let fluid_i = &fluids[fluid_id];
            let densities = &self.densities;
            let predicted_densities = &self.predicted_densities;
            let omega = self.omega;
            let aii = &self.aii[fluid_id];
            let dij_pjl = &self.dij_pjl;
            let dii = &self.dii;

            let it = par_iter_mut!(next_pressures)
                .enumerate()
                .map(|(i, next_pressure)| {
                    if aii[i].abs() > na::convert(1.0e-9) {
                        let mut sum = N::zero();
                        let pi = pressures[fluid_id][i];
                        let mi = fluid_i.particle_mass(i);
                        let rhoi = densities[fluid_id][i];
                        let derr = fluid_i.density0 - predicted_densities[fluid_id][i];

                        for c in fluid_fluid_contacts.particle_contacts(i) {
                            let mj = fluids[c.j_model].particle_mass(c.j);
                            let dji = c.gradient * (dt * dt * mi / (rhoi * rhoi));
                            let factor = dij_pjl[c.i_model][c.i]
                                - dii[c.j_model][c.j] * pressures[c.j_model][c.j]
                                - (dij_pjl[c.j_model][c.j] - dji * pi);
                            sum += mj * factor.dot(&c.gradient);
                        }

                        for c in fluid_boundary_contacts.particle_contacts(i) {
                            let mj = boundaries_volumes[c.j_model][c.j] * fluid_i.density0;
                            sum += mj * dij_pjl[c.i_model][c.i].dot(&c.gradient);
                        }

                        *next_pressure = (N::one() - omega) * pi + omega * (derr - sum) / aii[i];

                        if *next_pressure > N::zero() {
                            *next_pressure = next_pressure.max(N::zero());
                            (-sum - aii[i] * *next_pressure) / fluid_i.density0
                        } else {
                            // Clamp negative pressures.
                            *next_pressure = N::zero();
                            N::zero()
                        }
                    } else {
                        *next_pressure = N::zero();
                        N::zero()
                    }
                });
            let err = par_reduce_sum!(N::zero(), it);

            let nparts = fluids[fluid_id].num_particles();
            if nparts != 0 {
                max_error = max_error.max(err / na::convert(nparts as f64));
            }
        }

        max_error
    }

    fn compute_velocity_changes(
        &mut self,
        dt: N,
        _inv_dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        _boundaries: &[Boundary<N>],
    ) {
        let boundaries_volumes = &self.boundaries_volumes;
        let densities = &self.densities;
        let pressures = &self.pressures;

        for (fluid_id, _fluid1) in fluids.iter().enumerate() {
            par_iter_mut!(self.velocity_changes[fluid_id])
                .enumerate()
                .for_each(|(i, velocity_change)| {
                    let fluid_i = &fluids[fluid_id];
                    let pi = pressures[fluid_id][i];
                    let rhoi = densities[fluid_id][i];

                    for c in fluid_fluid_contacts[fluid_id].particle_contacts(i) {
                        let mj = fluids[c.j_model].particle_mass(c.j);
                        let pj = pressures[c.j_model][c.j];
                        let rhoj = densities[c.j_model][c.j];

                        *velocity_change -=
                            c.gradient * (dt * mj * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj)));
                    }

                    for c in fluid_boundary_contacts[fluid_id].particle_contacts(i) {
                        let mj = boundaries_volumes[c.j_model][c.j] * fluid_i.density0;
                        *velocity_change -= c.gradient * (dt * mj * pi / (rhoi * rhoi));
                    }
                })
        }
    }

    fn update_velocities_and_positions(&mut self, dt: N, fluids: &mut [Fluid<N>]) {
        for (fluid, delta) in fluids.iter_mut().zip(self.velocity_changes.iter()) {
            par_iter_mut!(fluid.positions)
                .zip(par_iter_mut!(fluid.velocities))
                .zip(par_iter!(delta))
                .for_each(|((pos, vel), delta)| {
                    *vel += delta;
                    *pos += *vel * dt;
                })
        }
    }

    fn pressure_solve(
        &mut self,
        dt: N,
        _inv_dt: N,
        _kernel_radius: N,
        contact_manager: &mut ContactManager<N>,
        fluids: &mut [Fluid<N>],
        _boundaries: &[Boundary<N>],
    ) {
        for i in 0..self.max_pressure_iter {
            self.compute_dij_pjl(
                dt,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
            );

            let avg_err = self.compute_next_pressures(
                dt,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
            );

            std::mem::swap(&mut self.pressures, &mut self.next_pressures);

            if avg_err <= self.max_density_error && i >= self.min_pressure_iter {
                println!(
                    "Average density error: {}, break after niters: {}",
                    avg_err, i
                );
                break;
            }
        }
    }
}

impl<N, KernelDensity, KernelGradient> PressureSolver<N>
    for IISPHSolver<N, KernelDensity, KernelGradient>
where
    N: RealField,
    KernelDensity: Kernel,
    KernelGradient: Kernel,
{
    fn velocity_changes(&self) -> &[Vec<Vector<N>>] {
        &self.velocity_changes
    }

    fn velocity_changes_mut(&mut self) -> &mut [Vec<Vector<N>>] {
        &mut self.velocity_changes
    }

    fn init_with_fluids(&mut self, fluids: &[Fluid<N>]) {
        // Resize every buffer.
        self.densities.resize(fluids.len(), Vec::new());
        self.predicted_densities.resize(fluids.len(), Vec::new());
        self.velocity_changes.resize(fluids.len(), Vec::new());
        self.aii.resize(fluids.len(), Vec::new());
        self.dii.resize(fluids.len(), Vec::new());
        self.dij_pjl.resize(fluids.len(), Vec::new());
        self.pressures.resize(fluids.len(), Vec::new());
        self.next_pressures.resize(fluids.len(), Vec::new());

        for i in 0..fluids.len() {
            let nparticles = fluids[i].num_particles();

            self.densities[i].resize(nparticles, N::zero());
            self.predicted_densities[i].resize(nparticles, N::zero());
            self.velocity_changes[i].resize(nparticles, Vector::zeros());
            self.aii[i].resize(nparticles, N::zero());
            self.dii[i].resize(nparticles, Vector::zeros());
            self.dij_pjl[i].resize(nparticles, Vector::zeros());
            self.pressures[i].resize(nparticles, N::zero());
            self.next_pressures[i].resize(nparticles, N::zero());
        }
    }

    fn init_with_boundaries(&mut self, boundaries: &[Boundary<N>]) {
        self.boundaries_volumes.resize(boundaries.len(), Vec::new());

        for (boundary, boundary_volumes) in
            boundaries.iter().zip(self.boundaries_volumes.iter_mut())
        {
            boundary_volumes.resize(boundary.num_particles(), N::zero())
        }
    }

    fn predict_advection(&mut self, dt: N, gravity: &Vector<N>, fluids: &[Fluid<N>]) {
        for (_fluid, velocity_changes) in fluids.iter().zip(self.velocity_changes.iter_mut()) {
            par_iter_mut!(velocity_changes).for_each(|velocity_change| {
                *velocity_change += gravity * dt;
            })
        }
    }

    fn step(
        &mut self,
        dt: N,
        contact_manager: &mut ContactManager<N>,
        kernel_radius: N,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        let inv_dt = N::one() / dt;

        // Init boundary-related data.
        self.update_boundary_contacts(
            kernel_radius,
            &mut contact_manager.boundary_boundary_contacts,
            boundaries,
        );

        self.compute_boundary_volumes(&contact_manager.boundary_boundary_contacts, boundaries);

        self.update_fluid_contacts(
            dt,
            kernel_radius,
            &mut contact_manager.fluid_fluid_contacts,
            &mut contact_manager.fluid_boundary_contacts,
            fluids,
            boundaries,
        );

        self.compute_densities(
            &contact_manager.fluid_fluid_contacts,
            &contact_manager.fluid_boundary_contacts,
            fluids,
        );

        for (fluid, fluid_fluid_contacts, densities, velocity_changes) in itertools::multizip((
            &mut *fluids,
            &contact_manager.fluid_fluid_contacts,
            &self.densities,
            &mut self.velocity_changes,
        )) {
            let mut forces = std::mem::replace(&mut fluid.nonpressure_forces, Vec::new());

            for np_force in &mut forces {
                np_force.solve(
                    dt,
                    kernel_radius,
                    fluid_fluid_contacts,
                    fluid,
                    densities,
                    velocity_changes,
                );
            }

            fluid.nonpressure_forces = forces;
        }

        self.compute_dii(
            dt,
            &contact_manager.fluid_fluid_contacts,
            &contact_manager.fluid_boundary_contacts,
            fluids,
        );

        let _0_5: N = na::convert(0.5);
        self.pressures
            .iter_mut()
            .flat_map(|v| v.iter_mut())
            .for_each(|p| *p *= _0_5);

        let _ = self.compute_predicted_densities(
            dt,
            &contact_manager.fluid_fluid_contacts,
            &contact_manager.fluid_boundary_contacts,
            fluids,
        );

        self.compute_aii(
            dt,
            &contact_manager.fluid_fluid_contacts,
            &contact_manager.fluid_boundary_contacts,
            fluids,
        );

        self.pressure_solve(
            dt,
            inv_dt,
            kernel_radius,
            contact_manager,
            fluids,
            boundaries,
        );

        self.compute_velocity_changes(
            dt,
            inv_dt,
            &contact_manager.fluid_fluid_contacts,
            &contact_manager.fluid_boundary_contacts,
            fluids,
            boundaries,
        );

        self.update_velocities_and_positions(dt, fluids);

        self.velocity_changes
            .iter_mut()
            .for_each(|vs| vs.iter_mut().for_each(|v| v.fill(N::zero())));
    }
}