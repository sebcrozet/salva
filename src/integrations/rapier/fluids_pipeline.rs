use crate::coupling::CouplingManager;
use crate::geometry::{HGrid, HGridEntry};
use crate::object::{BoundaryHandle, BoundarySet, Fluid};
use crate::solver::DFSPHSolver;
use crate::LiquidWorld;
use crate::TimestepManager;
use approx::AbsDiffEq;
use na::Unit;
use rapier::dynamics::RigidBodySet;
use rapier::geometry::{ColliderHandle, ColliderSet};
use rapier::math::{Point, Vector};
use rapier::parry::bounding_volume::BoundingVolume;
use rapier::parry::shape::FeatureId;
use std::collections::HashMap;
use std::sync::RwLock;

/// Pipeline for particle-based fluid simulation.
pub struct FluidsPipeline {
    // FIXME: keep these public?
    /// The liquid world responsible for the actual fluid simulation.
    pub liquid_world: LiquidWorld,
    /// Set of couplings between fluids and rigid-bodies.
    pub coupling: ColliderCouplingSet,
}

impl FluidsPipeline {
    /// Initialize a new pipeline for fluids simulation.
    ///
    /// # Parameters
    ///
    /// - `particle_radius`: the radius of every particle for the fluid simulation.
    /// - `smoothing_factor`: the smoothing factor used to compute the SPH kernel radius.
    ///    The kernel radius will be computed as `particle_radius * smoothing_factor * 2.0.
    pub fn new(particle_radius: f32, smoothing_factor: f32) -> Self {
        let dfsph: DFSPHSolver = DFSPHSolver::new();

        Self {
            liquid_world: LiquidWorld::new(dfsph, particle_radius, smoothing_factor),
            coupling: ColliderCouplingSet::new(),
        }
    }

    /// Advances the fluid simulation by `dt` seconds.
    ///
    /// All the fluid particles will be affected by an acceleration equal to `gravity`.
    /// This `step` function may apply forces to some rigid-bodies that interact with fluids.
    /// However, it will not integrate these forces. Use the `PhysicsPipeline` for this integration.
    pub fn step(
        &mut self,
        gravity: &Vector<f32>,
        dt: f32,
        colliders: &ColliderSet,
        bodies: &mut RigidBodySet,
    ) {
        self.liquid_world.step_with_coupling(
            dt,
            gravity,
            &mut self.coupling.as_manager_mut(colliders, bodies),
        )
    }
}

/// The way a collider is coupled to a boundary object.
pub enum ColliderSampling {
    /// The collider shape is approximated with the given sample points in local-space.
    ///
    /// It is recommended that those points are separated by a distance smaller or equal to twice
    /// the particle radius used to initialize the LiquidWorld.
    StaticSampling(Vec<Point<f32>>),
    /// The collider shape is approximated by a dynamic set of points automatically computed based on contacts with fluid particles.
    DynamicContactSampling,
}

pub struct ColliderCouplingEntry {
    pub sampling_method: ColliderSampling,
    pub boundary: BoundaryHandle,
    features: Vec<FeatureId>,
}

/// Structure managing all the coupling between colliders from rapier with boundaries and fluids from salva.
pub struct ColliderCouplingSet {
    /// The hashmap containing mappings from ColliderHandle to ColliderCouplingEntry
    pub entries: HashMap<ColliderHandle, ColliderCouplingEntry>,
}

impl ColliderCouplingSet {
    /// Create a new collider coupling manager.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register a coupling between a boundary and a collider.
    /// There can be only up to one coupling between a collider and a boundary object. If a coupling
    /// already exists for this collider when calling this function, the handle of the previously coupled
    /// boundary is returned.
    pub fn register_coupling(
        &mut self,
        boundary: BoundaryHandle,
        collider: ColliderHandle,
        sampling_method: ColliderSampling,
    ) -> Option<BoundaryHandle> {
        let old = self.entries.insert(
            collider,
            ColliderCouplingEntry {
                sampling_method,
                boundary,
                features: Vec::new(),
            },
        );

        old.map(|e| e.boundary)
    }

    /// Unregister a coupling between a boundary and a collider.
    /// Note that this does not remove the boundary itself from the liquid world.
    /// Returns the handle of the boundary this collider was coupled with.
    pub fn unregister_coupling(&mut self, collider: ColliderHandle) -> Option<BoundaryHandle> {
        let deleted = self.entries.remove(&collider);
        deleted.map(|e| e.boundary)
    }

    /// Use this collider coupling set as a coupling manager.
    pub fn as_manager_mut<'a>(
        &'a mut self,
        colliders: &'a ColliderSet,
        bodies: &'a mut RigidBodySet,
    ) -> ColliderCouplingManager<'a> {
        ColliderCouplingManager {
            coupling: self,
            colliders,
            bodies,
        }
    }
}

/// A manager for coupling colliders from rapier2d/rapier3D with the boundary
/// objects from salva.
pub struct ColliderCouplingManager<'a> {
    coupling: &'a mut ColliderCouplingSet,
    colliders: &'a ColliderSet,
    bodies: &'a mut RigidBodySet,
}

impl<'a> CouplingManager for ColliderCouplingManager<'a> {
    fn update_boundaries(
        &mut self,
        timestep: &TimestepManager,
        h: f32,
        particle_radius: f32,
        hgrid: &HGrid<HGridEntry>,
        fluids: &mut [Fluid],
        boundaries: &mut BoundarySet,
    ) {
        let bodies = &self.bodies;
        for (collider, coupling) in &mut self.coupling.entries {
            if let (Some(collider), Some(boundary)) = (
                self.colliders.get(*collider),
                boundaries.get_mut(coupling.boundary),
            ) {
                // Update the boundary's ability to receive forces.
                let body = collider.parent().and_then(|p| bodies.get(p));
                if let Some(body) = body {
                    if !body.is_dynamic() {
                        boundary.forces = None;
                    } else {
                        boundary.forces = Some(RwLock::new(Vec::new()));
                        boundary.clear_forces(true);
                    }
                }

                // Update positions and velocities.
                boundary.positions.clear();
                boundary.velocities.clear();
                boundary.volumes.clear();
                coupling.features.clear();

                match &coupling.sampling_method {
                    ColliderSampling::StaticSampling(points) => {
                        for pt in points {
                            boundary.positions.push(collider.position() * pt);
                            let velocity = body.map(|b| b.velocity_at_point(pt));

                            boundary
                                .velocities
                                .push(velocity.unwrap_or(Vector::zeros()));
                        }

                        boundary.volumes.resize(points.len(), na::zero::<f32>());
                    }
                    ColliderSampling::DynamicContactSampling => {
                        let prediction = h * na::convert::<_, f32>(0.5);
                        let margin = particle_radius * na::convert::<_, f32>(0.1);
                        let collider_pos = collider.position();
                        let aabb = collider
                            .shape()
                            .compute_aabb(&collider_pos)
                            .loosened(h + prediction);

                        for particle in hgrid
                            .cells_intersecting_aabb(&aabb.mins, &aabb.maxs)
                            .flat_map(|e| e.1)
                        {
                            match particle {
                                HGridEntry::FluidParticle(fluid_id, particle_id) => {
                                    let fluid = &mut fluids[*fluid_id];
                                    let particle_pos = fluid.positions[*particle_id]
                                        + fluid.velocities[*particle_id] * timestep.dt();

                                    if aabb.contains_local_point(&particle_pos) {
                                        let (proj, feature) =
                                            collider.shape().project_point_and_get_feature(
                                                &collider_pos,
                                                &particle_pos,
                                            );

                                        let dpt = particle_pos - proj.point;

                                        if let Some((normal, depth)) =
                                            Unit::try_new_and_get(dpt, f32::default_epsilon())
                                        {
                                            if proj.is_inside {
                                                fluid.positions[*particle_id] -=
                                                    *normal * (depth + margin);

                                                let vel_err =
                                                    normal.dot(&fluid.velocities[*particle_id]);

                                                if vel_err > na::zero::<f32>() {
                                                    fluid.velocities[*particle_id] -=
                                                        *normal * vel_err;
                                                }
                                            } else if depth > h + prediction {
                                                continue;
                                            }
                                        }

                                        let velocity =
                                            body.map(|b| b.velocity_at_point(&proj.point));

                                        boundary
                                            .velocities
                                            .push(velocity.unwrap_or(Vector::zeros()));
                                        boundary.positions.push(proj.point);
                                        boundary.volumes.push(na::zero::<f32>());
                                        coupling.features.push(feature);
                                    }
                                }
                                HGridEntry::BoundaryParticle(..) => {
                                    // Not yet implemented.
                                }
                            }
                        }
                    }
                }

                boundary.clear_forces(true);
            }
        }
    }

    fn transmit_forces(&mut self, timestep: &TimestepManager, boundaries: &BoundarySet) {
        for (collider, coupling) in &self.coupling.entries {
            if let (Some(collider), Some(boundary)) = (
                self.colliders.get(*collider),
                boundaries.get(coupling.boundary),
            ) {
                if boundary.positions.is_empty() {
                    continue;
                }

                if let Some(forces) = &boundary.forces {
                    let forces = forces.read().unwrap();
                    if let Some(parent) = collider.parent() {
                        if let Some(body) = self.bodies.get_mut(parent) {
                            for (pos, force) in
                                boundary.positions.iter().zip(forces.iter().cloned())
                            {
                                body.apply_impulse_at_point(force * timestep.dt(), *pos, true)
                            }
                        }
                    }
                }
            }
        }
    }
}
