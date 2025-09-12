use std::{sync::Arc, thread};

use azalea::{
    blocks::{BlockState, BlockTrait},
    core::{
        direction::Direction,
        position::{ChunkPos, ChunkSectionBlockPos, ChunkSectionPos},
    },
    physics::collision::BlockWithShape,
    world::Chunk,
};
use crossbeam::channel::{unbounded, Receiver, Sender};
use glam::{IVec3, Vec3};
use parking_lot::RwLock;

use crate::{
    assets::{processed::model::Cube, MeshAssets},
    renderer::mesh::Vertex,
};

pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub section_pos: ChunkSectionPos,
}

pub struct LocalSection {
    pub blocks: Box<[[[BlockState; 18]; 18]; 18]>,
    pub spos: ChunkSectionPos,
}

const NORTH: usize = 0;
const SOUTH: usize = 1;
const EAST: usize = 2;
const WEST: usize = 3;
const NE: usize = 4;
const NW: usize = 5;
const SE: usize = 6;
const SW: usize = 7;

pub struct LocalChunk {
    pub center: Arc<RwLock<Chunk>>,
    pub neighbors: [Option<Arc<RwLock<Chunk>>>; 8],
}

impl LocalChunk {
    pub fn build_local_section(&self, spos: ChunkSectionPos) -> LocalSection {
        let mut blocks = Box::new([[[BlockState::AIR; 18]; 18]; 18]);

        for lx in -1..17 {
            for ly in -1..17 {
                for lz in -1..17 {
                    let ix = (lx + 1) as usize;
                    let iy = (ly + 1) as usize;
                    let iz = (lz + 1) as usize;

                    blocks[ix][iy][iz] = self.get_block_local(spos.y, lx, ly, lz);
                }
            }
        }

        LocalSection { blocks, spos }
    }

    fn get_block_local(&self, base_y: i32, lx: i32, ly: i32, lz: i32) -> BlockState {
        let cx_off = lx.div_euclid(16);
        let sx = lx.rem_euclid(16) as u8;

        let cy_off = ly.div_euclid(16);
        let sy = ly.rem_euclid(16) as u8;

        let cz_off = lz.div_euclid(16);
        let sz = lz.rem_euclid(16) as u8;

        let chunk_opt = match (cx_off, cz_off) {
            (0, 0) => Some(&self.center),
            (0, -1) => self.neighbors[NORTH].as_ref(),
            (0, 1) => self.neighbors[SOUTH].as_ref(),
            (-1, 0) => self.neighbors[WEST].as_ref(),
            (1, 0) => self.neighbors[EAST].as_ref(),
            (-1, -1) => self.neighbors[NW].as_ref(),
            (1, -1) => self.neighbors[NE].as_ref(),
            (-1, 1) => self.neighbors[SW].as_ref(),
            (1, 1) => self.neighbors[SE].as_ref(),
            _ => None,
        };

        if let Some(chunk_arc) = chunk_opt {
            let chunk = chunk_arc.read();
            let section_index = (base_y + cy_off) as usize;
            if let Some(section) = chunk.sections.get(section_index) {
                return section.get_block_state(ChunkSectionBlockPos { x: sx, y: sy, z: sz });
            }
        }

        BlockState::AIR
    }
}

pub struct Mesher {
    work_tx: Sender<LocalSection>,
    result_rx: Receiver<MeshData>,
}

impl Mesher {
    pub fn new(assets: Arc<MeshAssets>) -> Self {
        let (work_tx, work_rx) = unbounded::<LocalSection>();
        let (result_tx, result_rx) = unbounded::<MeshData>();

        thread::spawn(move || {
            while let Ok(local_section) = work_rx.recv() {
                let mesh = mesh_section(&local_section, &assets);
                result_tx.send(mesh).unwrap();
            }
        });

        Self { work_tx, result_rx }
    }

    pub fn submit_chunk(&self, chunk_pos: ChunkPos, local_chunk: &LocalChunk) {
        let chunk = local_chunk.center.read();

        for (i, section) in chunk.sections.iter().enumerate() {
            if section.block_count == 0 {
                continue;
            }
            let spos = ChunkSectionPos::new(chunk_pos.x, i as i32, chunk_pos.z);
            let local_section = local_chunk.build_local_section(spos);
            self.submit(local_section);
        }
    }

    pub fn submit(&self, local_section: LocalSection) {
        self.work_tx.send(local_section).unwrap();
    }

    pub fn poll(&self) -> Option<MeshData> {
        self.result_rx.try_recv().ok()
    }
}

pub fn mesh_section(section: &LocalSection, assets: &MeshAssets) -> MeshData {
    let mut vertices = Vec::with_capacity(1000);
    let mut indices = Vec::with_capacity(1000);

    for y in 0..16 {
        for x in 0..16 {
            for z in 0..16 {
                let local = IVec3::new(x + 1, y + 1, z + 1);
                let block_state = section.blocks[local.x as usize][local.y as usize][local.z as usize];

                if block_state.is_air() {
                    continue;
                }

                for desc in assets.get_variant_descs(block_state) {
                    let model = assets
                        .get_block_model(&desc.model)
                        .expect("all block models must be loaded");

                    for element in &model.elements {
                        for face in FACES {
                            // rotate for culling/AO only (positions are rotated separately)
                            let rotated_dir = rotate_direction(face.dir, desc.x_rotation, desc.y_rotation);

                            // select JSON face by ORIGINAL side
                            let model_face = match face.dir {
                                Direction::Down => &element.faces.down,
                                Direction::Up => &element.faces.up,
                                Direction::North => &element.faces.north,
                                Direction::South => &element.faces.south,
                                Direction::West => &element.faces.west,
                                Direction::East => &element.faces.east,
                            };

                            if let Some(model_face) = model_face {
                                // cullface: rotate, then check neighbor
                                if let Some(cull_face) = model_face
                                    .cullface
                                    .as_deref()
                                    .and_then(|s| match s {
                                        "down" => Some(Direction::Down),
                                        "up" => Some(Direction::Up),
                                        "north" => Some(Direction::North),
                                        "south" => Some(Direction::South),
                                        "west" => Some(Direction::West),
                                        "east" => Some(Direction::East),
                                        _ => None,
                                    })
                                {
                                    let cull_dir = rotate_direction(cull_face, desc.x_rotation, desc.y_rotation);
                                    let n = cull_dir.normal();
                                    let neighbor = local + IVec3::new(n.x, n.y, n.z);
                                    let neighbor_state = section.blocks[neighbor.x as usize][neighbor.y as usize][neighbor.z as usize];
                                    let dyn_neighbor: Box<dyn BlockTrait> = Box::from(neighbor_state);

                                    let occlude = (|| {
                                        if dyn_neighbor.behavior().can_occlude {
                                            if neighbor_state.is_air() { return false; }
                                            if neighbor_state.is_collision_shape_empty() { return false; }
                                            true
                                        } else { false }
                                    })();
                                    if occlude { continue; }
                                }

                                // Resolve texture index ONCE per face
                                let tex_idx = model
                                    .resolve_texture(&model_face.texture)
                                    .and_then(|name: &str| assets.get_texture_id(name))
                                    .unwrap_or(0) as u32;

                                // Build per-vertex UVs from the face rect using face-local mapping
                                let mut uvs = [
                                    uv_for_vertex(face.dir, model_face.uv, element, face.offsets[0]),
                                    uv_for_vertex(face.dir, model_face.uv, element, face.offsets[1]),
                                    uv_for_vertex(face.dir, model_face.uv, element, face.offsets[2]),
                                    uv_for_vertex(face.dir, model_face.uv, element, face.offsets[3]),
                                ];

// 1) Apply per-face rotation from JSON (clockwise)
let face_rot = model_face.rotation.rem_euclid(360);
if face_rot != 0 {
    uvs = rotate_uv_about_rect(uvs, model_face.uv, face_rot as i32);
}

// 2) Apply uvlock transform only when uvlock is true
if desc.uvlock {
    uvs = apply_uvlock(uvs, face.dir, desc.y_rotation);
}

// 3) Shrink UVs toward rect center
let shrink = uv_shrink_ratio_for(tex_idx, assets);
uvs = shrink_uvs_toward_center(uvs, model_face.uv, shrink);
                                let len = vertices.len() as u32;

                                for (i, offset) in face.offsets.iter().enumerate() {
                                    let mut local_pos = offset_to_coord(*offset, element) / 16.0;
                                    local_pos = rotate_position(local_pos, desc.x_rotation, desc.y_rotation);

                                    let world_pos = Vec3::new(
                                        (local.x - 1) as f32 + section.spos.x as f32 * 16.0,
                                        (local.y - 1) as f32 + section.spos.y as f32 * 16.0,
                                        (local.z - 1) as f32 + section.spos.z as f32 * 16.0,
                                    );

                                    vertices.push(Vertex {
                                        position: (local_pos + world_pos).into(),
                                        ao: if model.ambient_occlusion {
                                            compute_ao(local, rotate_corner_offset(*offset, desc.x_rotation, desc.y_rotation), rotated_dir, section) as f32
                                        } else { 3.0 },
                                        tex_idx: tex_idx as u32,
                                        uv: uvs[i].into(),
                                    });
                                }

                                // 4) Indices with winding correction to match Mojang's recalculateWinding
                                let v0: glam::Vec3 = vertices[len as usize + 0].position.into();
                                let v1: glam::Vec3 = vertices[len as usize + 1].position.into();
                                let v2: glam::Vec3 = vertices[len as usize + 2].position.into();
                                let tri_n = (v1 - v0).cross(v2 - v0).normalize_or_zero();
                                let want = face_dir_vec(rotated_dir);
                                let (i0, i1, i2, i3, i4, i5) = if tri_n.dot(want) < 0.0 {
                                    (len, len + 2, len + 1, len, len + 3, len + 2)
                                } else {
                                    (len, len + 1, len + 2, len, len + 2, len + 3)
                                };
                                indices.extend_from_slice(&[i0, i1, i2, i3, i4, i5]);
                            }
                        }
                    }
                }
            }
        }
    }

    MeshData { section_pos: section.spos, vertices, indices }
}

fn rotate_position(mut p: glam::Vec3, x_rot: i32, y_rot: i32) -> glam::Vec3 {
    // rotate around block center (vanilla)
    p -= glam::Vec3::splat(0.5);

    // Y then X (matches FaceBakery)
    match y_rot.rem_euclid(360) {
        90 => p = glam::Vec3::new(-p.z, p.y, p.x),
        180 => p = glam::Vec3::new(-p.x, p.y, -p.z),
        270 => p = glam::Vec3::new(p.z, p.y, -p.x),
        _ => {}
    }
    match x_rot.rem_euclid(360) {
        90 => p = glam::Vec3::new(p.x, -p.z, p.y),
        180 => p = glam::Vec3::new(p.x, -p.y, -p.z),
        270 => p = glam::Vec3::new(p.x, p.z, -p.y),
        _ => {}
    }

    p += glam::Vec3::splat(0.5);
    p
}

fn rotate_direction(dir: Direction, x_rot: i32, y_rot: i32) -> Direction {
    let mut d = dir;

    // Y first
    d = match y_rot.rem_euclid(360) {
        90 => match d {
            Direction::North => Direction::East,
            Direction::East => Direction::South,
            Direction::South => Direction::West,
            Direction::West => Direction::North,
            other => other,
        },
        180 => match d {
            Direction::North => Direction::South,
            Direction::South => Direction::North,
            Direction::East => Direction::West,
            Direction::West => Direction::East,
            other => other,
        },
        270 => match d {
            Direction::North => Direction::West,
            Direction::West => Direction::South,
            Direction::South => Direction::East,
            Direction::East => Direction::North,
            other => other,
        },
        _ => d,
    };

    // then X
    d = match x_rot.rem_euclid(360) {
        90 => match d {
            Direction::Up => Direction::South,
            Direction::South => Direction::Down,
            Direction::Down => Direction::North,
            Direction::North => Direction::Up,
            other => other,
        },
        180 => match d {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::North => Direction::South,
            Direction::South => Direction::North,
            other => other,
        },
        270 => match d {
            Direction::Up => Direction::North,
            Direction::North => Direction::Down,
            Direction::Down => Direction::South,
            Direction::South => Direction::Up,
            other => other,
        },
        _ => d,
    };

    d
}

#[inline]
fn uv_mapping_is_mirrored(dir: Direction) -> bool {
    match dir {
        // South: U = (tx - x)/dx  (mirrored U)
        // West:  U = (tz - z)/dz  (mirrored U)
        // Down:  V = (tz - z)/dz  (mirrored V)
        Direction::South | Direction::West | Direction::Down => true,
        _ => false,
    }
}

#[inline]
fn rotate_uv_about_rect(
    mut uvs: [glam::Vec2; 4],
    rect: Option<[f32; 4]>,
    deg_cw: i32, // 0, 90, 180, 270 (clockwise)
) -> [glam::Vec2; 4] {
    let (u1, v1, u2, v2) = match rect {
        Some([u1, v1, u2, v2]) => (u1 / 16.0, v1 / 16.0, u2 / 16.0, v2 / 16.0),
        None => (0.0, 0.0, 1.0, 1.0),
    };
    let du = (u2 - u1).max(1e-8);
    let dv = (v2 - v1).max(1e-8);

    for uv in &mut uvs {
        // normalize to [0,1]^2 inside this rect
        let mut s = (uv.x - u1) / du;
        let mut t = (uv.y - v1) / dv;

        // vanilla uses V downward; rotations are clockwise
        match deg_cw.rem_euclid(360) {
            90 => {
                let ns = t;
                let nt = 1.0 - s;
                s = ns;
                t = nt;
            }
            180 => {
                s = 1.0 - s;
                t = 1.0 - t;
            }
            270 => {
                let ns = 1.0 - t;
                let nt = s;
                s = ns;
                t = nt;
            }
            _ => {}
        }

        // back to UV space of the rect
        *uv = glam::Vec2::new(u1 + s * du, v1 + t * dv);
    }
    uvs
}

#[inline]
fn apply_uvlock(
    mut uvs: [glam::Vec2; 4],
    face_dir: Direction,
    y_rot_deg: i32, // blockstate Y rotation (deg)
) -> [glam::Vec2; 4] {
    // Equivalent to ModelState.inverseFaceTransformation(face)
    let mut cw = (-y_rot_deg).rem_euclid(360);
    if uv_mapping_is_mirrored(face_dir) {
        // mirrored mapping flips perceived rotation
        cw = (-cw).rem_euclid(360);
    }
    if cw == 0 {
        return uvs;
    }

    let rad = (cw as f32).to_radians();
    let (s, c) = rad.sin_cos();
    for uv in &mut uvs {
        let mut p = *uv - glam::Vec2::splat(0.5);
        // clockwise 2D rotation
        p = glam::Vec2::new(c * p.x + s * p.y, -s * p.x + c * p.y);
        *uv = p + glam::Vec2::splat(0.5);
    }
    uvs
}

#[inline]
fn uv_for_vertex(
    dir: Direction,
    rect: Option<[f32; 4]>, // model_face.uv in pixels (0..16), or None
    e: &Cube,               // element bounds (pixels)
    c: glam::IVec3,         // this vertex corner: (0|1, 0|1, 0|1)
) -> glam::Vec2 {
    let (fx, fy, fz) = (e.from.x, e.from.y, e.from.z);
    let (tx, ty, tz) = (e.to.x, e.to.y, e.to.z);

    let x = if c.x == 0 { fx } else { tx };
    let y = if c.y == 0 { fy } else { ty };
    let z = if c.z == 0 { fz } else { tz };

    let dx = (tx - fx).max(1.0);
    let dy = (ty - fy).max(1.0);
    let dz = (tz - fz).max(1.0);

    // Face-local mapping: vertical faces V goes "down" with +Y; top face V goes +Z.
    let (u_norm, v_norm) = match dir {
        Direction::North => ((x - fx) / dx, (ty - y) / dy),
        Direction::South => ((tx - x) / dx, (ty - y) / dy),
        Direction::East => ((z - fz) / dz, (ty - y) / dy),
        Direction::West => ((tz - z) / dz, (ty - y) / dy),
        Direction::Up => ((x - fx) / dx, (z - fz) / dz),
        Direction::Down => ((x - fx) / dx, (tz - z) / dz),
    };

    let (u1, v1, u2, v2) = match rect {
        Some([u1, v1, u2, v2]) => (u1 / 16.0, v1 / 16.0, u2 / 16.0, v2 / 16.0),
        None => (0.0, 0.0, 1.0, 1.0),
    };

    glam::Vec2::new(u1 + (u2 - u1) * u_norm, v1 + (v2 - v1) * v_norm)
}

#[inline]
fn uv_shrink_ratio_for(_tex_idx: u32, _assets: &MeshAssets) -> f32 {
    // TODO: hook into your atlas if you have per-sprite shrink ratios.
    // Vanilla often ends up around 1/256 for 16px sprites on a typical atlas.
    1.0 / 256.0
}

#[inline]
fn shrink_uvs_toward_center(
    mut uvs: [glam::Vec2; 4],
    rect: Option<[f32; 4]>,
    ratio: f32,
) -> [glam::Vec2; 4] {
    if ratio <= 0.0 {
        return uvs;
    }

    let (u1, v1, u2, v2) = match rect {
        Some([u1, v1, u2, v2]) => (u1 / 16.0, v1 / 16.0, u2 / 16.0, v2 / 16.0),
        None => (0.0, 0.0, 1.0, 1.0),
    };
    let center = glam::Vec2::new((u1 + u2) * 0.5, (v1 + v2) * 0.5);
    for uv in &mut uvs {
        *uv = center.lerp(*uv, 1.0 - ratio);
    }
    uvs
}

fn offset_to_coord(offset: IVec3, element: &Cube) -> glam::Vec3 {
    glam::Vec3::new(
        if offset.x == 0 { element.from.x } else { element.to.x },
        if offset.y == 0 { element.from.y } else { element.to.y },
        if offset.z == 0 { element.from.z } else { element.to.z },
    )
}

fn rotate_corner_offset(offset: IVec3, x_rot: i32, y_rot: i32) -> IVec3 {
    // Rotate a face corner (0 or 1 in each axis) the same way as positions, then snap back to 0/1.
    let p = rotate_position(Vec3::new(offset.x as f32, offset.y as f32, offset.z as f32), x_rot, y_rot);
    IVec3::new((p.x >= 0.5) as i32, (p.y >= 0.5) as i32, (p.z >= 0.5) as i32)
}

fn compute_ao(local: IVec3, offset: IVec3, dir: Direction, section: &LocalSection) -> u32 {
    let get = |p: IVec3| {
        if p.x < 0 || p.y < 0 || p.z < 0 || p.x >= 18 || p.y >= 18 || p.z >= 18 {
            return false;
        }
        let state = section.blocks[p.x as usize][p.y as usize][p.z as usize];
        let dyn_state: Box<dyn BlockTrait> = Box::from(state);
        if state.is_air() {
            return false;
        }
        dyn_state.behavior().can_occlude && state.is_collision_shape_full()
    };

    let ox = offset.x * 2 - 1;
    let oy = offset.y * 2 - 1;
    let oz = offset.z * 2 - 1;

    match dir {
        Direction::East | Direction::West => {
            let side1 = get(local + IVec3::new(ox, 0, oz));
            let side2 = get(local + IVec3::new(ox, oy, 0));
            let corner = get(local + IVec3::new(ox, oy, oz));
            ao(side1, side2, corner)
        }
        Direction::Up | Direction::Down => {
            let side1 = get(local + IVec3::new(0, oy, oz));
            let side2 = get(local + IVec3::new(ox, oy, 0));
            let corner = get(local + IVec3::new(ox, oy, oz));
            ao(side1, side2, corner)
        }
        Direction::North | Direction::South => {
            let side1 = get(local + IVec3::new(0, oy, oz));
            let side2 = get(local + IVec3::new(ox, 0, oz));
            let corner = get(local + IVec3::new(ox, oy, oz));
            ao(side1, side2, corner)
        }
    }
}

fn ao(side1: bool, side2: bool, corner: bool) -> u32 {
    if side1 && side2 {
        0
    } else {
        3 - ((side1 || side2) as u32 + corner as u32)
    }
}

fn face_dir_vec(dir: Direction) -> glam::Vec3 {
    match dir {
        Direction::North => glam::Vec3::new(0.0, 0.0, -1.0),
        Direction::South => glam::Vec3::new(0.0, 0.0, 1.0),
        Direction::West => glam::Vec3::new(-1.0, 0.0, 0.0),
        Direction::East => glam::Vec3::new(1.0, 0.0, 0.0),
        Direction::Down => glam::Vec3::new(0.0, -1.0, 0.0),
        Direction::Up => glam::Vec3::new(0.0, 1.0, 0.0),
    }
}

struct Face {
    offsets: [IVec3; 4],
    dir: Direction,
}

// Corner order matters for UV orientation & triangle winding.
const FACES: [Face; 6] = [
    Face { offsets: [IVec3::new(0, 1, 0), IVec3::new(0, 1, 1), IVec3::new(1, 1, 1), IVec3::new(1, 1, 0)], dir: Direction::Up },
    Face { offsets: [IVec3::new(0, 0, 0), IVec3::new(1, 0, 0), IVec3::new(1, 0, 1), IVec3::new(0, 0, 1)], dir: Direction::Down },
    Face { offsets: [IVec3::new(0, 0, 1), IVec3::new(1, 0, 1), IVec3::new(1, 1, 1), IVec3::new(0, 1, 1)], dir: Direction::South },
    Face { offsets: [IVec3::new(0, 0, 0), IVec3::new(0, 1, 0), IVec3::new(1, 1, 0), IVec3::new(1, 0, 0)], dir: Direction::North },
    Face { offsets: [IVec3::new(1, 0, 0), IVec3::new(1, 1, 0), IVec3::new(1, 1, 1), IVec3::new(1, 0, 1)], dir: Direction::East },
    Face { offsets: [IVec3::new(0, 0, 0), IVec3::new(0, 0, 1), IVec3::new(0, 1, 1), IVec3::new(0, 1, 0)], dir: Direction::West },
];
