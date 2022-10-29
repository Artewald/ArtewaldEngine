use std::time::Instant;

use nalgebra::{Vector2, Vector3, Vector4};
use renderer::setup_renderer_and_run;
use voxel::{Chunk};
use artewald_engine_lib::threadpool::ThreadPoolHelper;
use artewald_engine_lib::voxel;

mod renderer;

// Optimization can be done by using flamegraph and cargo-asm
fn main() {
    //let time = Instant::now();
    let thread_pool = ThreadPoolHelper::new(Some(0));
    let mut chunk = Chunk::new(Vector2::new(0, 0), 16);
    chunk.fill_voxels(thread_pool.clone(), Vector3::new(Vector2::new(5, 10), Vector2::new(0, 10), Vector2::new(2, 15)), Vector4::new(1.0, 0.0, 0.0, 1.0));
    chunk.fill_voxels(thread_pool.clone(), Vector3::new(Vector2::new(0, 3), Vector2::new(2, 5), Vector2::new(5, 10)), Vector4::new(0.0, 1.0, 0.0, 1.0));
    let voxel_data = chunk.get_oct_tree(Vector3::new(0.0, 0.0, 0.0), (90.0 as f32/1080.0 as f32).to_radians());
    if voxel_data.len() > (u32::MAX-2) as usize {
        panic!("There are more than u32-2 indices in the voxel array for the gpu, that's too much for the GPU");
    }
    //println!("{:?}", data);
    setup_renderer_and_run(voxel_data);
}
