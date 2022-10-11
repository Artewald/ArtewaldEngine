use renderer::setup_renderer_and_run;
use voxel::{Chunk, math::{Vec2, Vec3, Vec4}};

mod renderer;
mod voxel;

fn main() {
    let mut chunk = Chunk::new(Vec2::new(0, 0), 16);
    chunk.fill_voxels(Vec3::new(Vec2::new(15, 40), Vec2::new(20, 50), Vec2::new(10, 80)), Vec4::new(0.0, 0.0, 1.0, 1.0));
    //chunk.print_chunk();
    let leaf_data = chunk.get_sorted_leaf_nodes();
    //for data in leaf_data {
    //    println!("{}", data);
    //}

    setup_renderer_and_run(leaf_data);
}
