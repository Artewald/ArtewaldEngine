use std::{sync::{Arc, RwLock}, thread::JoinHandle};
use std::thread;

use bytemuck::{Pod, Zeroable};
use nalgebra::{Vector2, Vector4, Vector3};

use crate::voxel::math::{distance_between_points, view_cm_size};
use crate::threadpool::ThreadPoolHelper;

use self::math::{vec2_one_d_lenght, mul_vector4, vec2_one_d_in_range, vec2_one_d_overlapping};

//use self::math::{Vec2, Vec4, Vec3};

pub mod math;

// Constants
pub const CHUNKPOWER: u32 = 8;
pub const CHUNKSIZE: u32 = (4 as u32).pow(CHUNKPOWER);

// Structs
#[derive(Debug, Clone)]
pub struct Voxel {
    pub pos: Vector3<f32>,
    pub range: Vector2<f32>,
    // pub x_range: Vector2<f32>,
    // pub y_range: Vector2<f32>,
    // pub z_range: Vector2<f32>,
    pub color: Vector4<f32>,
    pub children: [Option<Box<Voxel>>; 8]
}

/// # NOTE!
/// The `depth` tells us how many levels of voxels there are in the chunk.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub position: Vector2<i128>,
    depth: u32,
    pub start_voxel: Voxel,
}

#[derive(Debug, Copy, Clone)]
struct ColorWeight {
    color: Vector4<f32>,
    weight: f32,
}

#[allow(dead_code)]
#[derive(Debug)]
struct ParallellVoxelData {
    pub handle: JoinHandle<Option<Box<Voxel>>>,
    pub index: usize,
}

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct VoxelData {
    // pub x_range: Vector2<f32>,
    // pub y_range: Vector2<f32>,
    // pub z_range: Vector2<f32>,
    pub pos: Vector3<f32>,
    pub range: Vector2<f32>,
    // For some reason using a Vector4 does not work, so I need to split the color into two Vector2s.
    pub color_rg: Vector2<f32>,
    pub color_ba: Vector2<f32>,
    pub _0_0_index: u32,
    pub _0_1_index: u32,
    pub _0_2_index: u32,
    pub _0_3_index: u32,
    pub _1_0_index: u32,
    pub _1_1_index: u32,
    pub _1_2_index: u32,
    pub _1_3_index: u32,
}

impl Voxel {
    /// # Panics
    /// This function panics if the weight(volume) of a node somehow becomes negative
    fn recursive_color_calculator(&mut self, thread_pool: Arc<RwLock<ThreadPoolHelper>>) -> ColorWeight {
        //TODO: Parallellize this function. Can be done in a similar way as the filling function.
        let mut actual_children: Vec<Voxel> = vec![];
        for child in self.children.clone() {
            if !child.is_none() {
                actual_children.push(*child.unwrap())
            }
        }

        // let length = vec2_one_d_lenght(self.range);

        let x_length = vec2_one_d_lenght(Vector2::new(self.pos.x + self.range.x, self.pos.x + self.range.y));
        let y_length = vec2_one_d_lenght(Vector2::new(self.pos.y + self.range.x, self.pos.y + self.range.y));
        let z_length = vec2_one_d_lenght(Vector2::new(self.pos.z + self.range.x, self.pos.z + self.range.y));

        if actual_children.len() == 0 {
            return ColorWeight {color: self.color, weight: x_length * y_length * z_length}
        }

        let mut color_weights: Vec<ColorWeight> = vec![];
        let mut cpy: Vec<ColorWeight> = vec![];
        //let mut join_handles = vec![];

        let empty_weight = (x_length/2.0) * (y_length/2.0) * (z_length/2.0);

        for i in 0..self.children.len() {
            //TODO: Fix the multithreading
            // if thread_pool.clone().write().unwrap().try_starting_thread() {
            //     //thread_pool.clone().read().unwrap().print_num_utilized();
            //     let thread_pool_cpy = thread_pool.clone();
            //     let mut child = self.children[i].clone();
            //     join_handles.push(thread::spawn(move || {
            //         let mut cw = ColorWeight {color: Vector4::new(0.0, 0.0, 0.0, 0.0), weight: empty_weight};
            //         if child.is_some() {
            //             cw = child.as_deref_mut().unwrap().recursive_color_calculator(thread_pool_cpy.clone());
            //         }
            //         thread_pool_cpy.read().unwrap().end_thread();
            //         cw
            //     }));
            //     continue;
            // }

            let mut cw = ColorWeight {color: Vector4::new(0.0, 0.0, 0.0, 0.0), weight: empty_weight};
            if self.children[i].is_some() {
                cw = self.children[i].as_deref_mut().unwrap().recursive_color_calculator(thread_pool.clone());
            }

            color_weights.push(cw);
            cpy.push(cw);
        }

        // for data in join_handles {
        //     let cw = data.join().unwrap();
        //     color_weights.push(cw);
        //     cpy.push(cw);
        // }

        let mut total_weight: f32 = 0.0;
        for color_weight in color_weights {
            if color_weight.weight < 0.0 {
                panic!("Voxel::recursive_color_calculator(): TheColorWeight should never be less than zero!");
            }
            total_weight += color_weight.weight;
        }

        self.color = Vector4::new(0.0, 0.0, 0.0, 0.0);
        for color_weight in cpy {
            let percent: f32 = color_weight.weight/total_weight;
            self.color = self.color + mul_vector4(color_weight.color, Vector4::new(percent, percent, percent, percent));
        }

        ColorWeight {color: self.color, weight: x_length * y_length * z_length}
    }

    fn traverse_and_color(&mut self, thread_pool: Arc<RwLock<ThreadPoolHelper>>, depth: u32, current_depth: u32, fill_range: Vector3<Vector2<u32>>, color: Vector4<f32>) {
        if depth == current_depth {
            self.color = color;
            return;
        }

        println!("{}", depth);

        if vec2_one_d_in_range(Vector2::new(self.pos.x + self.range.x, self.pos.x + self.range.y), Vector2::new(fill_range.x.x as f32, fill_range.x.y as f32))
           && vec2_one_d_in_range(Vector2::new(self.pos.y + self.range.x, self.pos.y + self.range.y), Vector2::new(fill_range.y.x as f32, fill_range.y.y as f32))
           && vec2_one_d_in_range(Vector2::new(self.pos.z + self.range.x, self.pos.z + self.range.y), Vector2::new(fill_range.z.x as f32, fill_range.z.y as f32))
        {
            self.color = color;
            return;
        }

        let size = (CHUNKSIZE/(2 as u32).pow(current_depth)) as f32;

        //let mut join_handles: Vec<ParallellVoxelData> = vec![];

        for x in 0..self.children.len()/4 {
            for y in 0..self.children.len()/4 {
                for z in 0..self.children.len()/4 {

                    let i = x + z*2 + y*4;
                    let new_range = Vector2::new(0_f32, size as f32);
                    let new_pos = Vector3::new(self.pos.x + (x as f32 * size), self.pos.y + (y as f32 * size), self.pos.z + (z as f32 * size));
                    // let new_x_u_range = Vector2::new((self.x_range.x + (x as f32 * size)) as u32, (self.x_range.x + ((x as f32 + 1.0) * size)) as u32);
                    // let new_y_u_range = Vector2::new((self.y_range.x + (y as f32 * size)) as u32, (self.y_range.x + ((y as f32 + 1.0) * size)) as u32);
                    // let new_z_u_range = Vector2::new((self.z_range.x + (z as f32 * size)) as u32, (self.z_range.x + ((z as f32 + 1.0) * size)) as u32);

                    if vec2_one_d_overlapping(fill_range.x, Vector2::new((new_pos.x + new_range.x) as u32, (new_pos.x + new_range.y) as u32))
                       && vec2_one_d_overlapping(fill_range.y, Vector2::new((new_pos.y + new_range.x) as u32, (new_pos.y + new_range.y) as u32))
                       && vec2_one_d_overlapping(fill_range.z, Vector2::new((new_pos.z + new_range.x) as u32, (new_pos.z + new_range.y) as u32)) 
                    {
                        if self.children[i].is_none() {
                            self.children[i] = Some(Box::new(Voxel { // x_range: Vector2::new(new_x_u_range.x as f32, new_x_u_range.y as f32),
                                                                    //  y_range: Vector2::new(new_y_u_range.x as f32, new_y_u_range.y as f32),
                                                                    //  z_range: Vector2::new(new_z_u_range.x as f32, new_z_u_range.y as f32),
                                                                     pos: new_pos,
                                                                     range: new_range,
                                                                     color: Vector4::new(0.0, 0.0, 0.0, 0.0),
                                                                     children: Default::default() }));
                        }
                        //TODO: Fix the multithreading
                        // if thread_pool.clone().write().unwrap().try_starting_thread() {
                        //     let thread_pool_clone = thread_pool.clone();
                        //     let mut child = self.children[i].clone();
                        //     join_handles.push(ParallellVoxelData { 
                        //         handle: thread::spawn(move || {
                        //             child.as_deref_mut().unwrap().traverse_and_color(thread_pool_clone.clone(), depth, current_depth+1, fill_range, color);
                        //             thread_pool_clone.clone().read().unwrap().end_thread();
                        //             child
                        //             }),
                        //         index: i });
                        //     continue;
                        // }
                        self.children[i].as_deref_mut().unwrap().traverse_and_color(thread_pool.clone(), depth, current_depth+1, fill_range, color);
                    }
                }
            }
        }

        // for thread_data in join_handles {
        //     self.children[thread_data.index] = thread_data.handle.join().unwrap();
        // }
    }

    #[allow(dead_code)]
    fn traverse_and_print_voxel(&self, current_depth: u32) {
        println!("Depth: {}, Voxel{:?}", current_depth, self);
        for child in self.children.clone() {
            if child.is_none() {
                continue;
            }
            child.unwrap().traverse_and_print_voxel(current_depth + 1);
        }
    }

    fn traverse_and_append(&self, camera_pos: Vector3<f64>, pixel_rad: f32, current_vec_len: u32) -> Vec<VoxelData> {

        if view_cm_size(pixel_rad, distance_between_points(camera_pos, Vector3::new(self.range.x as f64, self.range.x as f64, self.range.x as f64))) >= vec2_one_d_lenght(self.range) {
            // return vec![VoxelData {x_range: self.x_range, y_range: self.y_range, z_range: self.z_range, color_rg:Vector2::new(self.color.x, self.color.y), color_ba:Vector2::new(self.color.z, self.color.w), _0_0_index: u32::MAX, _0_1_index: u32::MAX, _0_2_index: u32::MAX, _0_3_index: u32::MAX, _1_0_index: u32::MAX, _1_1_index: u32::MAX, _1_2_index: u32::MAX, _1_3_index: u32::MAX }]
            return vec![VoxelData {pos: self.pos, range: self.range, color_rg:Vector2::new(self.color.x, self.color.y), color_ba:Vector2::new(self.color.z, self.color.w), _0_0_index: u32::MAX, _0_1_index: u32::MAX, _0_2_index: u32::MAX, _0_3_index: u32::MAX, _1_0_index: u32::MAX, _1_1_index: u32::MAX, _1_2_index: u32::MAX, _1_3_index: u32::MAX }]
        }
        
        let mut voxels: Vec<VoxelData> = vec![];
        let mut index_array: [u32; 8] = [u32::MAX; 8];
        for y in 0..2_usize {
            for x in 0..4_usize {
                let child = self.children[x+y*4].clone();
                if child.is_some() {
                    let mut data = child.unwrap().traverse_and_append(camera_pos, pixel_rad, voxels.len() as u32 + current_vec_len);//child.clone().read().unwrap().unwrap().traverse_and_append(camera_pos, pixel_rad);
                    voxels.append(&mut data);
                    index_array[x+y*4] = voxels.len() as u32 - 1 + current_vec_len;
                }
            }
        }

        if voxels.len() == 0 {
            //return vec![VoxelData {x_range: self.x_range, y_range: self.y_range, z_range: self.z_range, color_rg: Vector2::new(self.color.x, self.color.y), color_ba: Vector2::new(self.color.z, self.color.w), _0_0_index: u32::MAX, _0_1_index: u32::MAX, _0_2_index: u32::MAX, _0_3_index: u32::MAX, _1_0_index: u32::MAX, _1_1_index: u32::MAX, _1_2_index: u32::MAX, _1_3_index: u32::MAX }]
            return vec![VoxelData {pos: self.pos, range: self.range, color_rg: Vector2::new(self.color.x, self.color.y), color_ba: Vector2::new(self.color.z, self.color.w), _0_0_index: u32::MAX, _0_1_index: u32::MAX, _0_2_index: u32::MAX, _0_3_index: u32::MAX, _1_0_index: u32::MAX, _1_1_index: u32::MAX, _1_2_index: u32::MAX, _1_3_index: u32::MAX }]
        }

        //voxels.append(&mut vec![VoxelData {x_range: self.x_range, y_range: self.y_range, z_range: self.z_range, color_rg: Vector2::new(self.color.x, self.color.y), color_ba:Vector2::new(self.color.z, self.color.w), _0_0_index: index_array[0], _0_1_index: index_array[1], _0_2_index: index_array[2], _0_3_index: index_array[3], _1_0_index: index_array[4], _1_1_index: index_array[5], _1_2_index: index_array[6], _1_3_index: index_array[7] }]);
        voxels.append(&mut vec![VoxelData {pos: self.pos, range: self.range, color_rg: Vector2::new(self.color.x, self.color.y), color_ba:Vector2::new(self.color.z, self.color.w), _0_0_index: index_array[0], _0_1_index: index_array[1], _0_2_index: index_array[2], _0_3_index: index_array[3], _1_0_index: index_array[4], _1_1_index: index_array[5], _1_2_index: index_array[6], _1_3_index: index_array[7] }]);
        voxels
    }


}

impl Chunk {
    /// Initializes empty chunk with specified depth.
    /// # Panics
    /// The function panics if the `depth` value is higher than `CHUNKPOWER`*2. 
    pub fn new(position: Vector2<i128>, depth: u32) -> Chunk {
        if depth > CHUNKPOWER*2 {
            panic!("Chunk::new(): A depth higher than the CHUNKPOWER*2 is not supported at this time");
        }
        Chunk { position: position, depth: depth, start_voxel: Voxel { // x_range: Vector2::new(0 as f32, CHUNKSIZE as f32),
                                                                    //    y_range: Vector2::new(0 as f32, CHUNKSIZE as f32),
                                                                    //    z_range: Vector2::new(0 as f32, CHUNKSIZE as f32),
                                                                       pos: Vector3::new(0_f32, 0_f32, 0_f32),
                                                                       range: Vector2::new(0_f32, CHUNKSIZE as f32),
                                                                       color: Vector4::new(0.0, 0.0, 0.0, 0.0),
                                                                       children: Default::default() }}
    }

    /// Fills the voxels in the specified range. However, the precision just goes as low as the `depth` specified for the chunk. 
    pub fn fill_voxels(&mut self, thread_pool: Arc<RwLock<ThreadPoolHelper>>, fill_range: Vector3<Vector2<u32>>, color: Vector4<f32>) {
        self.start_voxel.traverse_and_color(thread_pool.clone(), self.depth, 0, fill_range, color);
        self.start_voxel.recursive_color_calculator(thread_pool.clone());
    }

    #[allow(dead_code)]
    pub fn print_chunk(&self) {
        self.start_voxel.traverse_and_print_voxel(0);
    }

    /// Get's the oct tree data for this chunk so that it can be used on the GPU
    pub fn get_oct_tree(&self, camera_pos: Vector3<f64>, pixel_rad: f32) -> Vec<VoxelData> {
        self.start_voxel.traverse_and_append( camera_pos, pixel_rad, 0)
    }
}