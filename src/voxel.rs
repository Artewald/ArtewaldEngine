use std::fmt::{Display, Formatter, Result};

use self::math::{Vec2, Vec4, Vec3};

pub mod math;


// Constants
const CHUNKPOWER: u32 = 8;
const CHUNKSIZE: u32 = (4 as u32).pow(CHUNKPOWER);


#[derive(Debug, Clone)]
pub struct Voxel {
    pub x_range: Vec2<f32>,
    pub y_range: Vec2<f32>,
    pub z_range: Vec2<f32>,
    pub color: Vec4<f32>,
    pub children: [Option<Box<Voxel>>; 8]
}

impl Display for Voxel {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{{\n\tx_range: {},\n\ty_range: {},\n\tz_range: {},\n\tcolor: {}\n}}", self.x_range, self.y_range, self.z_range, self.color)
    }
}

/// # NOTE!
/// The `depth` tells us how many levels of voxels there are in the chunk.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub position: Vec2<i128>,
    depth: u32,
    pub start_voxel: Voxel,
}

#[derive(Debug, Copy, Clone)]
struct ColorWeight {
    color: Vec4<f32>,
    weight: f32,
}

impl Display for ColorWeight {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{{\n\tColor: {},\n\tWeight: {}\n}}", self.color, self.weight)
    }
}

impl Voxel {
    fn recoursive_color_calculator(&mut self) -> ColorWeight {
        let mut actual_children: Vec<Voxel> = vec![];
        for child in self.children.clone() {
            if !child.is_none() {
                actual_children.push(*child.unwrap())
            }
        }

        let x_length = self.x_range.length().abs();
        let y_length = self.y_range.length().abs();
        let z_length = self.z_range.length().abs();

        if actual_children.len() == 0 {
            //println!("x: {}, y: {}, z: {}, result: {}, color: {}", self.x_range, self.y_range, self.z_range, self.x_range.length().abs() * self.y_range.length().abs() * self.z_range.length().abs(), self.color);
            return ColorWeight {color: self.color, weight: x_length * y_length * z_length}
        }

        let mut color_weights: Vec<ColorWeight> = vec![];
        let mut cpy: Vec<ColorWeight> = vec![];

        let empty_weight = (x_length/2.0) * (y_length/2.0) * (z_length/2.0);

        for i in 0..self.children.len() {
            let mut cw = ColorWeight {color: Vec4::new(0.0, 0.0, 0.0, 0.0), weight: empty_weight};
            if self.children[i].is_some() {
                cw = self.children[i].as_deref_mut().unwrap().recoursive_color_calculator();
            }

            color_weights.push(cw);
            cpy.push(cw);
        }

        let mut total_weight: f32 = 0.0;
        for color_weight in color_weights {
            total_weight += color_weight.weight;
        }

        if total_weight < 0.0 {
            panic!("Voxel: TheColorWeight should never be less than zero!");
        }

        self.color = Vec4::new(0.0, 0.0, 0.0, 0.0);
        for color_weight in cpy {
            let percent: f32 = color_weight.weight/total_weight;
            self.color = self.color + (color_weight.color * Vec4::new(percent, percent, percent, percent));
        }

        ColorWeight {color: self.color, weight: x_length * y_length * z_length}
    }

    fn traverse_and_color(&mut self, depth: u32, current_depth: u32, fill_range: Vec3<Vec2<u32>>, color: Vec4<f32>) {
        if depth == current_depth {
            self.color = color;
            return;
        }

        if self.x_range.in_range(Vec2::new(fill_range.x.x as f32, fill_range.x.y as f32))
           && self.y_range.in_range(Vec2::new(fill_range.y.x as f32, fill_range.y.y as f32))
           && self.z_range.in_range(Vec2::new(fill_range.z.x as f32, fill_range.z.y as f32))
        {
            self.color = color;
            return;
        }

        let size = (CHUNKSIZE/(2 as u32).pow(current_depth)) as f32;

        for x in 0..self.children.len()/4 {
            for y in 0..self.children.len()/4 {
                for z in 0..self.children.len()/4 {

                    let i = z + y*2 + x*4;
                    let new_x_u_range = Vec2::new((self.x_range.x + (x as f32 *size)) as u32, (self.x_range.x + ((x as f32 + 1.0)*size)) as u32);
                    let new_y_u_range = Vec2::new((self.y_range.x + (y as f32 *size)) as u32, (self.y_range.x + ((y as f32 + 1.0)*size)) as u32);
                    let new_z_u_range = Vec2::new((self.z_range.x + (z as f32 *size)) as u32, (self.z_range.x + ((z as f32 + 1.0)*size)) as u32);

                    if fill_range.x.overlapping(new_x_u_range)
                       && fill_range.y.overlapping(new_y_u_range)
                       && fill_range.z.overlapping(new_z_u_range) 
                    {
                        if self.children[i].is_none() {
                            self.children[i] = Some(Box::new(Voxel { x_range: Vec2::new(new_x_u_range.x as f32, new_x_u_range.y as f32),
                                                                     y_range: Vec2::new(new_y_u_range.x as f32, new_y_u_range.y as f32),
                                                                     z_range: Vec2::new(new_z_u_range.x as f32, new_z_u_range.y as f32),
                                                                     color: Vec4::new(0.0, 0.0, 0.0, 0.0),
                                                                     children: Default::default() }));
                        }

                        self.children[i].as_deref_mut().unwrap().traverse_and_color(depth, current_depth+1, fill_range, color);
                    }    
                }
            }
        }
    }

    fn traverse_and_print_voxel(self, current_depth: u32) {
        println!("Depth: {}, Voxel{}", current_depth, self);
        for child in self.children {
            if child.is_none() {
                continue;
            }
            child.unwrap().traverse_and_print_voxel(current_depth + 1);
        }
    }
}

impl Chunk {
    /// Initializes empty chunk with specified depth.
    pub fn new(position: Vec2<i128>, depth: u32) -> Chunk {
        Chunk { position: position, depth: depth, start_voxel: Voxel { x_range: Vec2::new(0 as f32, CHUNKSIZE as f32),
                                                                       y_range: Vec2::new(0 as f32, CHUNKSIZE as f32),
                                                                       z_range: Vec2::new(0 as f32, CHUNKSIZE as f32),
                                                                       color: Vec4::new(0.0, 0.0, 0.0, 0.0),
                                                                       children: Default::default() } }
    }

    pub fn fill_voxels(&mut self, fill_range: Vec3<Vec2<u32>>, color: Vec4<f32>) {
        self.start_voxel.traverse_and_color(self.depth, 0, fill_range, color);
        self.start_voxel.recoursive_color_calculator();
    }

    pub fn print_chunk(self) {
        self.start_voxel.traverse_and_print_voxel(0);
    }
}