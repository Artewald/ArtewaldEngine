use self::math::{Vec2, Vec4, Vec3};

mod math;

#[derive(Debug, Clone)]
pub struct Voxel {
    pub x_range: Vec2<f32>,
    pub y_range: Vec2<f32>,
    pub z_range: Vec2<f32>,
    pub color: Vec4<f32>,
    pub children: [Option<Box<Voxel>>; 8]
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub position: Vec3<i128>,
    pub start_voxels: [Voxel; 8],
}

#[derive(Debug, Copy, Clone)]
struct ColorWeight {
    color: Vec4<f32>,
    weight: f32,
}

impl Voxel {
    fn recoursive_color_calculator(&mut self) -> ColorWeight {
        let mut actual_children: Vec<Voxel> = vec![];
        for child in self.children.clone() {
            if !child.is_none() {
                actual_children.push(*child.unwrap())
            }
        }

        if actual_children.len() == 0 {
            return ColorWeight {color: self.color, weight: self.x_range.length() * self.y_range.length() * self.z_range.length()}
        }

        let mut color_weights: Vec<ColorWeight> = vec![];
        let mut cpy: Vec<ColorWeight> = vec![];

        for mut child in actual_children {
            let val = child.recoursive_color_calculator();
            color_weights.push(val);
            cpy.push(val);
        }

        let mut total_weight: f32 = 0.0;
        for color_weight in color_weights {
            total_weight += color_weight.weight;
        }

        self.color = Vec4::new(0.0, 0.0, 0.0, 0.0);
        for color_weight in cpy {
            let percent: f32 = color_weight.weight/total_weight;
            self.color = self.color + (color_weight.color * Vec4::new(percent, percent, percent, percent));
        }

        ColorWeight {color: self.color, weight: self.x_range.length() * self.y_range.length() * self.z_range.length()}
    }

    pub fn recalculate_colors(&mut self) {
        self.recoursive_color_calculator();
    }
}