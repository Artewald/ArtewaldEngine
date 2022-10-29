use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix4, Vector3, Vector4, Point3};
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorType};
use vulkano::device::physical::{PhysicalDeviceType};
use vulkano::format::Format;
use vulkano::image::view::{ImageView};
use vulkano::image::{ImageUsage, SwapchainImage, ImageAccess, ImageViewAbstract, StorageImage, ImageDimensions};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::library::VulkanLibrary;
use vulkano::memory::pool::{PotentialDedicatedAllocation, StandardMemoryPool, StandardMemoryPoolAlloc};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo, PresentMode, SwapchainCreationError};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, Queue, DeviceExtensions};
use vulkano_win::VkSurfaceBuild;
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoop;
use winit::window::{WindowBuilder, Window};

use std::sync::Arc;

use crate::voxel::VoxelData;

pub struct VulkanData {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub swapchain: Arc<Swapchain<Window>>,
    pub images: Vec<Arc<SwapchainImage<Window>>>,
}

pub struct RenderImageData {
    pub image: Arc<StorageImage>,
    pub buffer: Arc<CpuAccessibleBuffer<[u8]>>,
    pub view: Arc<ImageView<StorageImage<Arc<StandardMemoryPool>>>>,
}

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct CameraData {
    pub field_of_view: u32,
    pub render_distance: f32,
    pub aspect_ratio: f32,
    pub fov_tan: f32,
    pub raw_camera_to_world: Matrix4<f32>,
    pub clear_color: Vector4<f32>,
}

impl CameraData {
    fn create_camera_to_world_space(forward: Vector3<f32>, up: Vector3<f32>) -> Matrix4<f32> {
        let camera_to_world = Matrix4::look_at_rh(&Point3::new(0.0, 0.0, 0.0), &Point3::new(forward.x, forward.y, forward.z), &up);//Matrix4::identity();
        // for x in 0..3 {
        //     camera_to_world[x * 4] = right[x];
        //     camera_to_world[x * 4 + 1] = up[x];
        //     camera_to_world[x * 4 + 2] = -forward[x];
        // }
        // camera_to_world = camera_to_world.try_inverse().unwrap();
        camera_to_world
    }

    pub fn new(fov: u32, render_distance: f32, aspect_ratio: f32, target: Vector3<f32>, up_ref: Vector3<f32>, clear_color: Vector4<f32>) -> Self {
        let forward: Vector3<f32> = target.normalize();
        let right: Vector3<f32> = forward.cross(&up_ref).normalize();
        let up: Vector3<f32> = right.cross(&forward).normalize();
        CameraData { field_of_view: fov,
                        render_distance,
                        aspect_ratio, 
                        fov_tan: (fov as f32/2.0).to_radians().tan(),
                        raw_camera_to_world: Self::create_camera_to_world_space(forward, up),
                        clear_color: clear_color, }
    }

    pub fn update_camera_dir(&mut self, target: Vector3<f32>, up_ref: Vector3<f32>) {
        let new_forward: Vector3<f32> = target.normalize();
        let new_right: Vector3<f32> = new_forward.cross(&up_ref).normalize();
        let new_up: Vector3<f32> = new_right.cross(&new_forward).normalize();
        self.raw_camera_to_world = Self::create_camera_to_world_space(new_forward, new_up);
    }
}

pub fn setup_vulkan(event_loop: &EventLoop<()>) -> (VulkanData, Arc<Surface<Window>>) {
    let lib = VulkanLibrary::new().unwrap();
    let req_ext = vulkano_win::required_extensions(&lib);

    let instance = Instance::new(lib, InstanceCreateInfo {
        enabled_extensions: req_ext, 
        enumerate_portability: true,
        ..Default::default()
    }).expect("failed to create instance");
    
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        khr_storage_buffer_storage_class: true,
        khr_swapchain_mutable_format: true,
        ..DeviceExtensions::empty()
    };

    let surface = WindowBuilder::new().with_title("Artewald Engine").build_vk_surface(&event_loop, instance.clone()).unwrap();

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            p.supported_extensions().contains(&device_extensions)
        }).filter_map(|p| {
            p.queue_family_properties().iter().enumerate().position(|(i, q)| {
                    q.queue_flags.compute && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| {
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("No suitable physical device found");

    println!("Using {}", physical_device.clone().properties().device_name);

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {queue_family_index: queue_family_index as u32, ..Default::default()}],
            ..Default::default()
        },
    ).expect("failed to create device");

    let queue = queues.next().unwrap();

    let (swapchain, images) = {
        let surface_capabilities = device.physical_device().surface_capabilities(&surface, Default::default()).unwrap();

        let image_format = Some(device.physical_device().surface_formats(&surface, Default::default()).unwrap()[0].0);

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format,
                image_extent: surface.window().inner_size().into(),
                image_usage: ImageUsage {
                    storage: true,
                    color_attachment: true,
                    transfer_dst: true,
                    ..Default::default()
                },
                composite_alpha: surface_capabilities.supported_composite_alpha.iter().next().unwrap(),
                present_mode: PresentMode::Fifo,
                ..Default::default()
            }
        ).unwrap()
    };

    (VulkanData { instance: instance.clone(), device: device.clone(), queue: queue.clone(), swapchain: swapchain, images: images }, surface)
}

pub fn create_raw_camera_data_buffer(data: CameraData, device: Arc<Device>) -> Arc<CpuAccessibleBuffer<CameraData>> {
    CpuAccessibleBuffer::from_data(device.clone(), BufferUsage {storage_buffer: true, ..BufferUsage::empty()}, false, data).unwrap()
}

pub fn create_voxel_buffer(data: Vec<VoxelData>, device: Arc<Device>) -> Arc<CpuAccessibleBuffer<[VoxelData], PotentialDedicatedAllocation<StandardMemoryPoolAlloc>>> {
    CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage {storage_buffer: true, ..BufferUsage::empty()}, false, data.into_iter()).unwrap()
}

pub fn recreate_swapchain(vulkan_data: &mut VulkanData, dim: PhysicalSize<u32>) {
    let (new_sc, new_imgs) = match vulkan_data.swapchain.recreate(SwapchainCreateInfo {image_extent: dim.into(), ..vulkan_data.swapchain.create_info()}) {
        Ok(r) => r,
        Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
        Err(e) => panic!("Failed to recreate the swapchain, error: {}", e),
    };

    vulkan_data.swapchain = new_sc;
    vulkan_data.images = new_imgs;

}

pub fn create_sets(set_layouts: &[Arc<DescriptorSetLayout>], voxel_buffer: Arc<CpuAccessibleBuffer<[VoxelData]>>, misc_buffer: Arc<CpuAccessibleBuffer<CameraData>>, img_view: Arc<dyn ImageViewAbstract>) -> Vec<Arc<PersistentDescriptorSet>> {
    let mut sets = vec![];

    for set_layout in set_layouts {
        let mut visited: Vec<DescriptorType> = vec![];
        for x in set_layout.bindings() {
            //println!("{:?}", x.1.descriptor_type);
            if visited.contains(&x.1.descriptor_type) {
                continue;
            }

            if x.1.descriptor_type == DescriptorType::StorageBuffer {
                sets.push(PersistentDescriptorSet::new(set_layout.clone(), [WriteDescriptorSet::buffer(0, voxel_buffer.clone()), WriteDescriptorSet::buffer(1, misc_buffer.clone())]).unwrap());
                
            } else if x.1.descriptor_type == DescriptorType::StorageImage {
                sets.push(PersistentDescriptorSet::new(set_layout.clone(), [WriteDescriptorSet::image_view(0, img_view.clone())]).unwrap())
            } else {
                panic!("There exists an unused descriptorset, it should be implemented!");
            }

            visited.push(x.1.descriptor_type);
        }
    }

    sets
}

pub fn create_render_image(vulkan_data: &mut VulkanData) -> RenderImageData {
    let image = StorageImage::new(
        vulkan_data.device.clone(),
        ImageDimensions::Dim2d {
            width: vulkan_data.images[0].dimensions().width(),
            height: vulkan_data.images[0].dimensions().height(),
            array_layers: 1,
        }, 
        Format::R8G8B8A8_UNORM,
        Some(vulkan_data.queue.queue_family_index()),
    ).unwrap();

    let buffer = CpuAccessibleBuffer::from_iter(
        vulkan_data.device.clone(),
        BufferUsage {
            transfer_src: true,
            transfer_dst: true,
            storage_buffer: true,
            ..Default::default()
        },
        false,
        (0..vulkan_data.images[0].dimensions().width() * vulkan_data.images[0].dimensions().height() * 4).map(|_| 0u8)
    ).unwrap();

    let view = ImageView::new_default(image.clone()).unwrap();

    RenderImageData { image, buffer, view }
}









pub fn create_main_shader(device: Arc<Device>) -> Arc<ShaderModule> {
    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
            #version 450
            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
            
            struct VoxelData
            {
                vec2 x_range;
                vec2 y_range;
                vec2 z_range;
                vec2 color_rg;
                vec2 color_ba;
                uint _0_0_index;
                uint _0_1_index;
                uint _0_2_index;
                uint _0_3_index;
                uint _1_0_index;
                uint _1_1_index;
                uint _1_2_index;
                uint _1_3_index;
            };
            
            layout(set = 0, binding = 0) readonly buffer Data {
                VoxelData data[];
            } voxel_data;
            
            layout(set = 0, binding = 1)  readonly buffer RawCameraData {
                uint field_of_view;
                float render_distance;
                float aspectRatio;
                float fov_tan;
                mat4 raw_camera_to_world;
                vec4 clear_color;
            } rawCameraData;
            
            layout(set = 1, binding = 0, rgba8) uniform image2D img_out; 
            
            struct ColorHit {
                bool hit;
                vec4 color;
            };
            
            struct Camera {
                uint field_of_view;
                float render_distance;
                float aspectRatio;
                float fov_tan;
                mat4 camera_to_world;
                vec4 clear_color;
            };
            
            // Const variables
            vec3 lookingDir = vec3(0.0, 0.0, 1.0);
            vec3 currentPos = vec3(0.0, 0.0, 0.0);
                        
            // Helper functions
            ColorHit voxel_hit(ivec3 pos, vec4 clear_col) {
                ColorHit ret_val;
                ret_val.hit = false;
            
                uint index = voxel_data.data.length()-1;
                uint counter = 0;
                uint max_count = 20;
                while (counter < max_count && !ret_val.hit) {
                    counter += 1;
            
                    VoxelData current_voxel = voxel_data.data[index];
                    ret_val.color = vec4(current_voxel.color_rg, current_voxel.color_ba);
            
                    if (!(current_voxel.x_range.x <= pos.x && pos.x <= current_voxel.x_range.y &&
                        current_voxel.y_range.x <= pos.y && pos.y <= current_voxel.y_range.y &&
                        current_voxel.z_range.x <= pos.z && pos.z <= current_voxel.z_range.y)) {
                        break;
                    }
            
                    if (current_voxel._0_0_index == uint(-1) && current_voxel._0_1_index == uint(-1) &&
                        current_voxel._0_2_index == uint(-1) && current_voxel._0_3_index == uint(-1) &&
                        current_voxel._1_0_index == uint(-1) && current_voxel._1_1_index == uint(-1) &&
                        current_voxel._1_2_index == uint(-1) && current_voxel._1_3_index == uint(-1)) {
                            ret_val.hit = true;
                            break;
                    }
            
                    bool x_small = false;
                    bool y_small = false;
                    bool z_small = false;
                
                    if (pos.x < (current_voxel.x_range.x + current_voxel.x_range.y)/2) x_small = true;
                    if (pos.y < (current_voxel.y_range.x + current_voxel.y_range.y)/2) y_small = true;
                    if (pos.z < (current_voxel.z_range.x + current_voxel.z_range.y)/2) z_small = true;
            
                    if (x_small && y_small && z_small) {
                        if (current_voxel._0_0_index != uint(-1)) index = current_voxel._0_0_index;
                        else break;
                    } else if (!x_small && y_small && z_small) {
                        if (current_voxel._0_1_index != uint(-1)) index = current_voxel._0_1_index;
                        else break;
                    } else if (x_small && y_small && !z_small) {
                        if (current_voxel._0_2_index != uint(-1)) index = current_voxel._0_2_index;
                        else break;
                    } else if (!x_small && y_small && !z_small) {
                        if (current_voxel._0_3_index != uint(-1)) index = current_voxel._0_3_index;
                        else break;
                    } else if (x_small && !y_small && z_small) {
                        if (current_voxel._1_0_index != uint(-1)) index = current_voxel._1_0_index;
                        else break;
                    } else if (!x_small && !y_small && z_small) {
                        if (current_voxel._1_1_index != uint(-1)) index = current_voxel._1_1_index;
                        else break;
                    } else if (x_small && !y_small && !z_small) {
                        if (current_voxel._1_2_index != uint(-1)) index = current_voxel._1_2_index;
                        else break;
                    } else if (!x_small && !y_small && !z_small) {
                        if (current_voxel._1_3_index != uint(-1)) index = current_voxel._1_3_index;
                        else break;
                    } else {
                        ret_val.hit = true;
                        ret_val.color = vec4(0.75, 0.5, 0.25, 1.0);
                    }
            
                }
                
                if (!ret_val.hit) ret_val.color = clear_col;
            
                return ret_val;
            }
            
            // Main         
            void main() {
                ivec2 IDxy = ivec2(gl_GlobalInvocationID.xy);
            
                Camera camera;
                camera.field_of_view = rawCameraData.field_of_view;
                camera.aspectRatio = rawCameraData.aspectRatio;
                camera.fov_tan = rawCameraData.fov_tan;
                camera.camera_to_world = rawCameraData.raw_camera_to_world;
                camera.clear_color = rawCameraData.clear_color;
                
                ivec2 screenSize = imageSize(img_out);
                vec2 pixel_NCD = vec2((float(IDxy.x)+0.5)/float(screenSize.x), (float(IDxy.y)+0.5)/float(screenSize.y));
                vec2 camera_pixel = vec2((2 * pixel_NCD.x - 1) * camera.aspectRatio * camera.fov_tan, (1 - 2 * pixel_NCD.y) * camera.fov_tan);
            
                highp vec4 world_search_pos = vec4(vec3(camera_pixel.x, camera_pixel.y, -1.0), 0.0)*camera.camera_to_world;
                highp vec3 current_search_pos = normalize(world_search_pos.xyz);
                current_search_pos.x = -current_search_pos.x;
                vec4 color_in_the_end = camera.clear_color;
                
                while (length(current_search_pos) < rawCameraData.render_distance) {
                    ivec3 current_box = ivec3(floor(current_search_pos.x), floor(current_search_pos.y), floor(current_search_pos.z));
                    ColorHit check = voxel_hit(current_box, camera.clear_color);
                    if (check.hit) {
                        color_in_the_end = check.color;
                        break;
                    }
            
                    highp float multiplier = 999.0;
                    highp float x_mul = ceil(current_search_pos.x)/current_search_pos.x + 0.001;
                    highp float y_mul = ceil(current_search_pos.y)/current_search_pos.y + 0.001;
                    highp float z_mul = ceil(current_search_pos.z)/current_search_pos.z + 0.001;
                    if (x_mul < multiplier) multiplier = x_mul;
                    if (y_mul < multiplier) multiplier = y_mul;
                    if (z_mul < multiplier) multiplier = z_mul;
                    if (multiplier <= 1.0) multiplier = 1.01;
            
                    current_search_pos *= multiplier;
                }
            
                imageStore(img_out, IDxy, vec4(color_in_the_end.b, color_in_the_end.g, color_in_the_end.r, color_in_the_end.a));
            }
            "
        }
    }
    
    cs::load(device).unwrap()
}