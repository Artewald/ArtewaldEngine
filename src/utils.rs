use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage, view};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorType};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::format::Format;
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::{ImageUsage, SwapchainImage, ImageAccess, ImageSubresourceRange, ImageAspects, ImageViewType, ImageViewAbstract, StorageImage, ImageDimensions};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::library::VulkanLibrary;
use vulkano::memory::pool::{PotentialDedicatedAllocation, StandardMemoryPool, StandardMemoryPoolAlloc};
use vulkano::pipeline::graphics::render_pass;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{RenderPass, Framebuffer, FramebufferCreateInfo};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::display::DisplayMode;
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo, PresentMode, SwapchainCreationError, acquire_next_image, self};
use vulkano::sync::PipelineStage;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, Queue, DeviceExtensions, self};
use vulkano_win::VkSurfaceBuild;
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoop;
use winit::window::{WindowBuilder, Window};

use std::sync::Arc;

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

        // let s_f = Some(device.physical_device().surface_formats(&surface, Default::default()).unwrap()).unwrap();
        // for x in s_f {
        //     println!("{}", x.0 as i32);
        // }

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

pub fn create_main_shader(device: Arc<Device>) -> Arc<ShaderModule> {

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
            #version 450

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
            
            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } data;
            
            layout(set = 1, binding = 0, rgba8) uniform writeonly image2D img_out; 
            
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                data.data[idx] *= 12;
            
                vec4 color_in_the_end = vec4(0.0, 0.0, 1.0, 1.0);
                ivec2 IDxy = ivec2(gl_GlobalInvocationID.xy);
                imageStore(img_out, IDxy, vec4(color_in_the_end.b, color_in_the_end.g, color_in_the_end.r, color_in_the_end.a));
            }
            "
        }
    }
    
    cs::load(device).unwrap()
}

pub fn create_buffer(device: Arc<Device>) -> Arc<CpuAccessibleBuffer<[u32], PotentialDedicatedAllocation<StandardMemoryPoolAlloc>>> {
    let data_iter = 0..65536u32;
    CpuAccessibleBuffer::from_iter(device, BufferUsage {storage_buffer: true, ..BufferUsage::empty()}, false, data_iter).unwrap()
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

pub fn create_sets(set_layouts: &[Arc<DescriptorSetLayout>], buffer: Arc<CpuAccessibleBuffer<[u32]>>, img_view: Arc<dyn ImageViewAbstract>) -> Vec<Arc<PersistentDescriptorSet>> {
    let mut sets = vec![];

    //println!("{}", img_view.format().unwrap() as u32);

    for set_layout in set_layouts {
        for x in set_layout.bindings() {
            if x.1.descriptor_type == DescriptorType::StorageBuffer {
                sets.push(PersistentDescriptorSet::new(set_layout.clone(), [WriteDescriptorSet::buffer(0, buffer.clone())]).unwrap());
            } else if x.1.descriptor_type == DescriptorType::StorageImage {
                sets.push(PersistentDescriptorSet::new(set_layout.clone(), [WriteDescriptorSet::image_view(0, img_view.clone())]).unwrap())
            } else {
                panic!("There exists an unused descriptorset, it should be implemented!");
            }
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