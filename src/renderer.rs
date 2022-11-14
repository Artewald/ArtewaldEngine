use std::{time::Instant, f32::consts::PI};

use nalgebra::{Vector4, Vector3};
use utils::{setup_vulkan, create_main_shader, create_sets, create_render_image};
use vulkano::{pipeline::{ComputePipeline, Pipeline, PipelineBindPoint}, command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, CopyImageInfo, CopyBufferToImageInfo}, sync::{self, GpuFuture, FlushError}, image::{ImageAccess}, swapchain::{self, acquire_next_image, AcquireError}};
use winit::{event_loop::{EventLoop, ControlFlow}, event::{Event, WindowEvent, VirtualKeyCode, ElementState}, dpi::PhysicalPosition};

use crate::voxel::{VoxelData};

use self::utils::{create_voxel_buffer, create_camera_data_buffer, CameraData};

mod utils;

const PRINT_RENDER_INFO: bool = false;

pub fn setup_renderer_and_run(voxel_data: Vec<VoxelData>) {
    // Settings

    // Setup window and device
    let event_loop = EventLoop::new();
    
    let (mut vulkan_data, surface) = setup_vulkan(&event_loop);

    // Setup shaders, pipeline and buffers and descriptor sets
    let main_shader = create_main_shader(vulkan_data.device.clone());
    let compute_pipline = ComputePipeline::new(
        vulkan_data.device.clone(), 
        main_shader.entry_point("main").unwrap(), 
        &(), 
        None, 
        |_| {}
    ).unwrap();

    let camera_data_buffer = create_camera_data_buffer(CameraData::new(90, 1000.0, (surface.window().inner_size().width as f32)/(surface.window().inner_size().height as f32), Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 1.0, 0.0), Vector4::new(0.0, 0.0, 0.1884, 1.0), Vector3::new(-2.0, 0.0, 0.0)), vulkan_data.device.clone());
    let voxel_buffer = create_voxel_buffer(voxel_data, vulkan_data.device.clone());
    let mut render_image_data = create_render_image(&mut vulkan_data);

    let compute_pipeline_clone = compute_pipline.clone();
    let set_layouts = compute_pipeline_clone.layout().set_layouts();

    let mut sets = create_sets(set_layouts, voxel_buffer.clone(), camera_data_buffer.clone(), render_image_data.view.clone());


    // Main render loop
    let mut recreate_swapchain = false;
    let mut prev_frame_end = Some(sync::now(vulkan_data.device.clone()).boxed());
    
    let mut time = Instant::now();
    let mut delta_time = Instant::now();

    let mut window_focused = true;
    let mut yaw: f32 = 0.0;
    let mut pitch: f32 = 0.2;
    let mut look_target: Vector3<f32> = Vector3::new(0.0, 0.0, -1.0);
    let mut movement_input: Vector3<f32> = Vector3::new(0.0, 0.0, 0.0);
    
    let mouse_sensitivity = 1.0 / (50.0 * 50.0);
    let movement_speed = 0.01;

    event_loop.run(move |event, _, control_flow| {
        match event {

            Event::WindowEvent { event: WindowEvent::Focused(true), ..} => {
                window_focused = true;
                surface.window().set_cursor_visible(false);
            }

            Event::WindowEvent { event: WindowEvent::Focused(false), ..} => {
                window_focused = false;
                surface.window().set_cursor_visible(true);
            }

            Event::WindowEvent { event: WindowEvent::CursorMoved { position, ..}, .. } => {
                if window_focused {
                    let dim = surface.window().inner_size();
                    let center = PhysicalPosition::new(dim.width/2, dim.height/2);
                    let mouse_movement = PhysicalPosition::new(position.x - center.x as f64, center.y as f64 - position.y);

                    yaw -= (mouse_movement.x * mouse_sensitivity) as f32;
                    if yaw >= 2.0 * PI {
                        yaw = 0.0;
                    } else if yaw <= 0.0 {
                        yaw = 2.0 * PI;
                    }

                    pitch -= (mouse_movement.y * mouse_sensitivity) as f32;
                    if pitch >= 1.5 * PI {
                        pitch = 1.499999 * PI;
                    }
                    if pitch <= -PI/2.0 {
                        pitch = -PI/2.00001;
                    }
                    look_target = Vector3::new(pitch.cos() * yaw.sin(), -pitch.sin(), pitch.cos() * yaw.cos()).normalize();

                    surface.window().set_cursor_position(center).unwrap();
                }
            }

            Event::WindowEvent {event: WindowEvent::KeyboardInput {input, ..}, ..} => {
                if window_focused {
                    let button = input.virtual_keycode;
                    match button {
                        Some(btn) => {
                            let state = input.state;

                            if btn == VirtualKeyCode::Escape {
                                *control_flow = ControlFlow::Exit;
                            }
                            
                            if state == ElementState::Pressed {
                                if btn == VirtualKeyCode::W {
                                    movement_input.x = 1.0
                                } else if btn == VirtualKeyCode::S {
                                    movement_input.x = -1.0
                                } else if btn == VirtualKeyCode::Space {
                                    movement_input.y = 1.0
                                } else if btn == VirtualKeyCode::LControl {
                                    movement_input.y = -1.0
                                } else if btn == VirtualKeyCode::D {
                                    movement_input.z = 1.0
                                } else if btn == VirtualKeyCode::A {
                                    movement_input.z = -1.0
                                }
                            } else {
                                if btn == VirtualKeyCode::W {
                                    movement_input.x = 0.0
                                } else if btn == VirtualKeyCode::S {
                                    movement_input.x = 0.0
                                } else if btn == VirtualKeyCode::Space {
                                    movement_input.y = 0.0
                                } else if btn == VirtualKeyCode::LControl {
                                    movement_input.y = 0.0
                                } else if btn == VirtualKeyCode::D {
                                    movement_input.z = 0.0
                                } else if btn == VirtualKeyCode::A {
                                    movement_input.z = 0.0
                                }
                            }
                            
                        }
                        None => (),
                    }
                }
            }

            Event::WindowEvent {event: WindowEvent::CloseRequested, ..} => *control_flow = ControlFlow::Exit,

            Event::WindowEvent {event: WindowEvent::Resized(_), ..} => recreate_swapchain = true,
            
            Event::RedrawEventsCleared => {
                // Get the window dimentions
                let dim = surface.window().inner_size();
                if dim.width == 0 || dim.height == 0 {
                    return;
                }

                // This frees up some resources from time to time based on what the GPU has managed to do and not
                prev_frame_end.as_mut().unwrap().cleanup_finished();

                // Recreates the swapchain, decriptor-sets and the image that is rendered to.
                if recreate_swapchain {
                    utils::recreate_swapchain(&mut vulkan_data, dim);
                    render_image_data = create_render_image(&mut vulkan_data);
                    let compute_pipeline_cpy = compute_pipline.clone();
                    let new_set_layouts = compute_pipeline_cpy.layout().set_layouts();
                    camera_data_buffer.clone().write().unwrap().aspect_ratio = (dim.width as f32)/(dim.height as f32);
                    sets = create_sets(new_set_layouts, voxel_buffer.clone(), camera_data_buffer.clone(), render_image_data.view.clone());
                    recreate_swapchain = false;
                }

                // Gets the current available swapchain image that the rendered image can be copied to.
                // If the swapchain is suboptimal then it will be recreated later.
                let (img_index, suboptimal, _acquire_future) = match acquire_next_image(vulkan_data.swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    },
                    Err(e) => panic!("Failed to get the next image: {}", e),
                };

                if suboptimal {
                    recreate_swapchain = true;
                }

                // Building the command buffer and executing it.
                camera_data_buffer.clone().write().unwrap().camera_move(movement_input, look_target, delta_time.elapsed().as_millis() as f32, movement_speed);
                delta_time = Instant::now();
                camera_data_buffer.clone().write().unwrap().update_camera_dir(look_target, Vector3::new(0.0, 1.0, 0.0));
                
                let mut builder = AutoCommandBufferBuilder::primary(vulkan_data.device.clone(), vulkan_data.queue.queue_family_index(), CommandBufferUsage::OneTimeSubmit,).unwrap();
                builder.bind_pipeline_compute(compute_pipline.clone())
                       .bind_descriptor_sets(PipelineBindPoint::Compute, compute_pipline.clone().layout().clone(), 0, sets.clone())
                       .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(render_image_data.buffer.clone(), render_image_data.image.clone())).unwrap()
                       .dispatch([vulkan_data.images[0].dimensions().width() / 8, vulkan_data.images[0].dimensions().height() / 8, 1]).unwrap()
                       .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(render_image_data.image.clone(), render_image_data.buffer.clone())).unwrap()
                       .copy_image(CopyImageInfo::images(render_image_data.image.clone(), vulkan_data.images[img_index].clone())).unwrap();
                
                let command_buffer = builder.build().unwrap();
                let future = sync::now(vulkan_data.device.clone())
                                                                    .then_execute(vulkan_data.queue.clone(), command_buffer).unwrap()
                                                                    .then_swapchain_present(vulkan_data.queue.clone(), swapchain::PresentInfo::swapchain(vulkan_data.swapchain.clone()))
                                                                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        // Wait for the GPU to finish and then proceed
                        future.wait(None).unwrap();
                        prev_frame_end = Some(future.boxed())
                    },
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        prev_frame_end = Some(sync::now(vulkan_data.device.clone()).boxed());
                    }
                    Err(e) => panic!("Failed to flush future: {}", e),
                }

                if PRINT_RENDER_INFO {
                    println!("Render time: {}", time.elapsed().as_millis());
                    println!("Window in focus: {}", window_focused);
                    print!("\x1B[2J\x1B[1;1H");
                    time = Instant::now();
                }

                // Cleanup
                // movement_input.x = 0.0;
                // movement_input.y = 0.0;
                // movement_input.z = 0.0;
            },

            _ => (),
        }
    });
}