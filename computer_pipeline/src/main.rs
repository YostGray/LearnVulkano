/// learn to draw perlin noise with vulkano computer pipeline

use std::sync::Arc;
use image::{Rgba, ImageBuffer};
use vulkano::{VulkanLibrary, instance::{Instance, InstanceCreateInfo}, device::{QueueFlags, DeviceCreateInfo, QueueCreateInfo, Device, DeviceExtensions}, memory::allocator::{StandardMemoryAllocator, AllocationCreateInfo, MemoryTypeFilter}, buffer::{Buffer, BufferCreateInfo, BufferUsage, BufferContents}, command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo}, CopyImageToBufferInfo}, pipeline::{PipelineShaderStageCreateInfo, PipelineLayout, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, compute::ComputePipelineCreateInfo, Pipeline, PipelineBindPoint}, descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet}, sync::{self, GpuFuture}, image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView}, format::Format};

struct State{
    device:Arc<vulkano::device::Device>,
    queue_family_index:u32,
    queue:Arc<vulkano::device::Queue>,
}

fn create_divce_and_queue() -> State{
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    
    let instance = Instance::new(
        library, 
        InstanceCreateInfo::default()
    )
    .expect("failed to create instance");

    let physical_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
        .expect("no devices available");

    // for family in physical_device.queue_family_properties() {
    //     println!("Found a queue family with {:?} queue(s)", family.queue_count);
    // }

    let queue_family_index = physical_device
        .queue_family_properties().iter().enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        })
        .expect("couldn't find a supported queue family") as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: DeviceExtensions {
                // khr_storage_buffer_storage_class: true,
                ..DeviceExtensions::empty()
            },
            ..Default::default()
        },
    )
    .expect("failed to create device");

    let queue = queues.next().unwrap();
    State{
        device,
        queue_family_index,
        queue,
    }
}

#[derive(BufferContents)]
#[repr(C)]
struct GenNoiseInput {
    seed: u32,
    size: u32,
    frequency: u32,
    fbm_time :u32,
}

mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "./res/shader.comp",
    }
}

fn gen_noise(state: &State){
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(state.device.clone()));
    
    let size = 512;

    let input_data = GenNoiseInput{
        seed: 929,
        size,
        frequency: 8,
        fbm_time: 4,
    };
    let input_buffer = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        input_data,
    )
    .expect("failed to create buffer");

    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [size, size, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC | ImageUsage::STORAGE,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .unwrap();
    let view = ImageView::new_default(image.clone()).unwrap();//image view

    let output_data: Vec<u8> = vec![0; (size * size * 4) as usize];
    let output_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        output_data,
    )
    .expect("failed to create destination buffer");

    let shader = cs::load(state.device.clone()).expect("failed to create shader module");
    
    let cs = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let pipeline_layout = PipelineLayout::new(
        state.device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(state.device.clone())
            .unwrap(),
    )
    .unwrap();

    let compute_pipeline = ComputePipeline::new(
        state.device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout),
    )
    .expect("failed to create compute pipeline");

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(state.device.clone(), Default::default());
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, input_buffer.clone()),
            WriteDescriptorSet::image_view(1, view.clone()),
        ], 
        [],
    )
    .unwrap();

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        state.device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );
    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        state.queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let work_group_counts = [size / 8, size / 8, 1];
    command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )
        .unwrap()
        .dispatch(work_group_counts)
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            output_buffer.clone(),
        ))    
        .unwrap();

    let command_buffer = command_buffer_builder.build().unwrap();
    let future = sync::now(state.device.clone())
        .then_execute(state.queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let buffer_content = output_buffer.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(size, size, &buffer_content[..]).unwrap();
    image.save("out/fbm_noise.png").unwrap();

    println!("Everything succeeded!");
}

fn main() {
    let state = create_divce_and_queue();
    gen_noise(&state);
}