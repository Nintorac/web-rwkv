//! Minimal test of HIP functionality using null stream

#[cfg(feature = "hip")]
fn main() {
    use web_rwkv::hip::{self, DeviceBuffer, Stream};

    println!("Testing HIP with null stream...");

    let count = hip::get_device_count().expect("Failed to get device count");
    println!("Device count: {}", count);

    let name = hip::get_device_name(0).expect("Failed to get device name");
    println!("Device name: {}", name);

    // Use null stream
    println!("Creating null stream...");
    let stream = Stream::null();
    println!("Null stream created (handle is {:?})", stream.handle());

    // Allocate memory
    println!("Allocating device memory...");
    let mut d_buf = DeviceBuffer::<f32>::new(1024).expect("Failed to allocate");
    println!("Allocated {} elements", d_buf.len());

    // Copy to device
    let host_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    println!("Copying to device...");
    d_buf
        .copy_from_host(&host_data, &stream)
        .expect("Failed to copy to device");

    // Copy back
    let mut result = vec![0.0f32; 1024];
    println!("Copying from device...");
    d_buf
        .copy_to_host(&mut result, &stream)
        .expect("Failed to copy from device");

    // Sync
    println!("Synchronizing...");
    stream.synchronize().expect("Failed to sync");

    // Verify
    assert_eq!(host_data, result, "Data mismatch!");

    println!("All tests passed!");
}

#[cfg(not(feature = "hip"))]
fn main() {
    println!("HIP feature not enabled");
}
