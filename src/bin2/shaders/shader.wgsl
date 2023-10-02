struct GratingConfigUniform {
    freq: f32,
    phase: f32,
    contrast: f32,
    orientation: f32
}

@group(0) @binding(0) // 1.
var<uniform> grating_config: GratingConfigUniform;

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    // draw full screen using two triangles
    let x = f32(i32(in_vertex_index & 1u) * 4 - 1);
    let y = f32(i32(in_vertex_index >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) in_position: vec4<f32>) -> @location(0) vec4<f32> {
    // get the x and y coordinates of the fragment
    let x = in_position.x;

    // Calculate the sine wave pattern
    let frequency: f32 = grating_config.freq; // Adjust the frequency as needed
    let phase: f32 = grating_config.phase; // Adjust the phase as needed
    let sine_wave = sin(frequency * x + phase);
    
    // Create the final color based on the sine wave value
    let color = vec3<f32>(sine_wave, sine_wave, sine_wave) * grating_config.contrast;

    return vec4<f32>(color.rgb, 1.0); // Use color.rgb to extract the RGB channels and set alpha to 1.0
}
