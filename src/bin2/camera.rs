use nokhwa::{
    native_api_backend,
    pixel_format::{self, RgbFormat, LumaFormat},
    query,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution, CameraFormat, FrameFormat},
    Camera, camera_traits, Buffer,
};
use show_image::{create_window, ImageInfo, ImageView};
use image::{*, buffer::ConvertBuffer};
use imageproc::{rect::Rect, drawing::draw_hollow_rect_mut, definitions::Image};
use imageproc::drawing::draw_line_segment_mut;
use imageproc::drawing::draw_cross_mut;
use imageproc::drawing::draw_hollow_circle_mut;
use imageproc::drawing::draw_text_mut;
use nalgebra::Point2;
use ndarray_npy::ReadNpyExt;
use std::time::SystemTime;
use std::io::Cursor;
use rusttype::{Font, Scale};

use nng::{Aio, AioResult, Context, Protocol, Socket};


use nalgebra as na;

use std::ops::Deref;

use ndarray::s;
use ort::{
    environment::Environment, tensor::OrtOwnedTensor, ExecutionProvider, GraphOptimizationLevel,
    LoggingLevel, SessionBuilder, Value,
};

fn get_rect_area([x1, y1, x2, y2]: [f32; 4]
) -> f32 {
    (x2 - x1) * (y2 - y1)
}

fn get_rects_distance_center(
    [x01, y01, x02, y02]: [f32; 4],
    [x11, y11, x12, y12]: [f32; 4],
) -> f32 {
    let x1 = (x02 + x01) / 2.0;
    let y1 = (y02 + y01) / 2.0;
    let x2 = (x12 + x11) / 2.0;
    let y2 = (y12 + y11) / 2.0;
    ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
}

fn adjust_facebox(
    [x1, y1, x2, y2]: [f32; 4],
    container_width: f32,
    container_height: f32,
) -> [f32; 4] {
    // Calculate the width and height of the first rectangle
    let width = x2 - x1;
    let height = y2 - y1;

    // Calculate the right and bottom edges of both rectangles
    let rect_right = x2;
    let rect_bottom = y2;
    let container_right = container_width;
    let container_bottom = container_height;

    // Check if the first rectangle is fully within the second rectangle
    let mut adjusted_x1 = x1;
    let mut adjusted_y1 = y1;

    if rect_right > container_right {
        // Adjust x1 to fit within the container
        adjusted_x1 -= rect_right - container_right;
    }
    if rect_bottom > container_bottom {
        // Adjust y1 to fit within the container
        adjusted_y1 -= rect_bottom - container_bottom;
    }

    // Calculate the adjusted x2 and y2 based on the adjusted x1 and y1
    let adjusted_x2 = adjusted_x1 + width;
    let adjusted_y2 = adjusted_y1 + height;

    // Return the adjusted coordinates (x1, y1, x2, y2)
    [adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2]
}

use ndarray::{Array, CowArray, Array3, Ix3};
fn fit_circle(points: &[Point2<f32>]) -> (Point2<f32>, f32) {

    let x1 = points[0].x;
    let y1 = points[0].y;
    let x2 = points[1].x;
    let y2 = points[1].y;
    let x3 = points[2].x;
    let y3 = points[2].y;
    let x4 = points[3].x;
    let y4 = points[3].y;

    // Calculate the coordinates of the barycenter (mean)
    let x_m = (x1 + x2 + x3 + x4) / 4.0;
    let y_m = (y1 + y2 + y3 + y4) / 4.0;

    // Calculate the reduced coordinates
    let u1 = x1 - x_m;
    let u2 = x2 - x_m;
    let u3 = x3 - x_m;
    let u4 = x4 - x_m;
    let v1 = y1 - y_m;
    let v2 = y2 - y_m;
    let v3 = y3 - y_m;
    let v4 = y4 - y_m;

    // Calculate the sums needed for the linear system
    let suv = u1 * v1 + u2 * v2 + u3 * v3 + u4 * v4;
    let suu = u1 * u1 + u2 * u2 + u3 * u3 + u4 * u4;
    let svv = v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4;
    let suuv = u1 * u1 * v1 + u2 * u2 * v2 + u3 * u3 * v3 + u4 * u4 * v4;
    let suvv = u1 * v1 * v1 + u2 * v2 * v2 + u3 * v3 * v3 + u4 * v4 * v4;
    let suuu = u1 * u1 * u1 + u2 * u2 * u2 + u3 * u3 * u3 + u4 * u4 * u4;
    let svvv = v1 * v1 * v1 + v2 * v2 * v2 + v3 * v3 * v3 + v4 * v4 * v4;

    // Solve the linear system
    let a = suu;
    let b = suv;
    let c = suv;
    let d = svv;
    let e = 0.5 * (suuu + suvv);
    let f = 0.5 * (svvv + suuv);

    // Calculate the center coordinates in reduced coordinates
    let uc = (d * e - b * f) / (a * d - b * c);
    let vc = (a * f - c * e) / (a * d - b * c);

    // Calculate the center coordinates in original coordinates
    let xc_1 = x_m + uc;
    let yc_1 = y_m + vc;

    // Calculate the radii of each point to the center
    let ri1 = ((x1 - xc_1).powi(2) + (y1 - yc_1).powi(2)).sqrt();
    let ri2 = ((x2 - xc_1).powi(2) + (y2 - yc_1).powi(2)).sqrt();
    let ri3 = ((x3 - xc_1).powi(2) + (y3 - yc_1).powi(2)).sqrt();
    let ri4 = ((x4 - xc_1).powi(2) + (y4 - yc_1).powi(2)).sqrt();

    // Calculate the mean radius
    let r_1 = (ri1 + ri2 + ri3 + ri4) / 4.0;

    // Calculate the residual sum of squares
    let residu_1 = (ri1 - r_1).powi(2) + (ri2 - r_1).powi(2) + (ri3 - r_1).powi(2) + (ri4 - r_1).powi(2);

    // return the center coordinates and the radius
    (Point2::new(xc_1, yc_1), r_1)
}



fn convert_lum_image_buffer_to_rgb(image_buffer: ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut rgb_image_buffer = ImageBuffer::new(image_buffer.width(), image_buffer.height());
    for (x, y, pixel) in rgb_image_buffer.enumerate_pixels_mut() {
        let luma_pixel = image_buffer.get_pixel(x, y);
        *pixel = image::Rgb([luma_pixel[0], luma_pixel[0], luma_pixel[0]]);
    }
    rgb_image_buffer
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn draw_hollow_polygon_mut(
    canvas: &mut RgbImage,
    polygon: &[Point2<f32>],
    color: Rgb<u8>,
) {
    // draw lines between all points
    for i in 0..polygon.len() - 1 {
        draw_line_segment_mut(canvas, (polygon[i].x, polygon[i].y), (polygon[i + 1].x, polygon[i + 1].y), color);
    }
}


fn argmax(input: Vec<f32>) -> usize {
    let mut max = 0.0;
    let mut max_index = 0;
    for (i, &item) in input.iter().enumerate() {
        if item > max {
            max = item;
            max_index = i;
        }
    }
    max_index
}

use ndarray::{Array2, Axis};

fn decode_boxes(
    raw_boxes: &Array3<f32>,
    anchors: &Array2<f32>,
    x_scale: f32,
    y_scale: f32,
    w_scale: f32,
    h_scale: f32,
) -> Array3<f32> {
    let shape = raw_boxes.shape();
    let num_boxes = shape[1];

    let mut boxes = Array3::zeros(Ix3(shape[0], shape[1], shape[2]));

    for i in 0..num_boxes {
        let x_center =
            &raw_boxes[[0, i, 0]] / x_scale * &anchors[[i, 2]] + &anchors[[i, 0]];
        let y_center =
            &raw_boxes[[0, i, 1]] / y_scale * &anchors[[i, 3]] + &anchors[[i, 1]];

        let w = &raw_boxes[[0, i, 2]] / w_scale * &anchors[[i, 2]];
        let h = &raw_boxes[[0, i, 3]] / h_scale * &anchors[[i, 3]];

        boxes[[0, i, 0]] = y_center - h / 2.0; // ymin
        boxes[[0, i, 1]] = x_center - w / 2.0; // xmin
        boxes[[0, i, 2]] = y_center + h / 2.0; // ymax
        boxes[[0, i, 3]] = x_center + w / 2.0; // xmax

        for k in 0..6 {
            let offset = 4 + k * 2;
            let keypoint_x =
                &raw_boxes[[0, i, offset]] / x_scale * &anchors[[i, 2]] + &anchors[[i, 0]];
            let keypoint_y =
                &raw_boxes[[0, i, offset + 1]] / y_scale * &anchors[[i, 3]] + &anchors[[i, 1]];

            boxes[[0, i, offset]] = keypoint_x;
            boxes[[0, i, offset + 1]] = keypoint_y;
        }
    }

    boxes
}



fn convert_nv12_to_grayscale(src_data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let src_data_new = {
        // step through src_data and take every 2nd element, starting at 1
        // create new vec with given size
        let mut src_data_new = vec![0u8; (src_data.len() / 2) as usize];
        for i in (1..src_data.len()).step_by(2) {
            src_data_new[i / 2] = src_data[i];
        }
        src_data_new
    };

    src_data_new
}

fn ngg_callback(aio: Aio, ctx: &Context, res: AioResult) {
    // This is the callback that will be called when we receive a message
    match res {
         // We successfully received a message.
         AioResult::Recv(m) => {
            let msg = m.unwrap();
            println!("Worker received: {}", String::from_utf8_lossy(&msg[..]));
            ctx.recv(&aio).unwrap();
        }
        // We successfully sent a message.
        AioResult::Send(m) => {
            println!("Worker sent message");
        }
        // We are sleeping.
        AioResult::Sleep(r) => {
            println!("Worker sleeping");
        } 

    }
}

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    let mut last_frame = std::time::Instant::now();

    let mut current_frame_i = 0;

    // where to store the frames
    let mut frame_path = r"D:\frames";
    let mut is_recording = false;

    // create ringbuffer for frame deltas
    let mut frame_deltas: Vec<std::time::Duration> = vec![std::time::Duration::from_millis(0); 1];

    // load a font for drawing text (use system default)
    let font: &[u8] = include_bytes!("../../Roboto-Regular.ttf");
    let font = Font::try_from_bytes(font).unwrap();


    // get backend
    let backend = native_api_backend().unwrap();

    // print backend info
    println!("Backend: {}", backend);

    let backend = native_api_backend().unwrap();
    let devices = query(backend).unwrap();
    println!("There are {} available cameras.", devices.len());
    for device in devices {
        println!("{device}", device = device);
        println!("{device}", device = device);
    }

    // select camera
    let index = CameraIndex::Index(0);

    // query camera

    let fps = 120;
    let fourcc = FrameFormat::NV12;

    let resolution = Resolution::new(1280, 720);
    let camera_format = CameraFormat::new(resolution, fourcc, fps);

    let request_frame_formats = vec![FrameFormat::NV12];
 
 
    let requested = RequestedFormat::with_formats(
        RequestedFormatType::AbsoluteHighestFrameRate,
        &request_frame_formats,
    );

    println!("Opening camera {} with format {:?}", index, requested);

    // make the camera
    let mut camera = Camera::new(index, requested).unwrap();

    // get resolution
    let resolution = camera.resolution();
    println!("Resolution: {} \n", resolution);
    let frame_format = camera.frame_format();
    println!("Frame format: {:?} \n", frame_format);
    let fps = camera.frame_rate();
    println!("FPS: {} \n", fps);

    // find supported formats
    let compatible_fourcc = camera.compatible_fourcc().unwrap();

    for fourcc in compatible_fourcc {
        println!("Compatible fourcc: {:?}:  \n", fourcc);
        let supported_formats = camera.compatible_list_by_resolution(fourcc);

        for format in supported_formats {
            println!("Supported modes: {:?} \n", format);
        }
    }
    

    // open the camera
    println!("Opening stram...");
    camera.open_stream();
    println!("Opened camera");

    // create a window
    let window = create_window("image", Default::default())?;

    let mut face_box_valid = false;
    let mut face_box: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

    // create buffer holding the last 100 pupil positions
    let mut pupil_positions: Vec<[f32; 2]> = vec![[0.0, 0.0]; 100];

    let mode = "facetime_hd";

    // get the frame
    loop {

       let mut start = SystemTime::now();
       
       let mut greyscale_img: ImageBuffer<Luma<u8>, Vec<u8>>;

        if mode == "facetime_hd"
        {

            let frame: std::borrow::Cow<'_, [u8]> = camera.frame_raw().unwrap();

            // get raw image as vector
            let data = frame.to_vec();

            start = SystemTime::now();

            // convert the broken NV12 format to grayscale (hint: its not nv12)
            let greyscale_vec: Vec<u8> =
                convert_nv12_to_grayscale(&data, resolution.width_x, resolution.height_y);

            // create greyscale imagebuffer
            greyscale_img = ImageBuffer::from_vec(
                resolution.width_x,
                resolution.height_y,
                Clone::clone(&greyscale_vec),            
            )
            .unwrap();
        } else {
            let frame = camera.frame().unwrap();

            // print length of buffer
            println!("Buffer length: {}", frame.buffer().len());

            let frame = Buffer::new(frame.resolution(), frame.buffer(), FrameFormat::YUYV);

    
            // // get raw image as imagebuffer
            greyscale_img = frame.decode_image::<LumaFormat>().unwrap();
          
            // // convert to grayscale
            // greyscale_img = rgb_img;
            
        }

                // plot fps (1s divided by average frame delta) (if there are less than 5 frames, fps is 0)
        let fps = if frame_deltas.len() < 5 {
            0.0
        } else {
            1.0 / (frame_deltas[2..].iter().sum::<std::time::Duration>().as_secs_f32() / frame_deltas.len() as f32)
        };

        draw_text_mut(
            &mut greyscale_img,
            image::Luma::<u8>([255u8]),
            0,
            0,
            Scale { x: 20.0, y: 20.0 },
            &font,
            &format!("FPS: {:.2}", fps),
        );

        // use image_show to display the frame (every 10th frame)
        if current_frame_i % 20 == 0 {
            window.set_image("image", greyscale_img)?;
        }

        // increment frame counter
        current_frame_i += 1;
        

        // calculate frame delta
        let frame_delta = last_frame.elapsed();
        last_frame = std::time::Instant::now();

        // push frame delta to ringbuffer
        frame_deltas.push(frame_delta);

        // remove first element if ringbuffer is full
        if frame_deltas.len() > 100 {
            frame_deltas.remove(0);
        }

        let end = SystemTime::now();

        // get inference time
        let inference_time = end.duration_since(start).unwrap();
        println!("Frame processed in {}ms", inference_time.as_millis());
    }

    // return
    Ok(())
}
