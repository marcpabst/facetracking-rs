#![feature(portable_simd)]
extern crate rust_eyetracking;

use std::simd::SimdFloat;
use std::simd::f32x4;
use nokhwa::pixel_format::LumaFormat;
use rayon::prelude::*;

use dcv_color_primitives as dcp;
use dcp::ColorSpace;
use dcp::ImageFormat;
use dcp::PixelFormat;
use dcp::convert_image;
use egui::plot::{Line, Plot, PlotPoints};
use egui_gizmo::Gizmo;
use egui_gizmo::GizmoMode;
use image::{DynamicImage, ImageBuffer, Luma};
use imageproc::drawing::draw_cross_mut;
use imageproc::drawing::draw_hollow_rect_mut;
use nalgebra as na;
use nokhwa::pixel_format::RgbAFormat;
use nokhwa::Buffer;
use nokhwa::CallbackCamera;
use nokhwa::{
    native_api_backend,
    pixel_format::RgbFormat,
    query,
    utils::{
        CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
    },
    Camera,
};
use re_ui::ReUi;

use rust_eyetracking::face_detection::model_blazeface::BlazefaceModel;
use rust_eyetracking::face_detection::FaceBoundingBox;
use rust_eyetracking::face_detection::FaceDetectionModel;

use rust_eyetracking::face_landmarks::model_mediapipe::MediapipeFaceLandmarksModel;
use rust_eyetracking::face_landmarks::FaceLandmarksModel;
use rust_eyetracking::face_landmarks::ScreenFaceLandmarks;
use show_image::AsImageView;

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use std::collections::VecDeque;

pub struct TimeSeries {
    data: VecDeque<f32>,
    timestamp: VecDeque<u128>,
    max_length: usize,
}

impl TimeSeries {
    fn new(max_length: usize) -> Self {
        Self {
            data: VecDeque::new(),
            timestamp: VecDeque::new(),
            max_length,
        }
    }

    fn push(&mut self, value: f32, timestamp: u128) {
        self.data.push_back(value);
        self.timestamp.push_back(timestamp);

        if self.data.len() > self.max_length {
            self.data.pop_front();
            self.timestamp.pop_front();
        }
    }

    fn get_mean(&self) -> f32 {
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }
}

// implement clone for TimeSeries
impl Clone for TimeSeries {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            timestamp: self.timestamp.clone(),
            max_length: self.max_length,
        }
    }
}

use nng::{Aio, AioResult, Context, Protocol, Socket};

const ADDRESS: &'static str = "tcp://127.0.0.1:54321";

fn server(shared_state: SharedState) -> Result<(), nng::Error> {
    // check if there is a shared state

    // Set up the server and listen for connections on the specified address.
    let s = Socket::new(Protocol::Rep0)?;

    // Set up the callback that will be called when the server receives a message
    // from the client.
    let ctx = Context::new(&s)?;
    let ctx_clone = ctx.clone();
    let aio =
        Aio::new(move |aio, res| worker_callback(aio, &ctx_clone, res, shared_state.clone()))?;

    s.listen(ADDRESS)?;

    // start the worker thread
    // subscribe to messages

    ctx.recv(&aio)?;

    println!("Server listening on {}", ADDRESS);

    loop {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

fn worker_callback(aio: Aio, ctx: &Context, res: AioResult, shared_state: SharedState) {
    // This is the callback that will be called when the worker receives a message
    // from the client. This fnction just prints the message.
    match res {
        // We successfully received a message.
        AioResult::Recv(m) => {
            let msg = m.unwrap();

            println!("Worker received message");

            // if message is "record PATH" start recording
            let msg_str = std::str::from_utf8(&msg).unwrap();
            if msg_str.starts_with("record") {
                let path = msg_str.split(" ").collect::<Vec<&str>>()[1];
                let mut guard = shared_state.lock().unwrap();
                guard.is_recording = Some(true);
                guard.recording_path = Some(path.to_string());
                std::mem::drop(guard);
            } else if msg_str.starts_with("stop") {
                let mut guard = shared_state.lock().unwrap();
                guard.is_recording = Some(false);
                std::mem::drop(guard);
            }

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

// #![warn(clippy::all, rust_2018_idioms)]
// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

fn convert_nv12_to_grayscale(src_data: &[u8], width: u32, height: u32) -> Vec<u8> {
    // let src_data_new = {
    //     // step through src_data and take every 2nd element, starting at 1
    //     // create new vec with given size
    //     let mut src_data_new = vec![0u8; (src_data.len() / 2) as usize];
    //     for i in (1..src_data.len()).step_by(2) {
    //         src_data_new[i / 2] = src_data[i];
    //     }
    //     src_data_new
    // };

  

    let mut dst_data = vec![255u8; (width * height * 3) as usize];

    yuv422_to_rgb24(
        &src_data,
        &mut dst_data,
    );

    dst_data

}

// make State an alias for a Mutex protected struct
type SharedState = Arc<Mutex<State>>;

pub struct State {
    // the data that will be shared between the threads
    pub fps: Option<f32>,
    pub fps_vec: Vec<f32>,
    pub last_frame_time: Option<SystemTime>,
    pub resolution: Option<(u32, u32)>,
    pub image_valid: Option<bool>,
    pub image: Option<DynamicImage>,
    pub devices: Option<Vec<String>>,
    pub current_device: Option<u32>,
    pub is_recording: Option<bool>,
    pub recording_path: Option<String>,

    pub face_bbox: Option<(u32, u32, u32, u32)>,
    pub screen_face_landmarks: Option<Arc<dyn ScreenFaceLandmarks>>,

    // time series
    pub left_eye_dist_ts: Option<TimeSeries>,
    pub right_eye_dist_ts: Option<TimeSeries>,
    pub left_iris_x_ts: Option<TimeSeries>,
    pub left_iris_y_ts: Option<TimeSeries>,
}

// by default, all fields are None
impl Default for State {
    fn default() -> Self {
        Self {
            fps: None,
            fps_vec: Vec::new(),
            last_frame_time: None,
            resolution: None,
            image_valid: None,
            image: None,
            devices: None,
            current_device: None,
            is_recording: None,
            recording_path: None,

            face_bbox: None,
            screen_face_landmarks: None,

            left_eye_dist_ts: None,
            right_eye_dist_ts: None,
            left_iris_x_ts: None,
            left_iris_y_ts: None,
        }
    }
}

// function that runs as a thread and performs the actual work
// function should accept a shared state as an argument
fn worker_thread(shared_state: Arc<Mutex<State>>) {
    loop {
        // get backend
        let backend = native_api_backend().unwrap();

        // print backend info
        println!("Backend: {}", backend);

        let backend = native_api_backend().unwrap();
        let devices = query(backend).unwrap();
        println!("There are {} available cameras.", devices.len());

        // set the shared state devices
        shared_state.lock().unwrap().devices =
            Some(devices.iter().map(|d| d.human_name()).collect());

        // set up face landmarks model
        let face_detection_model = BlazefaceModel::new();
        let mut face_landmarks_model = MediapipeFaceLandmarksModel::new();

        // select camera
        let index_i = shared_state
            .lock()
            .unwrap()
            .current_device
            .unwrap_or(0);

        let index = CameraIndex::Index(index_i);

        let fps = 30;
        let fourcc = FrameFormat::NV12;

        let resolution = Resolution::new(1280, 720);

        // set resolution in shared state
        shared_state.lock().unwrap().resolution = Some((resolution.width_x, resolution.height_y));

        let camera_format = CameraFormat::new(resolution, fourcc, fps);

        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(camera_format));
        println!("Opening camera {} with format {:?}", index, requested);

        // // get resolution
        // let resolution = camera.resolution();
        // println!("Resolution: {} \n", resolution);
        // let frame_format = camera.frame_format();
        // println!("Frame format: {:?} \n", frame_format);
        // let fps = camera.frame_rate();
        // println!("FPS: {} \n", fps);

        // create ringbuffer for frame deltas
        let mut fps_vec: Vec<f32> = Vec::new();

        // set last frame time
        shared_state.lock().unwrap().last_frame_time = Some(SystemTime::now());

        let mut threaded = CallbackCamera::new(index.clone(), requested, move |buffer| {
            let last_frame_time = shared_state.lock().unwrap().last_frame_time.unwrap();
            let mut fps_vec = shared_state.lock().unwrap().fps_vec.clone();

            //convert to rgb format
            let frame = buffer.buffer();

            let frame = convert_nv12_to_grayscale(&frame, resolution.width_x, resolution.height_y);
            let image_buffer = ImageBuffer::from_vec(resolution.width_x, resolution.height_y, frame).unwrap();

            // // decode image
            // let buffer = Buffer::new(buffer.resolution(), buffer.buffer(), FrameFormat::NV12);
            // let image_buffer = buffer.decode_image::<RgbFormat>().unwrap();

            
            let image = DynamicImage::ImageRgb8(image_buffer);

            // flip image
            let image = image.fliph();
            
            if (shared_state.lock().unwrap().screen_face_landmarks.is_none() || shared_state.lock().unwrap().screen_face_landmarks.clone().unwrap().get_confidence() < 0.5) {
                
                let face_bbox = face_detection_model.run(&image);
                // set face bbox
                face_landmarks_model.set_face_bbox(&*face_bbox);
            }

            // run through mediapipe model
            let face_landmarks = face_landmarks_model.run(&image);

            
            let ff = face_landmarks.to_metric();

            // print the first point of ff
            let metric_points = ff.get_metric_landmarks();




            // set image
            shared_state.lock().unwrap().image = Some(image.clone());

            // set face bbox
            shared_state.lock().unwrap().face_bbox = Some(face_landmarks.get_face_bbox());

            // set landmarks
            shared_state.lock().unwrap().screen_face_landmarks = Some(face_landmarks.into());

            let frame_delta = last_frame_time.elapsed().unwrap();
            let frame_delta = frame_delta.as_millis() as f32 / 1000.0;
       
            // calculate fps
            let fps = 1.0 / frame_delta;

            // push to vector
            fps_vec.push(fps);

            // remove first element if vector is longer than 25
            if fps_vec.len() > 10 {
                fps_vec.remove(0);
            }

            // set fps
            shared_state.lock().unwrap().fps =
                Some((fps_vec.iter().sum::<f32>() / fps_vec.len() as f32).round());
            shared_state.lock().unwrap().last_frame_time = Some(SystemTime::now());
            shared_state.lock().unwrap().fps_vec = fps_vec;
        })
        .unwrap();


        threaded.open_stream().unwrap();

        #[allow(clippy::empty_loop)] // keep it running
        loop {
            // wait for 1ms
            std::thread::sleep(Duration::from_millis(1));
            continue;
            // if macos
            #[cfg(target_os = "macos")]
            {

                //         let frame = threaded.poll_frame();

                //         if frame.is_err() {
                //             println!("Frame error");
                //             continue;
                //         }

                //         let frame = frame.unwrap();

                //         // set frame format manually to YUYV
                //         let frame = Buffer::new(frame.resolution(), frame.buffer(), FrameFormat::YUYV);

                //         // convert to rgb format
                //         let frame = frame.decode_image::<RgbAFormat>().unwrap();

                //         // get timestamp
                //         let timestamp = std::time::SystemTime::now()
                //             .duration_since(std::time::UNIX_EPOCH)
                //             .unwrap()
                //             .as_micros();

                //         // get raw image as vector
                //         let data = frame.to_vec();

                //         // convert the broken NV12 format to grayscale (hint: its not nv12)
                //         let greyscale_vec: Vec<u8> =
                //             convert_nv12_to_grayscale(&data, resolution.width_x, resolution.height_y);

                //         // create greyscale imagebuffer
                //         let image = DynamicImage::ImageLuma8(
                //             ImageBuffer::from_vec(resolution.width_x, resolution.height_y, greyscale_vec)
                //                 .unwrap(),
                //         );

                //         // convert to rgb
                //         let image_buffer = image.to_rgb8();
                //         let image = DynamicImage::ImageRgb8(image_buffer);

                //         // flip image
                //         let image = image.fliph();

                //         // // runt through blazeface model
                //         // let face_bbox = face_detection_model.run(&image);

                //         // // set face bbox
                //         // face_landmarks_model.set_face_bbox(&*face_bbox);

                //         // // run through mediapipe model
                //         // let face_landmarks = face_landmarks_model.run(&image);

                //         // set image
                //         shared_state.lock().unwrap().image = Some(image.clone());
                //         // set face bbox
                //         // shared_state.lock().unwrap().face_bbox = Some(face_bbox.to_tuple());
                //         // // set landmarks
                //         // shared_state.lock().unwrap().screen_face_landmarks = Some(face_landmarks.into());

                //     let frame_delta = last_frame_time.elapsed().unwrap();
                //     let frame_delta = frame_delta.as_millis() as f32 / 1000.0;

                //     // calculate fps
                //     let fps = 1.0 / frame_delta;

                //     // push to vector
                //     fps_vec.push(fps);

                //     // remove first element if vector is longer than 25
                //     if fps_vec.len() > 10 {
                //         fps_vec.remove(0);
                //     }

                //     // check if recording is enabled
                //     if shared_state
                //         .lock()
                //         .unwrap()
                //         .is_recording
                //         .unwrap_or_default()
                //     {
                //         println!("Recording frame");

                //         // // get current timestamp
                //         // let timestamp = std::time::SystemTime::now()
                //         //     .duration_since(std::time::UNIX_EPOCH)
                //         //     .unwrap()
                //         //     .as_millis();

                //         // // write frame to disk
                //         // let frame_path = &shared_state.lock().unwrap().recording_path;
                //         // let filename = format!("{}/frame_{}.png", frame_path, timestamp);

                //         // // save greyscale_img image as jpg
                //         // shared_state.lock().unwrap().image.save(filename).unwrap();
                //     }

                //     // update the shared state with the mean fps
                //     shared_state.lock().unwrap().fps =
                //         Some((fps_vec.iter().sum::<f32>() / fps_vec.len() as f32).round());

                //     if shared_state
                //         .lock()
                //         .unwrap()
                //         .current_device
                //         .unwrap_or_default()
                //         != index_i
                //     {
                //         // break out of the loop
                //         println!(
                //             "Device changed from {} to {}",
                //             shared_state
                //                 .lock()
                //                 .unwrap()
                //                 .current_device
                //                 .unwrap_or_default(),
                //             index_i
                //         );
                //         break;
                //     }

                //     last_frame_time = SystemTime::now();
            }
        }
    }
}

// When compiling natively:
fn main() -> eframe::Result<()> {
    dcp::initialize();

    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    // create rgba dummy image
    let vec: Vec<u8> = vec![0u8; 1280 * 720 * 3];

    // create new shared state
    let shared_state: SharedState = Arc::new(Mutex::new(State::default()));

    // create a thread and run the worker function in it
    let thread_data = Arc::clone(&shared_state);
    std::thread::spawn(|| worker_thread(thread_data));

    // // start the nng server
    // let thread_data2 = Arc::clone(&shared_state);
    // std::thread::spawn(|| server(thread_data2));

    // start the gui
    let gui_data = Arc::clone(&shared_state);

    let native_options = eframe::NativeOptions::default();

    eframe::run_native(
        "Camera Recorder v0.0.2",
        native_options,
        Box::new(move |cc| {
            let _re_ui = re_ui::ReUi::load_and_apply(&cc.egui_ctx);
            Box::new(TemplateApp::new(cc, gui_data))
        }),
    )
}

pub struct TemplateApp {
    // this how you opt-out of serialization of a member
    saturation: f32,
    brightness: f32,
    contrast: f32,
    shared_state: SharedState,
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>, shared_state: SharedState) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        Self {
            // Example stuff:
            saturation: 1.0,
            brightness: 1.0,
            contrast: 1.0,
            shared_state,
        }
    }
}

impl eframe::App for TemplateApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let Self {
            saturation,
            brightness,
            contrast,
            shared_state,
        } = self;

        // Examples of how to create different panels and windows.
        // Pick whichever suits you.
        // Tip: a good default choice is to just keep the `CentralPanel`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        #[cfg(not(target_arch = "wasm32"))] // no File->Quit on web pages!
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Quit").clicked() {
                        _frame.close();
                    }
                });
            });
        });

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            // add scrollable area

            egui::ScrollArea::vertical().show(ui, |ui| {
                // dropdown with devices;

                // let devices = shared_state.lock().unwrap().devices.clone().unwrap_or_default();
                // let mut selected_device = shared_state
                //     .lock()
                //     .unwrap()
                //     .current_device
                //     .unwrap_or_default() as usize;

                // egui::ComboBox::from_label("Camera")
                //     .selected_text(format!("{:?}", devices[selected_device]))
                //     .show_ui(ui, |ui| {
                //         for (i, device) in devices.iter().enumerate() {
                //             ui.selectable_value(&mut selected_device, i as usize, device);
                //         }
                //     });

                // // set mutex
                // shared_state.lock().unwrap().current_device = Some(selected_device as u32);

                // ui.add(egui::Slider::new(saturation, 0.0..=10.0).text("Saturation"));
                // ui.add(egui::Slider::new(brightness, 0.0..=10.0).text("Brightness"));
                // ui.add(egui::Slider::new(contrast, 0.0..=10.0).text("Contrast"));
                // add text (from shared state)
                let fps = shared_state.lock().unwrap().fps.unwrap_or(0.0);
                let resolution = shared_state.lock().unwrap().resolution.unwrap_or((0, 0));

                ui.add(egui::Label::new(format!("FPS: {}", fps)));
                ui.add(egui::Label::new(format!(
                    "Resolution: {}x{}",
                    resolution.0, resolution.1
                )));

                ui.add(egui::Label::new(format!(
                    "Recording: {}",
                    shared_state
                        .lock()
                        .unwrap()
                        .is_recording
                        .unwrap_or_default()
                )));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // check if image is valid

            if shared_state.lock().unwrap().image.clone().is_some()
                && shared_state.lock().unwrap().face_bbox.clone().is_some()
            {
                let mut image = shared_state.lock().unwrap().image.clone().unwrap();
                let face_bbox = shared_state.lock().unwrap().face_bbox.clone().unwrap();

                // draw face box
                let green = image::Rgba([0u8, 255u8, 0u8, 255u8]);
                let rect: imageproc::rect::Rect =
                    imageproc::rect::Rect::at(face_bbox.0 as i32, face_bbox.1 as i32)
                        .of_size(face_bbox.2, face_bbox.3);

                draw_hollow_rect_mut(&mut image, rect, green);

                // draw all landmarks
                let face_landmarks = shared_state
                    .lock()
                    .unwrap()
                    .screen_face_landmarks
                    .clone()
                    .unwrap();

                let landmarks = face_landmarks.get_landmarks();

                for landmark in landmarks.iter() {
                    draw_cross_mut(&mut image, green, landmark.x as i32, landmark.y as i32);
                }

                let imgage_size = [image.width() as usize, image.height() as usize];

                let ui_image = egui::ColorImage::from_rgb(
                    imgage_size,
                    image.as_rgb8().unwrap().as_raw().as_slice(),
                );
                let texture_hdl =
                    ctx.load_texture("image", ui_image, egui::TextureOptions::default());

                // add image and set size to panel width
                let ui_img_width = ui.available_width();
                let ui_img_height = ui_img_width / (image.width() as f32 / image.height() as f32);

                ui.image(&texture_hdl, egui::Vec2::new(ui_img_width, ui_img_height));

                // // aquire the mutex
                // let guard = shared_state.lock().unwrap();
                // // clone the image
                // let mut image = guard.image.clone();
                // let face_landmarks = guard.face_landmarks.clone();
                // let face_box = guard.face_box.clone();
                // let image_valid = guard.image_valid.clone();
                // let rotation_matrix = guard.rotation_matrix.clone();
                // let left_eye_dist_ts = guard.left_eye_dist_ts.clone();
                // let right_eye_dist_ts = guard.right_eye_dist_ts.clone();
                // let left_iris_x_ts = guard.left_iris_x_ts.clone();
                // let left_iris_y_ts = guard.left_iris_y_ts.clone();
                // // unlock the mutex
                // std::mem::drop(guard);

                // // check if image is valid
                // if image_valid {
                //     // flip the image
                //     let mut image = image.fliph();

                //     let green = image::Rgba([0u8, 255u8, 0u8, 255u8]);
                //     let red = image::Rgba([255u8, 0u8, 0u8, 255u8]);

                //     let imgage_size = [image.width() as usize, image.height() as usize];

                //     // draw face box
                //     let face_box_px: [f32; 4] = face_box
                //         .fliph()
                //         .to_pixel_space(imgage_size[0] as u32, imgage_size[1] as u32);
                //     let rect = imageproc::rect::Rect::at(face_box_px[0] as i32, face_box_px[1] as i32)
                //         .of_size(
                //             (face_box_px[2] - face_box_px[0]) as u32,
                //             (face_box_px[3] - face_box_px[1]) as u32,
                //         );

                //     draw_hollow_rect_mut(&mut image, rect, green);

                //     // plot face landmarks
                //     for p in face_landmarks.points.iter() {
                //         // convert to image coords
                //         let x = (1.0 - p.x) * (face_box_px[2] - face_box_px[0]) + face_box_px[0];
                //         let y = p.y * (face_box_px[3] - face_box_px[1]) + face_box_px[1];

                //         draw_cross_mut(&mut image, green, x as i32, y as i32);
                //     }

                //     let panel_width = ui.available_width();

                //     let ui_image = egui::ColorImage::from_rgb(
                //         imgage_size,
                //         image.as_rgb8().unwrap().as_raw().as_slice(),
                //     );
                //     let texture_hdl =
                //         ctx.load_texture("image", ui_image, egui::TextureOptions::default());

                //     ui.image(&texture_hdl, egui::Vec2::new(panel_width, panel_width));

                //     // cut the face and show it as well
                //     let mut image_face = image
                //         .crop_imm(
                //             (face_box_px[0] as i32) as u32,
                //             (face_box_px[1] as i32) as u32,
                //             (face_box_px[2] - face_box_px[0]) as u32,
                //             (face_box_px[3] - face_box_px[1]) as u32,
                //         )
                //         .clone();

                //     let imgage_size = [image_face.width() as usize, image_face.height() as usize];

                //     let ui_image = egui::ColorImage::from_rgb(
                //         imgage_size,
                //         image_face.as_rgb8().unwrap().as_raw().as_slice(),
                //     );
                //     let texture_hdl =
                //         ctx.load_texture("image", ui_image, egui::TextureOptions::default());

                //     // add image and set size to panel width

                //     ui.image(&texture_hdl, egui::Vec2::new(panel_width, panel_width));

                //     let projection_matrix =
                //         na::Perspective3::new(1.0, 1.0, 0.1, 1000.0).to_homogeneous();

                //     // create model matrix
                //     let model_matrix = na::Matrix4::identity();

                //     let mut view_matrix = na::Matrix4::look_at_rh(
                //         &na::Point3::new(0.0, 0.0, 2.0),
                //         &na::Point3::new(0.0, 0.0, 0.0),
                //         &na::Vector3::new(0.0, 1.0, 0.0),
                //     );

                //     // extract euler angles
                //     let euler_angles = na::Rotation3::from_matrix(&rotation_matrix).euler_angles();
                //     ui.add(egui::Label::new(format!(
                //         "Euler angles (deg): pitch: {:.2}, yaw: {:.2}, roll: {:.2}",
                //         euler_angles.0.to_degrees(),
                //         euler_angles.1.to_degrees(),
                //         euler_angles.2.to_degrees()
                //     )));

                //     // rotate the view matrix
                //     view_matrix = view_matrix
                //         * na::Matrix4::from_euler_angles(
                //             euler_angles.0,
                //             euler_angles.1,
                //             euler_angles.2,
                //         );

                //     let gizmo = Gizmo::new("My gizmo")
                //         .view_matrix(view_matrix)
                //         .projection_matrix(projection_matrix)
                //         .model_matrix(model_matrix)
                //         .mode(GizmoMode::Rotate);

                // // get current timestamp
                // let current_timestamp = std::time::SystemTime::now()
                //     .duration_since(std::time::UNIX_EPOCH)
                //     .unwrap()
                //     .as_micros();

                // // plot left eye distance
                // let line1: PlotPoints = left_eye_dist_ts
                //     .data
                //     .iter()
                //     .zip(left_eye_dist_ts.timestamp.iter())
                //     .map(|(value, timestamp)| {
                //         let x = (current_timestamp - *timestamp) as f64 * -0.000001;
                //         let y = *value as f64;
                //         [x, y]
                //     })
                //     .collect();

                // let line2: PlotPoints = right_eye_dist_ts
                //     .data
                //     .iter()
                //     .zip(right_eye_dist_ts.timestamp.iter())
                //     .map(|(value, timestamp)| {
                //         let x = (current_timestamp - *timestamp) as f64 * -0.000001;
                //         let y = *value as f64;
                //         [x, y]
                //     })
                //     .collect();

                // Plot::new("my_plot").view_aspect(2.0).show(ui, |plot_ui| {
                //     plot_ui.line(Line::new(line1));
                //     plot_ui.line(Line::new(line2));
                // });
            }
        });

        // redraw everything 30 times per second by default:
        ctx.request_repaint_after(Duration::from_secs(1 / 30));
    }
}




#[inline]
pub fn yuv422_to_rgb24(in_buf: &[u8], out_buf: &mut [u8]) {
    debug_assert!(out_buf.len() as f32 == in_buf.len() as f32 * 1.5);

    in_buf
        .par_chunks_exact(4) // FIXME: use par_array_chunks() when stabalized (https://github.com/rayon-rs/rayon/pull/789)
        .zip(out_buf.par_chunks_exact_mut(6))
        .for_each(|(ch, out)| {
            let y1 = ch[1];
            let y2 = ch[3];
            let cb = ch[0];
            let cr = ch[2];

            let (r, g, b) = ycbcr_to_rgb(y1, cb, cr);

            out[0] = r;
            out[1] = g;
            out[2] = b;

            let (r, g, b) = ycbcr_to_rgb(y2, cb, cr);

            out[3] = r;
            out[4] = g;
            out[5] = b;
        });
}

// COLOR CONVERSION: https://stackoverflow.com/questions/28079010/rgb-to-ycbcr-using-simd-vectors-lose-some-data

#[inline]
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let ycbcr = f32x4::from_array([y as f32, cb as f32 - 128.0f32, cr as f32 - 128.0f32, 0.0]);

    // rec 709: https://mymusing.co/bt-709-yuv-to-rgb-conversion-color/
    let r = (ycbcr * f32x4::from_array([1.0, 0.00000, 1.5748, 0.0])).reduce_sum();
    let g = (ycbcr * f32x4::from_array([1.0, -0.187324, -0.468124, 0.0])).reduce_sum();
    let b = (ycbcr * f32x4::from_array([1.0, 1.8556, 0.00000, 0.0])).reduce_sum();

    (clamp(r), clamp(g), clamp(b))
}

// fn rgb_to_ycbcr((r, g, b): (u8, u8, u8)) -> (u8, u8, u8) {
//     let rgb = F32x4(r as f32, g as f32, b as f32, 1.0);
//     let y = sum(mul(&rgb, F32x4(0.299000, 0.587000, 0.114000, 0.0)));
//     let cb = sum(mul(&rgb, F32x4(-0.168736, -0.331264, 0.500000, 128.0)));
//     let cr = sum(mul(&rgb, F32x4(0.500000, -0.418688, -0.081312, 128.0)));

//     (clamp(y), clamp(cb), clamp(cr))
// }

#[inline]
fn clamp(val: f32) -> u8 {
    if val < 0.0 {
        0
    } else if val > 255.0 {
        255
    } else {
        val.round() as u8
    }
}