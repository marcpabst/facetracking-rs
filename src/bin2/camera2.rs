extern crate rust_eyetracking;
use egui_gizmo::Gizmo;
use egui_gizmo::GizmoMode;
use imageproc::drawing::draw_cross_mut;
use imageproc::drawing::draw_hollow_rect_mut;
use rust_eyetracking::geometry_pipeline::MediapipeLandmarkConverter;
use nalgebra as na;
use rust_eyetracking::facetracking::{MediapipeFaceLandmarksModel, MediapipeNormalizedFaceLandmarks, BlazefaceModel, FaceBox, LandmarkPoint};
use egui::plot::{Line, Plot, PlotPoints};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, Duration};
use image::{DynamicImage, ImageBuffer, Luma};
use nokhwa::{
    native_api_backend,
    pixel_format::{RgbFormat},
    query,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution, CameraFormat, FrameFormat},
    Camera,
};
use re_ui::ReUi;

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


fn server(shared_state: Arc<Mutex<State>>) -> Result<(), nng::Error> {
    // Set up the server and listen for connections on the specified address.
    let s = Socket::new(Protocol::Rep0)?;

    // Set up the callback that will be called when the server receives a message
    // from the client.
    let ctx = Context::new(&s)?;
    let ctx_clone = ctx.clone();
    let aio = Aio::new(move |aio, res| worker_callback(aio, &ctx_clone, res, shared_state.clone()))?;

    s.listen(ADDRESS)?;

    // start the worker thread
    // subscribe to messages
    
    ctx.recv(&aio)?;

    println!("Server listening on {}", ADDRESS);

    loop {
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }

}

fn worker_callback(aio: Aio, ctx: &Context, res: AioResult, shared_state: Arc<Mutex<State>>) {
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
                guard.is_recording = true;
                guard.recording_path = path.to_string();
                std::mem::drop(guard);
            }
            else if  msg_str.starts_with("stop") {
                let mut guard = shared_state.lock().unwrap();
                guard.is_recording = false;
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


// create a state struct to hold the data that will be shared between the threads
pub struct State {
    // the data that will be shared between the threads
    pub fps: f32,
    pub resolution: (u32, u32),
    pub image_valid: bool,
    pub image: DynamicImage,
    pub devices: Vec<String>,
    pub current_device: u32,
    pub is_recording: bool,
    pub recording_path: String,
    pub face_landmarks: MediapipeNormalizedFaceLandmarks,
    pub face_box: FaceBox,
    pub rotation_matrix: na::Matrix3<f32>,

    // time series
    pub left_eye_dist_ts: TimeSeries,
    pub right_eye_dist_ts: TimeSeries,
    pub left_iris_x_ts: TimeSeries,
    pub left_iris_y_ts: TimeSeries,
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
    let mut guard = shared_state.lock().unwrap();
    guard.devices = devices.iter().map(|d| d.human_name()).collect();


    // set up face landmarks model
    let blazeface_model = BlazefaceModel::new();
    let face_landmarks_model = MediapipeFaceLandmarksModel::new();
    
    // select camera
    let index_i = guard.current_device.clone() as u32;

    std::mem::drop(guard);
    let index = CameraIndex::Index(index_i);

    let fps = 60;
    let fourcc = FrameFormat::YUYV;

    let resolution = Resolution::new(1280, 720);

    // set resolution in shared state
    let mut guard = shared_state.lock().unwrap();
    guard.resolution = (resolution.width_x, resolution.height_y);
    std::mem::drop(guard);

    let camera_format = CameraFormat::new(resolution, fourcc, fps);
 
    let requested = RequestedFormat::new::<RgbFormat>(
        RequestedFormatType::AbsoluteHighestResolution
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


    // open the camera
    println!("Opening stram...");
    camera.open_stream();
    println!("Opened camera");

    // create ringbuffer for frame deltas
    let mut fps_vec: Vec<f32> = Vec::new();

    let mut last_frame_time = SystemTime::now();

    loop { 
        
        // if macos
        #[cfg(target_os = "macos")]
        {
        
            let frame: std::borrow::Cow<'_, [u8]> = camera.frame_raw().unwrap();

            // get raw image as vector
            let data = frame.to_vec();

            // convert the broken NV12 format to grayscale (hint: its not nv12)
            let greyscale_vec: Vec<u8> =
                convert_nv12_to_grayscale(&data, resolution.width_x, resolution.height_y);

            // create greyscale imagebuffer
            let image = DynamicImage::ImageLuma8(ImageBuffer::from_vec(resolution.width_x, resolution.height_y, greyscale_vec).unwrap());

            // convert to rgb
            let image_buffer = image.to_rgb8();
            let image = DynamicImage::ImageRgb8(image_buffer);

            // push image through face landmarks model
            // crop to square, using the shorter edge (centered)
            let shorter_edge = image.width().min(image.height());
            
            let croped_image = image.crop_imm(
                (image.width() - shorter_edge) / 2,
                (image.height() - shorter_edge) / 2,
                shorter_edge,
                shorter_edge,
            );

            // resize to 256x256
            let resized_image = croped_image.resize_exact(256, 256, image::imageops::FilterType::Nearest);
  

            // runt through blazeface model
            let face_box: FaceBox = blazeface_model.forward(resized_image.clone()).add_margin(0.2);

            // use face box to crop image
            let image_face = croped_image.crop_imm(
                (face_box.x1 * croped_image.width() as f32) as u32,
                (face_box.y1 * croped_image.height() as f32) as u32,
                (face_box.x2 * croped_image.width() as f32) as u32 - (face_box.x1 * croped_image.width() as f32) as u32,
                (face_box.y2 * croped_image.height() as f32) as u32 - (face_box.y1 * croped_image.height() as f32) as u32,
            );

            // resize to 256x256
            let image_face = image_face.resize_exact(256, 256, image::imageops::FilterType::Nearest);

            let face_landmarks = face_landmarks_model.forward(image_face.clone());

            let normal = face_landmarks.get_rotation_matrix();

            // get timestamp
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros();

            // calculate distance between LeftEyeTop and LeftEyeBottom
            let face_width = face_landmarks.get_distance(LandmarkPoint::LeftCheek, LandmarkPoint::RightCheek);

            let mut left_eye_dist = face_landmarks.get_distance(LandmarkPoint::LeftEyeTop, LandmarkPoint::LeftEyeBottom);
            left_eye_dist /= face_width;
            let right_eye_dist = face_landmarks.get_distance(LandmarkPoint::RightEyeTop, LandmarkPoint::RightEyeBottom);
            let right_eye_dist = right_eye_dist / face_width;

            // get iris locations
            let (left_iris_pos, right_iris_pos) = face_landmarks.get_iris_locations();

            // update the shared state image and face landmarks
            let mut guard = shared_state.lock().unwrap();
           
            guard.left_eye_dist_ts.push(left_eye_dist, timestamp);
            guard.right_eye_dist_ts.push(right_eye_dist, timestamp);
            guard.left_iris_x_ts.push(left_iris_pos.x, timestamp);
            guard.left_iris_y_ts.push(left_iris_pos.y, timestamp);

            guard.image = croped_image;
            guard.face_landmarks = face_landmarks;
            guard.face_box = face_box;
            guard.image_valid = true;
            guard.rotation_matrix = normal;
            std::mem::drop(guard);

        }
        #[cfg(not(target_os = "macos"))]
        {
            // get frame
            let frame = camera.frame().unwrap().decode_image::<RgbFormat>().unwrap();
            // create dynamic image
            let image = DynamicImage::ImageRgb8(frame);

            // update the shared state image
            let mut guard = shared_state.lock().unwrap();
            guard.image = image;
            std::mem::drop(guard);
        }

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

        // check if recording is enabled
        if shared_state.lock().unwrap().is_recording {

            println!("Recording frame");

            // // get current timestamp
            // let timestamp = std::time::SystemTime::now()
            //     .duration_since(std::time::UNIX_EPOCH)
            //     .unwrap()
            //     .as_millis();

            // // write frame to disk
            // let frame_path = &shared_state.lock().unwrap().recording_path;
            // let filename = format!("{}/frame_{}.png", frame_path, timestamp);

            // // save greyscale_img image as jpg
            // shared_state.lock().unwrap().image.save(filename).unwrap();

        }

        // update the shared state with the mean fps
        let mut guard = shared_state.lock().unwrap();
        guard.fps = (fps_vec.iter().sum::<f32>() / fps_vec.len() as f32).round();
        std::mem::drop(guard);

        // check if the current device has changed
        let mut guard = shared_state.lock().unwrap();
        if guard.current_device != index_i {
            // break out of the loop
            println!("Device changed from {} to {}", guard.current_device, index_i);
            break;
        }
        std::mem::drop(guard);

        last_frame_time = SystemTime::now();
    }
}
}


// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    // create rgba dummy image
    let vec: Vec<u8> = vec![0u8; 1280 * 720 * 3];

    // create new shared state
    let shared_state = Arc::new(Mutex::new(State {
        fps: 0.0,
        resolution: (0, 0),
        image_valid: false,
        image: DynamicImage::ImageRgb8(ImageBuffer::from_vec(1280, 720, vec).unwrap()),
        devices: Vec::new(),
        current_device: 0,
        is_recording: false,
        recording_path: String::new(),
        face_landmarks: MediapipeNormalizedFaceLandmarks::from_vec(vec![0.0; 478 * 3]),
        face_box: FaceBox{x1: 0.0, y1: 0.0, x2: 0.0, y2: 0.0},
        rotation_matrix: na::Matrix3::identity(),

        // time series
        left_eye_dist_ts: TimeSeries::new(1000),
        right_eye_dist_ts: TimeSeries::new(1000),
        left_iris_x_ts: TimeSeries::new(50),
        left_iris_y_ts: TimeSeries::new(50),
    }));

    // create a thread and run the worker function in it
    let thread_data = Arc::clone(&shared_state);
    std::thread::spawn(|| worker_thread(thread_data));

    // start the nng server
    let thread_data2 = Arc::clone(&shared_state);
    std::thread::spawn(|| server(thread_data2));

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
    shared_state: Arc<Mutex<State>>,
}


impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>, shared_state: Arc<Mutex<State>>    ) -> Self {
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
        let Self { saturation, brightness, contrast , shared_state} = self;

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

            // dropdown with devices
            let guard = shared_state.lock().unwrap();
            let devices = &guard.devices.clone();
            std::mem::drop(guard);

            let mut selected_device = shared_state.lock().unwrap().current_device.clone() as usize;

            egui::ComboBox::from_label("Camera")
                .selected_text(format!("{:?}", devices[selected_device]))
                .show_ui(ui, |ui| {
                    for (i, device) in devices.iter().enumerate() {
                        ui.selectable_value(&mut selected_device, i as usize, device);
                    }
            });

            // set mutex
            let mut guard = shared_state.lock().unwrap(); 
            guard.current_device = selected_device as u32;
            std::mem::drop(guard);
        
         
            // ui.add(egui::Slider::new(saturation, 0.0..=10.0).text("Saturation"));
            // ui.add(egui::Slider::new(brightness, 0.0..=10.0).text("Brightness"));
            // ui.add(egui::Slider::new(contrast, 0.0..=10.0).text("Contrast"));
            // add text (from shared state)
            let fps = shared_state.lock().unwrap().fps;
            let resolution = shared_state.lock().unwrap().resolution;
            ui.add(egui::Label::new(format!("FPS: {}", fps)));
            ui.add(egui::Label::new(format!("Resolution: {}x{}", resolution.0, resolution.1)));
            ui.add(egui::Label::new(format!("Recording: {}", shared_state.lock().unwrap().is_recording)));


            // aquire the mutex
            let guard = shared_state.lock().unwrap();
            // clone the image
            let mut image = guard.image.clone();
            let face_landmarks = guard.face_landmarks.clone();
            let face_box = guard.face_box.clone();
            let image_valid = guard.image_valid.clone();
            let rotation_matrix = guard.rotation_matrix.clone();
            let left_eye_dist_ts = guard.left_eye_dist_ts.clone();
            let right_eye_dist_ts = guard.right_eye_dist_ts.clone();
            let left_iris_x_ts = guard.left_iris_x_ts.clone();
            let left_iris_y_ts = guard.left_iris_y_ts.clone();
            // unlock the mutex
            std::mem::drop(guard);

            // check if image is valid
            if image_valid {
        
                // flip the image
                let mut image = image.fliph();

                let green = image::Rgba([0u8, 255u8, 0u8, 255u8]);
                let red = image::Rgba([255u8, 0u8, 0u8, 255u8]);

                let imgage_size = [image.width() as usize, image.height() as usize];

                // draw face box
                let face_box_px: [f32; 4] = face_box.fliph().to_pixel_space(imgage_size[0] as u32, imgage_size[1] as u32);
                let rect = imageproc::rect::Rect::at(face_box_px[0] as i32, face_box_px[1] as i32)
                    .of_size((face_box_px[2] - face_box_px[0]) as u32, (face_box_px[3] - face_box_px[1]) as u32);


                draw_hollow_rect_mut(&mut image, rect, green);

                // plot face landmarks
                for p in face_landmarks.points.iter() {

                    // convert to image coords
                    let x = (1.0 - p.x) * (face_box_px[2] - face_box_px[0]) + face_box_px[0];
                    let y = p.y * (face_box_px[3] - face_box_px[1]) + face_box_px[1];

                    draw_cross_mut(&mut image, green, x as i32, y as i32);
                }

                let panel_width = ui.available_width();

                let ui_image = egui::ColorImage::from_rgb(imgage_size, image.as_rgb8().unwrap().as_raw().as_slice());
                let texture_hdl = ctx.load_texture("image", ui_image, egui::TextureOptions::default());
                
                ui.image(&texture_hdl, egui::Vec2::new(panel_width, panel_width));

                // cut the face and show it as well
                let mut image_face = image.crop_imm(
                    (face_box_px[0] as i32) as u32,
                    (face_box_px[1] as i32) as u32,
                    (face_box_px[2] - face_box_px[0]) as u32,
                    (face_box_px[3] - face_box_px[1]) as u32,
                ).clone();

                let imgage_size = [image_face.width() as usize, image_face.height() as usize];

                let ui_image = egui::ColorImage::from_rgb(imgage_size, image_face.as_rgb8().unwrap().as_raw().as_slice());
                let texture_hdl = ctx.load_texture("image", ui_image, egui::TextureOptions::default());

                // add image and set size to panel width
                
                ui.image(&texture_hdl, egui::Vec2::new(panel_width, panel_width));

                let projection_matrix = na::Perspective3::new(1.0, 1.0, 0.1, 1000.0).to_homogeneous();
        
                // create model matrix
                let model_matrix = na::Matrix4::identity();
                
                let mut view_matrix = na::Matrix4::look_at_rh(
                    &na::Point3::new(0.0, 0.0, 2.0),
                    &na::Point3::new(0.0, 0.0, 0.0),
                    &na::Vector3::new(0.0, 1.0, 0.0),
                );

                // extract euler angles
                let euler_angles = na::Rotation3::from_matrix(&rotation_matrix).euler_angles();
                ui.add(egui::Label::new(format!("Euler angles (deg): pitch: {:.2}, yaw: {:.2}, roll: {:.2}", euler_angles.0.to_degrees(), euler_angles.1.to_degrees(), euler_angles.2.to_degrees())));

                // rotate the view matrix
                view_matrix = view_matrix * na::Matrix4::from_euler_angles(euler_angles.0, euler_angles.1, euler_angles.2);

                let gizmo = Gizmo::new("My gizmo")
                    .view_matrix(view_matrix)
                    .projection_matrix(projection_matrix)
                    .model_matrix(model_matrix)
                    .mode(GizmoMode::Rotate);
    
                // // create container for the gizmo
                // let container = egui::Area::new("gizmo")
                //     .fixed_pos(egui::Pos2::new(0.0, 0.0))
                //     .
                //     .show(ui.ctx(), |ui| {
                //         ui.allocate_ui(egui::Vec2::new(250.0, 250.0), |ui| {
                //             gizmo.interact(ui);
                //         });
                //     });
            
    

            }

            // get current timestamp
            let current_timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros();
           
            // plot left eye distance
            let line1: PlotPoints = left_eye_dist_ts.data.iter().zip(left_eye_dist_ts.timestamp.iter()).map(|(value, timestamp)| {
                let x = (current_timestamp - *timestamp) as f64 * -0.000001;
                let y = *value as f64;
                [x, y]
            }).collect();

            let line2: PlotPoints = right_eye_dist_ts.data.iter().zip(right_eye_dist_ts.timestamp.iter()).map(|(value, timestamp)| {
                let x = (current_timestamp - *timestamp) as f64 * -0.000001;
                let y = *value as f64;
                [x, y]
            }).collect();

         
            Plot::new("my_plot").view_aspect(2.0).show(ui, |plot_ui| {
                plot_ui.line(Line::new(line1));
                plot_ui.line(Line::new(line2));
            });

            // plot left iris x and y
            // let line: PlotPoints = left_iris_x_ts.data.iter().zip(left_iris_y_ts.data.iter()).map(|(value_x, value_y)| {
            //     let x = - *value_x as f64;
            //     let y = *value_y as f64;
            //     [x, y]
            // }).collect();

            // Plot::new("my_plot2").view_aspect(1.0).show(ui, |plot_ui| {
            //     plot_ui.line(Line::new(line));
            // });
   
            
        });

        // redraw everything 30 times per second by default:
        ctx.request_repaint_after(Duration::from_secs(1/30));

 
    }
}


