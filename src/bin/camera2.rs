use std::sync::{Arc, Mutex};
use std::time::{SystemTime, Duration};
use image::{DynamicImage, ImageBuffer};
use nokhwa::{
    native_api_backend,
    pixel_format::{RgbFormat},
    query,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution, CameraFormat, FrameFormat},
    Camera,
};

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
    pub image: DynamicImage,
    pub devices: Vec<String>,
    pub is_recording: bool,
    pub recording_path: String,
}

// function that runs as a thread and performs the actual work
// function should accept a shared state as an argument
fn worker_thread(shared_state: Arc<Mutex<State>>) {
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
    std::mem::drop(guard);

    // select camera
    let index_i = 0;
    let index = CameraIndex::Index(index_i);

    let fps = 60;
    let fourcc = FrameFormat::YUYV;

    let resolution = Resolution::new(1280, 720);
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
            let image = image.to_rgb8();
            let image = DynamicImage::ImageRgb8(image);

            // update the shared state image
            let mut guard = shared_state.lock().unwrap();
            guard.image = image;
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

        last_frame_time = SystemTime::now();
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
        image: DynamicImage::ImageRgb8(ImageBuffer::from_vec(1280, 720, vec).unwrap()),
        devices: Vec::new(),
        is_recording: false,
        recording_path: String::new(),
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
        Box::new(|cc| Box::new(TemplateApp::new(cc, gui_data))),
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

            let mut selected_device = 0;

            egui::ComboBox::from_label("Capture device")
                .selected_text(format!("{:?}", devices[selected_device]))
                .show_ui(ui, |ui| {
                    for (i, device) in devices.iter().enumerate() {
                        ui.selectable_value(&mut selected_device, i, device);
                    }
            });
        
         
            ui.add(egui::Slider::new(saturation, 0.0..=10.0).text("Saturation"));
            ui.add(egui::Slider::new(brightness, 0.0..=10.0).text("Brightness"));
            ui.add(egui::Slider::new(contrast, 0.0..=10.0).text("Contrast"));
            // add text (from shared state)
            let fps = shared_state.lock().unwrap().fps;
            ui.add(egui::Label::new(format!("FPS: {}", fps)));

            // aquire the mutex
            let guard = shared_state.lock().unwrap();
            // clone the image
            let image = guard.image.clone();
            // unlock the mutex
            std::mem::drop(guard);

            // flip the image
            let image = image.fliph();

            let imgage_size = [image.width() as usize, image.height() as usize];

            let image = egui::ColorImage::from_rgb(imgage_size, image.as_rgb8().unwrap().as_raw().as_slice());
            let texture_hdl = ctx.load_texture("image", image, egui::TextureOptions::default());
            
            ui.image(&texture_hdl, egui::Vec2::new(1280.0, 720.0));

            
        });

        // redraw everything 60 times per second by default:
        ctx.request_repaint_after(Duration::from_secs(1/30));

 
    }
}
