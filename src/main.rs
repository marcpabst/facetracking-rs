extern crate facetracking;

use facetracking::face_detection::model_blazeface::BlazefaceModel;
use facetracking::face_detection::FaceDetectionModel;

use facetracking::face_landmarks::model_mediapipe::MediapipeFaceLandmarksModel;
use facetracking::face_landmarks::FaceLandmarksModel;

use facetracking::utils::SharedState;
use facetracking::utils::State;
use facetracking::webcam::*;

use crate::facetracking::app::FacetrackingApp;

use rayon::spawn_fifo;

use image::{DynamicImage, ImageBuffer};

use openpnp_capture::{context::CONTEXT, Device, Format, Stream};

use std::sync::atomic::AtomicI32;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

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
                guard.recording_path = Some(path.to_string());
                std::mem::drop(guard);
            } else if msg_str.starts_with("stop") {
                let mut guard = shared_state.lock().unwrap();
                guard.recording_path = None;
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

// function that runs as a thread and performs the actual work
// function should accept a shared state as an argument
fn camera_thread_fn(shared_state: Arc<Mutex<State>>) {
    loop {
        // Fetch some generic device information
        let devices = Device::enumerate();

        let current_device = shared_state.lock().unwrap().current_device.unwrap_or(0) as usize;

        let mut inputs: Vec<(u32, String, Format)> = Vec::new();

        for index in devices {
            let dev = Device::new(index).expect("Failed to open device");
            let formats = dev.formats();
            let index = dev.index;
            // push formats to vector
            for format in formats.iter() {
                inputs.push((index as u32, dev.name.clone(), *format));
            }
        }

        // fetch the device
        let dev = Device::new(inputs[current_device].0).expect("Failed to open device");

        // create the stream
        let format = inputs[current_device].2;
        let stream = Stream::new(&dev, &format).expect("Failed to create stream");
        let resolution = (format.width, format.height);

        // add devices to shared state
        shared_state.lock().unwrap().devices = Some(
            inputs
                .iter()
                .map(|d| format!("{}: {}x{} @ {} fps", d.1, d.2.width, d.2.height, d.2.fps))
                .collect(),
        );

        // set up face landmarks model
        let face_detection_model = Arc::new(BlazefaceModel::new());
        let face_landmarks_model = Arc::new(MediapipeFaceLandmarksModel::new());

        // set resolution in shared state
        shared_state.lock().unwrap().resolution = Some(resolution);

        // prepare a buffer to hold camera frames
        let mut rgb_buffer = Vec::new();

        let current_num_parallel_tasks = Arc::new(AtomicI32::new(0));

        // get context
        let ctx = CONTEXT.lock().unwrap().inner;

        // read camera options and add them to shared state
        let mut camera_options = read_camera_options(ctx, stream.id());
        shared_state.lock().unwrap().camera_options = camera_options.clone();

        // set last frame time
        loop {
            shared_state.lock().unwrap().last_frame_time = Some(SystemTime::now());

            // check if the current device has changed (through the GUI)
            if shared_state.lock().unwrap().current_device.unwrap_or(0) as usize != current_device {
                // if so, break out of the loop and start over
                break;
            }

            let new_camera_options = shared_state.lock().unwrap().camera_options.clone();
            let diff = camera_options.diff(&new_camera_options);
            camera_options = new_camera_options;

            // set camera options
            set_camera_options(ctx, stream.id(), diff);

            // poll the stream using stream.poll() until it returns true, every 0.5 ms
            while !stream.poll() {
                std::thread::sleep(std::time::Duration::from_millis(0));
            }

            // obtain timestamp
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros();

            // read the frame into the buffer
            let _ = stream.read(&mut rgb_buffer);

            // create ImageBuffer from buffer
            let image_buffer =
                ImageBuffer::from_vec(resolution.0, resolution.1, rgb_buffer.clone()).unwrap();

            // check current number of concurrent tasks is higher than the number of threads
            if current_num_parallel_tasks.load(std::sync::atomic::Ordering::SeqCst)
                > rayon::current_num_threads() as i32
            {
                // we should skip this frame to avoid lag
                println!("Too busy, skipping frame. Consider increasing the number of threads or descreasing the frame rate.");
                continue;
            }

            // clone values to pass to rayon threadpool
            let shared_state = shared_state.clone();
            let face_detection_model = face_detection_model.clone();
            let face_landmarks_model = face_landmarks_model.clone();
            let current_num_parallel_tasks = current_num_parallel_tasks.clone();

            // increase the number of concurrent tasks by 1
            current_num_parallel_tasks.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            // hand frame off to rayon threadpool for processing an IO
            spawn_fifo(move || {
                let image = DynamicImage::ImageRgb8(image_buffer);

                let recording_path = shared_state.lock().unwrap().recording_path.clone();

                // flip image
                let image = image.fliph();

                let face_bbox;

                // if we have no face landmarks or the confidence is too low, run through blazeface model
                if shared_state.lock().unwrap().screen_face_landmarks.is_none()
                    || shared_state
                        .lock()
                        .unwrap()
                        .screen_face_landmarks
                        .clone()
                        .unwrap()
                        .get_confidence()
                        < 0.5
                {
                    // run through blazeface model to get face bbox
                    face_bbox = Some(face_detection_model.run(&image).to_tuple());
                } else {
                    // get face bbox from shared state
                    face_bbox = shared_state.lock().unwrap().face_bbox.clone();
                }

                // run through mediapipe model
                let (face_landmarks, face_bbox) = face_landmarks_model.run(&image, face_bbox);

                // set image
                shared_state.lock().unwrap().image = Some(image.clone());

                // set face bbox
                shared_state.lock().unwrap().face_bbox = Some(face_bbox);

                // set landmarks
                shared_state.lock().unwrap().screen_face_landmarks = Some(face_landmarks.into());

                // if recording is enabled, save the frame to disk
                if recording_path.is_some() {
                    let filename = format!("{}/frame_{}.jpg", recording_path.unwrap(), timestamp);
                    image
                        .save(filename)
                        .unwrap_or_else(|_| println!("Error saving frame."));
                }

                // decrease the number of concurrent tasks by 1
                current_num_parallel_tasks.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            });
        }
    }
}

fn main() -> eframe::Result<()> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    // CREATE SHARED STATE
    let shared_state: SharedState = Arc::new(Mutex::new(State::default()));

    // set recording path
    //shared_state.lock().unwrap().recording_path = Some("/Users/marc/test".to_string());

    // CREATE A THREAD FOR THE CAMERA
    let thread_data = Arc::clone(&shared_state);
    std::thread::spawn(|| camera_thread_fn(thread_data));

    // CREATE A THREAD FOR THE SERVER
    // let thread_data2 = Arc::clone(&shared_state);
    // std::thread::spawn(|| server(thread_data2));

    // START GUI
    let gui_data = Arc::clone(&shared_state);

    let native_options = eframe::NativeOptions::default();

    let version = env!("CARGO_PKG_VERSION");
    let window_name = format!("Camera Recorder v{}", version);

    eframe::run_native(
        &window_name,
        native_options,
        Box::new(move |cc| {
            let _re_ui = re_ui::ReUi::load_and_apply(&cc.egui_ctx);
            Box::new(FacetrackingApp::new(cc, gui_data))
        }),
    )
}