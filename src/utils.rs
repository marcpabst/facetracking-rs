use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use image::DynamicImage;

use crate::face_landmarks::ScreenFaceLandmarks;
use crate::webcam::CameraOptions;

// make SharedState an alias for a Mutex protected struct State
pub type SharedState = Arc<Mutex<State>>;

// define the data that will be shared between the threads
pub struct State {
    // the data that will be shared between the threads
    pub fps: Option<f32>,
    pub fps_vec: Vec<f32>,
    pub last_frame_time: Option<SystemTime>,
    pub resolution: Option<(u32, u32)>,
    pub image: Option<DynamicImage>,
    pub devices: Option<Vec<String>>,
    pub current_device: Option<u32>,
    pub recording_path: Option<String>,

    pub face_bbox: Option<(u32, u32, u32, u32)>,
    pub screen_face_landmarks: Option<Arc<dyn ScreenFaceLandmarks>>,

    // time series
    pub left_eye_dist_ts: Option<TimeSeries>,
    pub right_eye_dist_ts: Option<TimeSeries>,
    pub left_iris_x_ts: Option<TimeSeries>,
    pub left_iris_y_ts: Option<TimeSeries>,

    // camera options
    pub camera_options: CameraOptions,
}

// by default, all fields are None
impl Default for State {
    fn default() -> Self {
        Self {
            fps: None,
            fps_vec: Vec::new(),
            last_frame_time: None,
            resolution: None,
            image: None,
            devices: None,
            current_device: None,
            recording_path: None,

            face_bbox: None,
            screen_face_landmarks: None,

            left_eye_dist_ts: None,
            right_eye_dist_ts: None,
            left_iris_x_ts: None,
            left_iris_y_ts: None,

            camera_options: CameraOptions::default(),
        }
    }
}

pub struct TimeSeries {
    data: VecDeque<f32>,
    timestamp: VecDeque<u128>,
    max_length: usize,
}

impl TimeSeries {
    pub fn new(max_length: usize) -> Self {
        Self {
            data: VecDeque::new(),
            timestamp: VecDeque::new(),
            max_length,
        }
    }

    pub fn push(&mut self, value: f32, timestamp: u128) {
        self.data.push_back(value);
        self.timestamp.push_back(timestamp);

        if self.data.len() > self.max_length {
            self.data.pop_front();
            self.timestamp.pop_front();
        }
    }

    pub fn get_mean(&self) -> f32 {
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
