use imageproc::drawing::draw_cross_mut;
use imageproc::drawing::draw_hollow_rect_mut;

use std::time::Duration;

use crate::webcam::CameraOptionBool;
use crate::webcam::CameraOptionInt;

use crate::utils::*;


pub struct FacetrackingApp {
    shared_state: SharedState,
}

impl FacetrackingApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, shared_state: SharedState) -> Self {
        Self {
            shared_state,
        }
    }
}

impl eframe::App for FacetrackingApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let Self {
            shared_state,
        } = self;

        // Examples of how to create different panels and windows.
        // Pick whichever suits you.
        // Tip: a good default choice is to just keep the `CentralPanel`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

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
                if shared_state.lock().unwrap().devices.is_none() {
                    ui.label("No devices found.");
                    return;
                }
                // dropdown with devices;
                let devices = shared_state
                    .lock()
                    .unwrap()
                    .devices
                    .clone()
                    .unwrap_or_default();
                let mut selected_device = shared_state
                    .lock()
                    .unwrap()
                    .current_device
                    .unwrap_or_default() as usize;

                egui::ComboBox::from_label("Camera")
                    .selected_text(format!("{:?}", devices[selected_device]))
                    .show_ui(ui, |ui| {
                        for (i, device) in devices.iter().enumerate() {
                            ui.selectable_value(&mut selected_device, i as usize, device);
                        }
                    });

                // set selected device in shared state
                shared_state.lock().unwrap().current_device = Some(selected_device as u32);

                let camera_options = shared_state.lock().unwrap().camera_options.clone();

                let mut camera_options = camera_options;

                // exposure
                create_camera_option_slider(&mut camera_options.exposure, "Exposure", ui);
                // gain
                create_camera_option_slider(&mut camera_options.gain, "Gain", ui);
                // brightness
                create_camera_option_slider(&mut camera_options.brightness, "Brightness", ui);
                // contrast
                create_camera_option_slider(&mut camera_options.contrast, "Contrast", ui);
                // saturation
                create_camera_option_slider(&mut camera_options.saturation, "Saturation", ui);
                // sharpness
                create_camera_option_slider(&mut camera_options.sharpness, "Sharpness", ui);
                // auto exposure
                create_camera_option_checkbox(
                    &mut camera_options.auto_exposure,
                    "Auto exposure",
                    ui,
                );
                // hue
                create_camera_option_slider(&mut camera_options.hue, "Hue", ui);
                // gamma
                create_camera_option_slider(&mut camera_options.gamma, "Gamma", ui);
                // white balance
                create_camera_option_slider(&mut camera_options.white_balance, "White balance", ui);
                // backlight compensation
                create_camera_option_slider(
                    &mut camera_options.backlight_compensation,
                    "Backlight compensation",
                    ui,
                );

                // reset button
                if ui.button("Reset").clicked() {
                    camera_options.reset();
                }

                // set exposure in shared state
                shared_state.lock().unwrap().camera_options = camera_options;

                let fps = shared_state.lock().unwrap().fps.unwrap_or(0.0);
                let resolution = shared_state.lock().unwrap().resolution.unwrap_or((0, 0));

                ui.add(egui::Label::new(format!("FPS: {}", fps)));
                ui.add(egui::Label::new(format!(
                    "Resolution: {}x{}",
                    resolution.0, resolution.1
                )));

                ui.add(egui::Label::new(format!(
                    "Recording: {}",
                    shared_state.lock().unwrap().recording_path.is_some()
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

                // create ui image
                let ui_image = egui::ColorImage::from_rgb(
                    imgage_size,
                    image.as_rgb8().unwrap().as_raw().as_slice(),
                );

                // create texture handle
                let texture_hdl =
                    ctx.load_texture("image", ui_image, egui::TextureOptions::default());

                // add image and set size to panel width
                let ui_img_width = ui.available_width();
                let ui_img_height = ui_img_width / (image.width() as f32 / image.height() as f32);

                // add image to ui
                ui.image(&texture_hdl, egui::Vec2::new(ui_img_width, ui_img_height));
            }
        });

        // redraw everything 30 times per second by default:
        ctx.request_repaint_after(Duration::from_secs(1 / 30));
    }
}

fn create_camera_option_checkbox<'a>(
    option: &'a mut Option<CameraOptionBool>,
    name: &'a str,
    ui: &mut egui::Ui,
) -> () {
    // if option is None, use grayed out checkbox

    if option.is_none() {
        let mut value = false;
        let checkbox = egui::Checkbox::new(&mut value, format!("{}", name));

        ui.add_enabled(false, checkbox);

        return;
    }

    let _option = option.as_mut().unwrap();
    let mut value = _option.value;

    let checkbox = egui::Checkbox::new(&mut value, format!("{}", name));

    ui.add(checkbox);

    option.as_mut().unwrap().value = value;
}

fn create_camera_option_slider<'a>(
    option: &'a mut Option<CameraOptionInt>,
    name: &'a str,
    ui: &mut egui::Ui,
) -> () {
    // if option is None, use grayed out slider

    if option.is_none() {
        let mut value = 0.0;
        let slider = egui::Slider::new(&mut value, 0.0..=1.0).text(format!("{}", name));

        ui.add_enabled(false, slider);

        return;
    }

    let _option = option.as_mut().unwrap();

    let mut value = _option.value as f32;
    let min = _option.min as f32;
    let max = _option.max as f32;

    let slider = egui::Slider::new(&mut value, min..=max).text(format!("{}", name));

    ui.add(slider);

    option.as_mut().unwrap().value = value as i32;
}
