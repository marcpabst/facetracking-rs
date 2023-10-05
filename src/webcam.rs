// thin wrapper around openpnp_capture
use openpnp_sys::CapContext;
use openpnp_sys::CapPropertyID;
use openpnp_sys::CapStream;

use openpnp_capture_sys as openpnp_sys;

#[derive(Debug, Clone)]
pub struct CameraOptionInt {
    pub value: i32,
    pub default: i32,
    pub min: i32,
    pub max: i32,
}

#[derive(Debug, Clone)]
pub struct CameraOptionBool {
    pub value: bool,
    pub default: bool,
}

impl CameraOptionInt {
    pub fn reset(&mut self) {
        self.value = self.default;
    }

    pub fn compare(&self, other: &CameraOptionInt) -> bool {
        self.value == other.value
    }
}

impl CameraOptionBool {
    pub fn reset(&mut self) {
        self.value = self.default;
    }

    pub fn compare(&self, other: &CameraOptionBool) -> bool {
        self.value == other.value
    }
}

#[derive(Debug, Clone)]
pub struct CameraOptions {
    pub exposure: Option<CameraOptionInt>,
    pub gain: Option<CameraOptionInt>,
    pub brightness: Option<CameraOptionInt>,
    pub contrast: Option<CameraOptionInt>,
    pub saturation: Option<CameraOptionInt>,
    pub sharpness: Option<CameraOptionInt>,
    pub auto_exposure: Option<CameraOptionBool>,
    pub hue: Option<CameraOptionInt>,
    pub gamma: Option<CameraOptionInt>,
    pub white_balance: Option<CameraOptionInt>,
    pub backlight_compensation: Option<CameraOptionInt>,
    pub auto_white_balance: Option<CameraOptionBool>,
}

// implement default for CameraOptions
impl Default for CameraOptions {
    fn default() -> Self {
        Self {
            exposure: None,
            gain: None,
            brightness: None,
            contrast: None,
            saturation: None,
            sharpness: None,
            auto_exposure: None,
            hue: None,
            gamma: None,
            white_balance: None,
            backlight_compensation: None,
            auto_white_balance: None,
        }
    }
}

impl CameraOptions {
    pub fn reset(&mut self) {
        if self.exposure.is_some() {
            self.exposure.as_mut().unwrap().reset();
        }
        if self.gain.is_some() {
            self.gain.as_mut().unwrap().reset();
        }
        if self.brightness.is_some() {
            self.brightness.as_mut().unwrap().reset();
        }
        if self.contrast.is_some() {
            self.contrast.as_mut().unwrap().reset();
        }
        if self.saturation.is_some() {
            self.saturation.as_mut().unwrap().reset();
        }
        if self.sharpness.is_some() {
            self.sharpness.as_mut().unwrap().reset();
        }
        if self.hue.is_some() {
            self.hue.as_mut().unwrap().reset();
        }
        if self.gamma.is_some() {
            self.gamma.as_mut().unwrap().reset();
        }
        if self.white_balance.is_some() {
            self.white_balance.as_mut().unwrap().reset();
        }
        if self.backlight_compensation.is_some() {
            self.backlight_compensation.as_mut().unwrap().reset();
        }
    }

    // compare two CameraOptions, and returns a copy of self with values that are identical to other set to None
    pub fn diff(&self, other: &CameraOptions) -> CameraOptions
    {
        let mut diff = self.clone();

        if self.exposure.is_some() && other.exposure.is_some() && self.exposure.as_ref().unwrap().compare(other.exposure.as_ref().unwrap()) {
            diff.exposure = None;
        }

        if self.gain.is_some() && other.gain.is_some() && self.gain.as_ref().unwrap().compare(other.gain.as_ref().unwrap()) {
            diff.gain = None;
        }

        if self.brightness.is_some() && other.brightness.is_some() && self.brightness.as_ref().unwrap().compare(other.brightness.as_ref().unwrap()) {
            diff.brightness = None;
        }

        if self.contrast.is_some() && other.contrast.is_some() && self.contrast.as_ref().unwrap().compare(other.contrast.as_ref().unwrap()) {
            diff.contrast = None;
        }

        if self.saturation.is_some() && other.saturation.is_some() && self.saturation.as_ref().unwrap().compare(other.saturation.as_ref().unwrap()) {
            diff.saturation = None;
        }

        if self.sharpness.is_some() && other.sharpness.is_some() && self.sharpness.as_ref().unwrap().compare(other.sharpness.as_ref().unwrap()) {
            diff.sharpness = None;
        }

        if self.hue.is_some() && other.hue.is_some() && self.hue.as_ref().unwrap().compare(other.hue.as_ref().unwrap()) {
            diff.hue = None;
        }

        if self.gamma.is_some() && other.gamma.is_some() && self.gamma.as_ref().unwrap().compare(other.gamma.as_ref().unwrap()) {
            diff.gamma = None;
        }

        if self.white_balance.is_some() && other.white_balance.is_some() && self.white_balance.as_ref().unwrap().compare(other.white_balance.as_ref().unwrap()) {
            diff.white_balance = None;
        }

        if self.backlight_compensation.is_some() && other.backlight_compensation.is_some() && self.backlight_compensation.as_ref().unwrap().compare(other.backlight_compensation.as_ref().unwrap()) {
            diff.backlight_compensation = None;
        }

        if self.auto_exposure.is_some() && other.auto_exposure.is_some() && self.auto_exposure.as_ref().unwrap().compare(other.auto_exposure.as_ref().unwrap()) {
            diff.auto_exposure = None;
        }

        if self.auto_white_balance.is_some() && other.auto_white_balance.is_some() && self.auto_white_balance.as_ref().unwrap().compare(other.auto_white_balance.as_ref().unwrap()) {
            diff.auto_white_balance = None;
        }

        diff


    }

}


fn read_camera_properth(
    ctx: CapContext,
    stream: CapStream,
    property: CapPropertyID,
) -> Option<i32> {
    let mut value = 0;
    let res = unsafe { openpnp_sys::Cap_getProperty(ctx, stream, property, &mut value) };
    if res == openpnp_sys::CAPRESULT_OK {
        Some(value)
    } else {
        None
    }
}

fn read_camera_property_and_limits(
    ctx: CapContext,
    stream: CapStream,
    property: CapPropertyID,
) -> Option<CameraOptionInt> {
    let mut min = 0;
    let mut max = 0;
    let mut default = 0;
    let res = unsafe {
        openpnp_sys::Cap_getPropertyLimits(ctx, stream, property, &mut min, &mut max, &mut default)
    };

    let value = read_camera_properth(ctx, stream, property);

    if res == openpnp_sys::CAPRESULT_OK {
        Some(CameraOptionInt {
            value: value.unwrap_or(default),
            default,
            min,
            max,
        })
    } else {
        None
    }
}

fn read_camera_auto_properth(
    ctx: CapContext,
    stream: CapStream,
    property: CapPropertyID,
) -> Option<CameraOptionBool> {
    let mut value = 0;
    let res = unsafe { openpnp_sys::Cap_getAutoProperty(ctx, stream, property, &mut value) };
    if res == openpnp_sys::CAPRESULT_OK {
        Some(CameraOptionBool {
            value: value == 1,
            default: true,
        })
    } else {
        None
    }
}

pub fn read_camera_options(ctx: CapContext, stream: CapStream) -> CameraOptions {
    // get camera options

    let exposure = read_camera_property_and_limits(ctx, stream, openpnp_sys::CAPPROPID_EXPOSURE);
    let gain = read_camera_property_and_limits(ctx, stream, openpnp_sys::CAPPROPID_GAIN);
    let brightness =
        read_camera_property_and_limits(ctx, stream, openpnp_sys::CAPPROPID_BRIGHTNESS);
    let contrast = read_camera_property_and_limits(ctx, stream, openpnp_sys::CAPPROPID_CONTRAST);
    let saturation =
        read_camera_property_and_limits(ctx, stream, openpnp_sys::CAPPROPID_SATURATION);
    let sharpness = read_camera_property_and_limits(ctx, stream, openpnp_sys::CAPPROPID_SHARPNESS);
    let auto_exposure = read_camera_auto_properth(ctx, stream, openpnp_sys::CAPPROPID_EXPOSURE);
    let hue = read_camera_property_and_limits(ctx, stream, openpnp_sys::CAPPROPID_HUE);
    let gamma = read_camera_property_and_limits(ctx, stream, openpnp_sys::CAPPROPID_GAMMA);
    let white_balance =
        read_camera_property_and_limits(ctx, stream, openpnp_sys::CAPPROPID_WHITEBALANCE);
    let backlight_compensation =
        read_camera_property_and_limits(ctx, stream, openpnp_sys::CAPPROPID_BACKLIGHTCOMP);
    let auto_white_balance =
        read_camera_auto_properth(ctx, stream, openpnp_sys::CAPPROPID_WHITEBALANCE);

    CameraOptions {
        exposure,
        gain,
        brightness,
        contrast,
        saturation,
        sharpness,
        auto_exposure,
        hue,
        gamma,
        white_balance,
        backlight_compensation,
        auto_white_balance,
    }
}

pub fn set_camera_property(ctx: CapContext, stream: CapStream, property: CapPropertyID, value: i32) {
    let res = unsafe { openpnp_sys::Cap_setProperty(ctx, stream, property, value) };
    if res != openpnp_sys::CAPRESULT_OK {
        println!(
            "Error setting camera property {} with value {}. Result: {}",
            property, value, res
        );
    }
}

pub fn set_camera_auto_property(
    ctx: CapContext,
    stream: CapStream,
    property: CapPropertyID,
    value: bool,
) {
    let res = unsafe { openpnp_sys::Cap_setAutoProperty(ctx, stream, property, !value as u32) };
    if res != openpnp_sys::CAPRESULT_OK {
        println!("Error setting camera property auto.");
    }
}

pub fn set_camera_options(ctx: CapContext, stream: CapStream, options: CameraOptions) {
    // set camera options

    if options.exposure.is_some()
        && !options
            .auto_exposure
            .clone()
            .unwrap_or(CameraOptionBool {
                value: false,
                default: false,
            })
            .value
    {
        set_camera_property(
            ctx,
            stream,
            openpnp_sys::CAPPROPID_EXPOSURE,
            options.exposure.unwrap().value,
        );
    }

    if options.gain.is_some() {
        set_camera_property(
            ctx,
            stream,
            openpnp_sys::CAPPROPID_GAIN,
            options.gain.unwrap().value,
        );
    }

    if options.brightness.is_some() {
        set_camera_property(
            ctx,
            stream,
            openpnp_sys::CAPPROPID_BRIGHTNESS,
            options.brightness.unwrap().value,
        );
    }

    if options.contrast.is_some() {
        set_camera_property(
            ctx,
            stream,
            openpnp_sys::CAPPROPID_CONTRAST,
            options.contrast.unwrap().value,
        );
    }

    if options.saturation.is_some() {
        set_camera_property(
            ctx,
            stream,
            openpnp_sys::CAPPROPID_SATURATION,
            options.saturation.unwrap().value,
        );
    }

    if options.sharpness.is_some() {
        set_camera_property(
            ctx,
            stream,
            openpnp_sys::CAPPROPID_SHARPNESS,
            options.sharpness.unwrap().value,
        );
    }

    if options.auto_exposure.is_some() {
        set_camera_auto_property(
            ctx,
            stream,
            openpnp_sys::CAPPROPID_EXPOSURE,
            options.auto_exposure.unwrap().value,
        );
    }

    if options.hue.is_some() {
        set_camera_property(
            ctx,
            stream,
            openpnp_sys::CAPPROPID_HUE,
            options.hue.unwrap().value,
        );
    }

    if options.gamma.is_some() {
        set_camera_property(
            ctx,
            stream,
            openpnp_sys::CAPPROPID_GAMMA,
            options.gamma.unwrap().value,
        );
    }

    if options.white_balance.is_some() {
        set_camera_property(
            ctx,
            stream,
            openpnp_sys::CAPPROPID_WHITEBALANCE,
            options.white_balance.unwrap().value,
        );
    }

    if options.backlight_compensation.is_some() {
        set_camera_property(
            ctx,
            stream,
            openpnp_sys::CAPPROPID_BACKLIGHTCOMP,
            options.backlight_compensation.unwrap().value,
        );
    }

    if options.auto_white_balance.is_some() {
        set_camera_auto_property(
            ctx,
            stream,
            openpnp_sys::CAPPROPID_WHITEBALANCE,
            options.auto_white_balance.unwrap().value,
        );
    }
}