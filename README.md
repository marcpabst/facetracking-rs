# 👩 facetracking-rs
**Fast and accurate facetracking in Rust.**

[![build](https://github.com/marcpabst/facetracking-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/marcpabst/facetracking-rs/actions/workflows/rust.yml)


`facetracking-rs` is a Rust library that provides a high-level API for face detection and tracking. It uses `onnxruntime` through the `ort` crate to run deep learning models. 

> [!WARNING] 
>  This is very much a work in progress. Use with care!

> [!IMPORTANT] 
> On macOS, this currently needs to be compiled using Rust nightly. This is due to the usage of unstable standard library features.


### Usage
This projects consists of two crates: `facetracking` and `facetracking-webcam`. The former provides the core functionality, while the latter provides an advanced GUI application that can enables tracking facial landmarks in real-time using a webcam.

It is also planned to provide Python bindings for this library.

### Features
- [x] :woman: Face detection and tracking
- [x] :pushpin: 2D face landmarks
- [x] :camera: Webcam GUI application
- [ ] 🌐 3D (metric) face landmarks
- [ ] :computer: Head pose estimation
- [ ] :eyes: Eye gaze estimation
- [ ] :snake: Python bindings


### Supported platforms

| Platform      | Status        | DNN backends   |
| ------------- | ------------- | ------------- |
| MacOS (Apple Silicon)         | :heavy_check_mark: | CoreML, CPU |
| MacOS (Intel)         | untested  | CPU |
| Linux (x86_64)         | untested  | CPU |
| Windows (x86_64)         | :heavy_check_mark:  | CPU |
| Browser (WASM)         | planned | |

### Models
| Type      | Model      | Capabilities        |
| ------------- | ------------- | ------------- |
| Face detection | BlazeFace         |  Fast face detection, mainly used for providing initial face bounding boxes for other models |
| Face inference | MediaPipe Face Mesh v2         |  468 face landmarks in 2D (screen) space and 3D (metric) space, custom eye gaze estimation |
