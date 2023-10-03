# facetracking-rs // Facetracking in Rust

`facetracking-rs` is a Rust library that provides a high-level API for face detection and tracking. It uses `onnxruntime` through the `ort` crate to run deep learning models. 

:construction: This is very much a work in progress. :construction:

> [!IMPORTANT] 
> On macOS, this currently needs to be compiled using Rust nightly. This is due to the usage of unstable standard library features in a workaround.

### Features
- [x] :woman: Face detection and tracking
- [x] :pushpin: 2D face landmarks
- [ ] 🌐 3D (metric) face landmarks
- [ ] :eyes: Eye gaze estimation

### Supported platforms

| Platform      | Status        | DNN backends   |
| ------------- | ------------- | ------------- |
| MacOS (Apple Silicon)         | :heavy_check_mark: | CoreML, CPU |
| MacOS (Intel)         | untested  |  |
| Linux (x86_64)         | untested  | |
| Windows (x86_64)         | :heavy_check_mark:  | CUDA, oneDNN, CPU |
| Browser (WASM)         | untested | |

### Models
| Type      | Model      | Capabilities        |
| ------------- | ------------- | ------------- |
| Face detection | BlazeFace         |  Fast face detection, mainly used for providing initial face bounding boxes for other models |
| Face inference | MediaPipe Face Mesh v2         |  468 face landmarks in 2D (screen) space and 3D (metric) space, custom eye gaze estimation |
