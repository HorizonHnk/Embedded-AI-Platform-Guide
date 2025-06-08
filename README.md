# 🧠 Embedded AI Platform Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-ESP32%20%7C%20Arduino%20%7C%20RaspberryPi%20%7C%20STM32-blue)](https://github.com)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-orange)](https://www.tensorflow.org/lite)
[![Edge Impulse](https://img.shields.io/badge/Edge-Impulse-green)](https://edgeimpulse.com)
[![Contributors Welcome](https://img.shields.io/badge/Contributors-Welcome-brightgreen)](CONTRIBUTING.md)

> **Comprehensive guide and examples for deploying AI/ML models on microcontrollers and embedded systems**

A complete resource for implementing artificial intelligence on resource-constrained embedded platforms including ESP32, Arduino, Raspberry Pi, and STM32 microcontrollers. This repository provides practical examples, optimization techniques, and deployment strategies for running machine learning models on edge devices.

---

## 📚 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🎯 Supported Platforms](#-supported-platforms)
- [⚡ Features](#-features)
- [🛠️ Installation](#️-installation)
- [📖 Platform Guides](#-platform-guides)
- [🔧 Model Optimization](#-model-optimization)
- [📊 Performance Benchmarks](#-performance-benchmarks)
- [🏗️ Project Examples](#️-project-examples)
- [📝 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🚀 Quick Start

Get up and running with embedded AI in under 5 minutes:

```bash
git clone https://github.com/yourusername/embedded-ai-guide.git
cd embedded-ai-guide
```

### Choose Your Platform:

| Platform | Memory | AI Frameworks | Best For |
|----------|--------|---------------|----------|
| **ESP32-S3** | 512KB + 8MB PSRAM | TensorFlow Lite, ESP-DL | IoT + AI applications |
| **Arduino Nano 33 BLE** | 256KB RAM | TensorFlow Lite, AIfES | Ultra-low power sensors |
| **Raspberry Pi 5** | 4-8GB RAM | Full TensorFlow, PyTorch | Computer vision, edge computing |
| **STM32N6** | 4.2MB RAM + NPU | STM32Cube.AI | Industrial, automotive |

---

## 🎯 Supported Platforms

### 🔌 ESP32 Family
- **ESP32-S3** - *AIoT optimized with vector instructions*
- **ESP32-CAM** - *Computer vision applications*
- **ESP32-C3** - *Low-cost RISC-V alternative*

### 🤖 Arduino Ecosystem
- **Arduino Nano 33 BLE Sense** - *Rich sensor suite for TinyML*
- **Arduino Portenta H7** - *Industrial-grade dual-core performance*
- **Arduino Nano RP2040 Connect** - *Raspberry Pi silicon on Arduino*

### 🍓 Raspberry Pi Series
- **Raspberry Pi 5** - *Latest with AI Kit support*
- **Raspberry Pi 4** - *Proven platform with extensive ecosystem*
- **Raspberry Pi Zero 2 W** - *Compact form factor*

### ⚙️ STM32 Microcontrollers
- **STM32N6 Series** - *Dedicated NPU acceleration*
- **STM32H7 Series** - *High-performance Cortex-M7*
- **STM32L4+ Series** - *Ultra-low power AI*

---

## ⚡ Features

### 🔥 Core Capabilities
- **Real-time Inference** - Sub-millisecond response times
- **Model Optimization** - 90% memory reduction techniques
- **Cross-Platform** - Unified development workflow
- **Edge-First Design** - No cloud dependency required

### 🛡️ Advanced Features
- **Quantization Support** - 8-bit, 16-bit, and custom precision
- **Model Pruning** - Automated parameter reduction
- **Hardware Acceleration** - NPU and vector instruction support
- **OTA Updates** - Remote model deployment

### 📱 Application Areas
- **Computer Vision** - Object detection, image classification
- **Audio Processing** - Voice recognition, sound classification
- **Sensor Fusion** - Multi-modal data processing
- **Predictive Maintenance** - Industrial monitoring systems

---

## 🛠️ Installation

### Prerequisites

```bash
# Python environment
python3 -m venv embedded-ai
source embedded-ai/bin/activate  # On Windows: embedded-ai\Scripts\activate
pip install -r requirements.txt
```

### Platform-Specific Setup

#### ESP32 Development
```bash
# Install ESP-IDF
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh
source export.sh
```

#### Arduino IDE Configuration
```bash
# Install Arduino CLI (optional)
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
arduino-cli core install arduino:mbed_nano
```

#### Raspberry Pi Setup
```bash
# Enable AI Kit (if available)
sudo raspi-config
# Advanced Options > SPI > Enable
sudo reboot
```

#### STM32CubeIDE
> Download from [STMicroelectronics official website](https://www.st.com/en/development-tools/stm32cubeide.html)

---

## 📖 Platform Guides

### 🔌 ESP32-S3 AI Development

**Key Advantages:**
- 512KB internal SRAM + 8MB external PSRAM support
- Vector instructions deliver 6.25x performance improvement
- Built-in wireless connectivity (Wi-Fi + Bluetooth)
- Optimized ESP-DL library for computer vision

**Getting Started:**
```cpp
#include "esp_log.h"
#include "dl_lib_matrix3d.h"
#include "human_face_detect_msr01.hpp"

void app_main() {
    // Initialize camera and AI model
    camera_init();
    human_face_detect_msr01_init();
    
    while(1) {
        // Capture frame and run inference
        camera_fb_t *fb = esp_camera_fb_get();
        std::list<dl::detect::result_t> results = 
            human_face_detect_msr01(fb->buf, fb->height, fb->width);
        
        ESP_LOGI("AI", "Detected %d faces", results.size());
        esp_camera_fb_return(fb);
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }
}
```

**Performance Metrics:**
- Face detection: ~700ms per frame
- Voice command recognition: ~200ms
- Sensor classification: ~4ms

### 🤖 Arduino Nano 33 BLE Sense

**Key Advantages:**
- Comprehensive sensor suite (IMU, microphone, environmental)
- TinyML optimized with 256KB RAM
- Ultra-low power consumption (5-15mA during inference)
- AIfES framework enables on-device training

**Example Implementation:**
```cpp
#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>

// Model and tensor arena
#include "gesture_model.h"
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
    Serial.begin(9600);
    IMU.begin();
    
    // Initialize TensorFlow Lite
    model = tflite::GetModel(gesture_model);
    interpreter = new tflite::MicroInterpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
    interpreter->AllocateTensors();
}

void loop() {
    // Read sensor data
    float ax, ay, az;
    if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(ax, ay, az);
        
        // Run inference
        float* input = interpreter->input(0)->data.f;
        input[0] = ax; input[1] = ay; input[2] = az;
        
        interpreter->Invoke();
        
        // Get prediction
        float* output = interpreter->output(0)->data.f;
        int predicted_gesture = argmax(output, NUM_CLASSES);
        
        Serial.print("Gesture: "); Serial.println(predicted_gesture);
    }
    delay(50);
}
```

### 🍓 Raspberry Pi 5 with AI Kit

**Key Advantages:**
- Quad-core ARM Cortex-A76 at 2.4GHz
- Hailo-8L NPU delivering 13 TOPS performance
- Full Linux environment with complete AI frameworks
- 5.8x faster inference compared to CPU-only

**Computer Vision Example:**
```python
import cv2
import numpy as np
from hailo_platform import HailoRT

# Initialize Hailo AI accelerator
device = HailoRT.create_device()
network_group = device.configure("yolov8s.har")

def detect_objects(frame):
    # Preprocess frame
    input_frame = cv2.resize(frame, (640, 640))
    input_frame = input_frame.astype(np.float32) / 255.0
    
    # Run inference on Hailo NPU
    results = network_group.infer(input_frame)
    
    # Post-process results
    boxes, scores, class_ids = post_process_yolo(results)
    
    return boxes, scores, class_ids

# Main processing loop
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        boxes, scores, class_ids = detect_objects(frame)
        
        # Draw bounding boxes
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score > 0.5:
                cv2.rectangle(frame, box[:2], box[2:], (0, 255, 0), 2)
                cv2.putText(frame, f"Class {class_id}: {score:.2f}", 
                           box[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("AI Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

### ⚙️ STM32N6 with NPU Acceleration

**Key Advantages:**
- Dedicated Neural-ART Accelerator NPU (600 GOPS)
- 4.2MB contiguous embedded RAM
- STM32Cube.AI ecosystem integration
- Industrial temperature range and safety certifications

**STM32CubeIDE Workflow:**
1. **Model Import** - Load TensorFlow Lite, ONNX, or Keras models
2. **Optimization** - Automatic quantization and pruning
3. **Code Generation** - C code with HAL integration
4. **Validation** - On-target performance verification

---

## 🔧 Model Optimization

### Quantization Strategies

> **8-bit quantization typically reduces model size by 75% with minimal accuracy loss**

#### Post-Training Quantization
```python
import tensorflow as tf

# Convert to TensorFlow Lite with quantization
converter = tf.lite.TFLiteConverter.from_saved_model("model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

# For full integer quantization
def representative_dataset():
    for _ in range(100):
        yield [np.random.random((1, 224, 224, 3)).astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
```

#### Quantization-Aware Training
```python
import tensorflow_model_optimization as tfmot

# Define quantization config
quantize_config = tfmot.quantization.keras.QuantizeConfig()

# Apply to model
q_aware_model = tfmot.quantization.keras.quantize_model(
    model, quantize_config)

# Continue training with quantization simulation
q_aware_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

q_aware_model.fit(train_data, epochs=1)
```

### Model Pruning

#### Magnitude-Based Pruning
```python
import tensorflow_model_optimization as tfmot

# Define pruning schedule
pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=0.9,
    begin_step=2000,
    end_step=10000)

# Apply pruning
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
    model, pruning_schedule=pruning_schedule)

# Training loop with pruning
pruned_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Add pruning callbacks
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
pruned_model.fit(train_data, epochs=10, callbacks=callbacks)
```

### Memory Optimization Techniques

#### Static Memory Allocation (Arduino/ESP32)
```cpp
// Avoid dynamic allocation
#define TENSOR_ARENA_SIZE 30000
static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

// Use PROGMEM for constants
const uint8_t model_data[] PROGMEM = {
    // Model weights stored in Flash
};

// Circular buffer for sensor data
#define BUFFER_SIZE 128
static float sensor_buffer[BUFFER_SIZE];
static uint16_t buffer_index = 0;
```

#### External Memory Management (ESP32)
```cpp
// Allocate large buffers in PSRAM
#include "esp_heap_caps.h"

void* psram_buffer = heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
if (psram_buffer == NULL) {
    ESP_LOGE("MEMORY", "Failed to allocate PSRAM");
}

// Monitor memory usage
size_t free_heap = esp_get_free_heap_size();
size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
```

---

## 📊 Performance Benchmarks

### Inference Time Comparison

| Model Type | Arduino Nano 33 | ESP32-S3 | Raspberry Pi 5 | STM32N6 |
|------------|------------------|----------|----------------|---------|
| **Image Classification** | 45ms | 28ms | 3ms | 12ms |
| **Object Detection** | N/A | 700ms | 25ms | 80ms |
| **Audio Classification** | 15ms | 8ms | 2ms | 5ms |
| **Sensor Fusion** | 5ms | 3ms | 1ms | 2ms |

### Memory Usage (Optimized Models)

| Framework | Flash Usage | RAM Usage | Accuracy Loss |
|-----------|-------------|-----------|---------------|
| **TensorFlow Lite** | 50-500KB | 20-200KB | <2% |
| **Edge Impulse EON** | 30-350KB | 15-140KB | <1% |
| **STM32Cube.AI** | 40-400KB | 18-180KB | <1.5% |
| **ESP-DL** | 60-600KB | 25-250KB | <2.5% |

### Power Consumption

| Platform | Active Inference | Deep Sleep | Battery Life* |
|----------|------------------|------------|---------------|
| **Arduino Nano 33** | 15mA | 3µA | 6+ months |
| **ESP32-S3** | 80mA | 100µA | 2-4 weeks |
| **Raspberry Pi 5** | 2000mA | N/A | Continuous power |
| **STM32L4+** | 12mA | 2µA | 8+ months |

> *Estimated with 2000mAh battery, 1 inference per minute

---

## 🏗️ Project Examples

### 1. Smart Security Camera (ESP32-S3)
**Features:**
- Real-time person detection
- Wi-Fi image streaming
- Battery operation with solar charging
- Mobile app notifications

**Key Components:**
- ESP32-S3-CAM module
- PIR motion sensor
- Solar panel + LiPo battery
- Custom mobile app

### 2. Gesture Recognition Wearable (Arduino Nano 33)
**Features:**
- 10 gesture classifications
- Bluetooth Low Energy connectivity
- Weeks of battery life
- Real-time haptic feedback

**Applications:**
- Smart home control
- Assistive technology
- Gaming interfaces
- Industrial control

### 3. Industrial Quality Control (Raspberry Pi 5 + AI Kit)
**Features:**
- Real-time defect detection
- 95%+ accuracy rates
- Integration with existing systems
- Comprehensive logging and analytics

**Performance:**
- 80+ FPS processing
- Sub-10ms decision latency
- 24/7 operation capability
- Remote monitoring dashboard

### 4. Predictive Maintenance System (STM32N6)
**Features:**
- Vibration pattern analysis
- Temperature monitoring
- Wireless sensor networks
- Edge-based anomaly detection

**Benefits:**
- 40% reduction in downtime
- Predictive alerts 2-4 weeks early
- Integration with existing SCADA
- Proven industrial reliability

---

## 📝 Documentation

### 📚 Complete Guides
- [ESP32 AI Development Guide](docs/esp32-guide.md)
- [Arduino TinyML Tutorial](docs/arduino-guide.md)
- [Raspberry Pi AI Kit Setup](docs/raspberry-pi-guide.md)
- [STM32 Industrial AI](docs/stm32-guide.md)

### 🔧 Technical References
- [Model Optimization Techniques](docs/optimization.md)
- [Hardware Selection Guide](docs/hardware-selection.md)
- [Performance Benchmarking](docs/benchmarks.md)
- [Troubleshooting Common Issues](docs/troubleshooting.md)

### 📖 API Documentation
- [TensorFlow Lite Micro API](docs/api/tflite-micro.md)
- [Edge Impulse SDK Reference](docs/api/edge-impulse.md)
- [ESP-DL Library Functions](docs/api/esp-dl.md)
- [STM32Cube.AI Integration](docs/api/stm32-cube-ai.md)

---

## 🤝 Contributing

We welcome contributions from the embedded AI community! Here's how you can help:

### 🐛 Bug Reports
Found an issue? Please create a detailed bug report including:
- Platform and hardware details
- Framework versions
- Steps to reproduce
- Expected vs actual behavior

### ✨ Feature Requests
Have an idea for improvement? We'd love to hear it:
- Describe the use case
- Explain the expected behavior
- Consider implementation complexity

### 🔧 Code Contributions

1. **Fork the repository**
   ```bash
   git fork https://github.com/yourusername/embedded-ai-guide.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-new-feature
   ```

3. **Make your changes**
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Submit a pull request**
   - Describe your changes clearly
   - Reference any related issues
   - Include test results

### 📋 Development Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/embedded-ai-guide.git
cd embedded-ai-guide

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black src/
isort src/
```

### 🏆 Contributors

Thanks to all the amazing contributors who have helped make this project better:

<a href="https://github.com/yourusername/embedded-ai-guide/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=yourusername/embedded-ai-guide" />
</a>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Embedded AI Guide Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🔗 Links and Resources

### 🏢 Official Documentation
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [Edge Impulse Platform](https://edgeimpulse.com/)
- [STM32Cube.AI Ecosystem](https://www.st.com/en/embedded-software/x-cube-ai.html)
- [ESP-DL Library](https://github.com/espressif/esp-dl)

### 🎓 Learning Resources
- [TinyML Course (Harvard)](https://www.edx.org/course/introduction-to-embedded-machine-learning)
- [Embedded AI Specialization (Coursera)](https://www.coursera.org/specializations/embedded-machine-learning)
- [Arduino AI Workshop](https://blog.arduino.cc/category/artificial-intelligence/)

### 🛠️ Development Tools
- [Arduino IDE](https://www.arduino.cc/en/software)
- [ESP-IDF Framework](https://docs.espressif.com/projects/esp-idf/)
- [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html)
- [Visual Studio Code](https://code.visualstudio.com/)

### 🌟 Community
- [TinyML Foundation](https://www.tinyml.org/)
- [Edge Impulse Forum](https://forum.edgeimpulse.com/)
- [Arduino AI Community](https://forum.arduino.cc/c/hardware/nano-family/12)
- [ESP32 AI Discord](https://discord.gg/esp32)

---

<div align="center">

**[⬆ Back to Top](#-embedded-ai-platform-guide)**

Made with ❤️ by the Embedded AI Community

[![Star this repo](https://img.shields.io/github/stars/yourusername/embedded-ai-guide?style=social)](https://github.com/yourusername/embedded-ai-guide)
[![Follow](https://img.shields.io/github/followers/yourusername?style=social)](https://github.com/yourusername)

</div>
