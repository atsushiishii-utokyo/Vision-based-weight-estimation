# Vision-based modal identification and weight estimation of vehicles

This repository implements a **vision-based** vehicle weight estimatation using:
- Vision-based wheel & body tracking
- Half-car dynamic modeling
- SRIM-based modal identification
- Weight estimation using complex matrix composition
---
This work was published in journal of **Computer-Aided Civil and Infrastructure Engineering**

📄 **Paper:** [Vision-based modal identification and weight estimation of vehicles](https://www.sciencedirect.com/science/article/pii/S1093968726018050)

## 🚀 Features

### ✔ 1. Tracking
- YOLOv5 detection
- Subpixel wheel center localization
- Body + wheels response extraction
- Background shift correction

Example of tracking:
![alt text](tracking.gif)

This gives you vertical displacement as below:
![alt text](examples/plots/image-15.png)


### ✔ 2. Weight Estimation
- SRIM-based modal parameter extraction
- Filtering
- Complex matrix formulation in state-space using modal parameters

Using SRIM and complex matrix formulation, vehicle body weight can be estimated. System order is a hyper-parameter to select the number of orders in SRIM. If the weight estimation plot (c) is consistent across this value, it means the estimation is stable and reliable.

 In the below case, the weight is estimated as 0.95 ton.
 Natural frequency, damping ratio, and moment of inertia are also estimated.

![alt text](examples/plots/image-19.png)

### ✔ Half-Car Simulation
It can simulate the vehicle behaiors using half-car modeling.
Please run [this notebook](notebooks/half_car_simulation.ipynb).

In the notebook, you can simulate:

- Vertical bounce + pitch dynamics are simulated from random road profile input excitation.
- Weight estimation method is applied to the simulated vehicle responses,

## 📘 Notebooks Included
#### 1. notebooks/half_car_simulation.ipynb

Includes:

- Half-car dynamic equations

- Road input generation

- Simulated displacement & pitch

- Comparison with experimental tracking

Useful for verifying:

- extracted modal frequencies

- damping ratio

- suspension parameters

- regression stability

#### 2. notebooks/tracking_and_weight_estimation.ipynb

Includes:

- Using real world videos

- Tracking points of vehicle

- Weight estimation

- Video generation of point tracking

## 🔧 Installation
Clone the repository

```
git clone https://github.com/atsushiishii-utokyo/Vision-based-weight-estimation.git
```

## ▶️ How to Run (Command Line)

This repository provides a command-line entry point via `main.py` to run **tracking → response extraction → weight estimation** on a real video.

### 1️⃣ Basic usage
Run the command below in the console.
```bash
python main.py path/to/video.mp4
```

Example:

```bash
python main.py videos/case-1.mov
```

This will:

- Track vehicle body and wheels from the video
- Extract vertical displacement responses
- Perform SRIM-based modal identification
- Estimate vehicle weight, inertia, and CG location
- Plot SRIM diagnostics if enabled

2️⃣ Optional arguments

You can customize the behavior using CLI options:

```bash
python main.py path/to/video.mp4 \
    --fs 120 \
    --p 40 \
    --no-subpixel \
    --plot
```

| Argument        | Description                 | Default      |
| --------------- | --------------------------- | ------------ |
| `video`         | Path to input video         | **required** |
| `--fs`          | Sampling frequency [Hz]. <br> Need to set the correct number. Use the exact frame rate as recording.     | `120`        |
| `--p`           | SRIM block size             | `40`         |
| `--no-subpixel` | Disable subpixel processing | disabled      |
| `--plot`        | Show SRIM diagnostic plots  | disabled     |

Example with plotting enabled:

```
python3 main.py videos/case-1.mov --plot
```

3️⃣ Output

The script prints estimated values to the console, for example:
```
Estimated weight: 0.95 ton
Estimated pitch inertia: 1.42
Estimated CG ratio: 0.48
Natural frequencies: [1.23, 2.87] Hz
```

If --plot is enabled, SRIM diagnostics such as:

- natural frequencies
- damping ratios
- EMAC
- mass / inertia consistency are displayed.

📚 Citation

If you use this repository, please cite the following paper:
[Vision-based modal identification and weight estimation of vehicles](https://www.sciencedirect.com/science/article/pii/S1093968726018050)
