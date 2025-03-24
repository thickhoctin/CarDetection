<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

# CAR DETECTION PROJECT

<em></em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/thickhoctin/CarDetection?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/thickhoctin/CarDetection?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/thickhoctin/CarDetection?style=default&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/thickhoctin/CarDetection?style=default&color=0080ff" alt="repo-language-count">

<!-- default option, no dependency badges. -->


<!-- default option, no dependency badges. -->

</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview
This project implements a computer vision system for vehicle detection, tracking, and speed estimation using YOLO (You Only Look Once) and DeepSORT (Deep Simple Online and Realtime Tracking). The system processes video footage of traffic scenarios to:

Detect vehicles using a YOLO model
Track their movement across frames using DeepSORT
Estimate vehicle speeds in kilometers per hour
Generate output video with annotated tracking information

---

## Technical Components

### Object Detection
- **Model**: The project uses YOLOv8/YOLOv11 (with fallback mechanisms), a state-of-the-art object detection algorithm
- **Implementation**: Leverages the Ultralytics YOLO implementation for efficient inference
- **GPU Acceleration**: Includes CUDA support detection for hardware acceleration

### Object Tracking
- **Algorithm**: Implements DeepSORT, which combines appearance features with motion information
- **Feature Extraction**: Uses MobileNet as an embedding network for appearance features
- **Track Management**: Handles track creation, updating, and deletion with configurable parameters (max age, NN budget)

### Speed Estimation
- **Method**: Calculates speed based on pixel displacement between frames
- **Conversion**: Applies scaling factor to convert from pixels/second to km/h
- **Visualization**: Color-codes vehicles based on estimated speed (green, orange, red)

### Video Processing
- **Input/Output**: Handles video file reading and writing with identical parameters
- **Progress Tracking**: Provides processing progress indicators during execution
- **Real-time Display**: Shows processed frames in real-time while generating output
---

## Project Structure

```sh
└── CarDetection/
    ├── CarDetection.py
    ├── Images
    │   └── 1.jpeg
    ├── LICENSE
    ├── README.md
    ├── YOLO-Basic.py
    ├── YOLO-Webcam.py
    ├── main.py
    ├── requirements.txt
    └── yolo_classes.json
```

### Project Index

<details open>
	<summary><b><code>CARDETECTION/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/thickhoctin/CarDetection/blob/master/LICENSE'>LICENSE</a></b></td>
					<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/thickhoctin/CarDetection/blob/master/CarDetection.py'>CarDetection.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/thickhoctin/CarDetection/blob/master/YOLO-Basic.py'>YOLO-Basic.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/thickhoctin/CarDetection/blob/master/YOLO-Webcam.py'>YOLO-Webcam.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/thickhoctin/CarDetection/blob/master/yolo_classes.json'>yolo_classes.json</a></b></td>
					<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/thickhoctin/CarDetection/blob/master/main.py'>main.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/thickhoctin/CarDetection/blob/master/requirements.txt'>requirements.txt</a></b></td>
					<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Pip

### Installation

Build CarDetection from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone https://github.com/thickhoctin/CarDetection
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd CarDetection
    ```

3. **Install the dependencies:**

<!-- SHIELDS BADGE CURRENTLY DISABLED -->
	<!-- [![pip][pip-shield]][pip-link] -->
	<!-- REFERENCE LINKS -->
	<!-- [pip-shield]: https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white -->
	<!-- [pip-link]: https://pypi.org/project/pip/ -->

	**Using [pip](https://pypi.org/project/pip/):**

	```sh
	❯ pip install -r requirements.txt
	```

### Usage

Run the project with:

**Using [pip](https://pypi.org/project/pip/):**
```sh
python CarDetection.py
```

### Testing

Cardetection uses the {__test_framework__} test framework. Run the test suite with:

**Using [pip](https://pypi.org/project/pip/):**
```sh
pytest
```

---

## Roadmap

- [X] **`Task 1`**: Implement YOLO object detection
- [X] **`Task 2`**: Implement DeepSort for Object tracking.
- [X] **`Task 3`**: Implement feature to show velocity and object ID.
- [X] **`Task 4`**: Implement feature to rendering output for file.

---

## Contributing

- **💬 [Join the Discussions](https://github.com/thickhoctin/CarDetection/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/thickhoctin/CarDetection/issues)**: Submit bugs found or log feature requests for the `CarDetection` project.
- **💡 [Submit Pull Requests](https://github.com/thickhoctin/CarDetection/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/thickhoctin/CarDetection
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/thickhoctin/CarDetection/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=thickhoctin/CarDetection">
   </a>
</p>
</details>

---

## License

Cardetection is protected under the [MIT LICENSE](https://choosealicense.com/licenses/mit) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/mit/) file.

---

## Acknowledgments

- Credit `contributors`, `inspiration`, `references`, etc.

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---
