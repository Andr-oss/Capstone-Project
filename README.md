# Rodent Video Tracking Software

## Overview

The **Rodent Video Tracking Software** is a state-of-the-art web application designed to automate the tracking and analysis of rodent behavior in both live and recorded video experiments. Developed to seamlessly integrate into laboratory environments, this software leverages advanced computer vision techniques to deliver an automated, efficient, and highly reliable solution for behavioral studies.

## Key Features

### Real-Time Tracking
- **Multi-Rodent Tracking**: Detect and track the movements of up to four rodents simultaneously in distinct environments.
- **Live & Recorded Video Support**: Compatible with both live video feeds and pre-recorded footage.

### Behavior Identification
- **Automated Behavior Detection**: Identifies behaviors such as staying still or nose poking.
- **Expandable Behavior List**: Dynamically add new behaviors based on evolving research needs.

### Data Analysis
- **Comprehensive Metrics**: Provides detailed metrics for in-depth analysis.
- **Comparison Tools**: Supports side-by-side comparisons between manual scoring and system-generated results.

### Visualization
- **Trajectory Visualizations**: Visualize rodent movement paths for better understanding.
- **Heatmaps**: Generate heatmaps to analyze movement patterns and hotspots.

### User Management
- **Role-Based Access**: Supports multiple user roles including Admin, Researcher, and Lab Technician, each with specific responsibilities and permissions.

## Technologies

### Programming Languages
- **Python**

### Framework
- **Django**

### Libraries
- **OpenCV**: For real-time image processing.

### Database
- **Oracle**

### Operating Systems
- **Windows 10 and above**
- **Ubuntu Linux**

## Purpose

The **Rodent Video Tracking Software** aims to enhance the reproducibility and reliability of rodent behavioral experiments by replacing manual tracking methods with a fully automated solution. This system not only increases the accuracy of behavioral analysis but also significantly reduces the time and effort involved in traditional methods.

## Audience

This software is primarily designed for PennState Health Department as Capstone project.

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Django 3.1 or higher
- OpenCV 4.5 or higher
- Oracle Database

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/rodent-video-tracking.git
   cd rodent-video-tracking
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Database Configuration**:
   - Update the `settings.py` file with your Oracle database credentials.

5. **Run Migrations**:
   ```bash
   python manage.py migrate
   ```

6. **Start the Server**:
   ```bash
   python manage.py runserver
   ```

7. **Access the Application**:
   - Open your browser and navigate to `http://127.0.0.1:8000/`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please contact us at ybc5276@psu.edu(Frontend) or mqa5988@psu.edu or aja7182@psu.edu.

---

**Thank you for using the Rodent Video Tracking Software!**
