# Offside_Detection_System

#  Offside Detector with AI

Artificial Vision System that automatically detects offsides in football, it detects the players using YOLO and divides the players into different teams using CLIP.

##  Description

This porject implements a complete computer vision pipeline that:

1. **Detects Players** in images using a retrained YOLOv11 model
2. **Classifies teams** automatically through CLIP (Vision Transformer)
3. **Calculates Homography** to project image positions onto real coordinates in the field
4. **Allows manual selection** of the player marking the offside line
5. **Automatically determines** which players are in an offside position
6. **Generate visualizations** and quantitative metrics from the analysis

##  Features

- Robust player detection with custom YOLO model
- Automatic team classification without prior training (CLIP)
- Field-to-image perspective transformation using homography
- Automatic goalkeeper identification by position
- Offside analysis with interactive selection
- Quantitative metrics: histograms, distributions, heatmaps
- Adaptive image preprocessing

## Technologies Used

- **YOLOv11 Retrained** (Ultralytics) - Object detection (If the retrained model is not detected, the program automatically uses YOLOv8)
- **CLIP** (OpenAI) - Classification of teams by jersey color
- **OpenCV** - Image processing and homography
- **scikit-learn** - K-means clustering
- **PyTorch** - CLIP backend
- **Matplotlib/Seaborn** - Visualizations and metrics

## Requirements
```bash
ultralytics
opencv-python-headless
scikit-learn
pillow
matplotlib
seaborn
torch
clip (OpenAI)
numpy
```

## How tu run

### In Google Colab (Recommended)

1. **Upload your trained model** `best.pt` to `/content/`

2. **Copy the entire code** into a Colab cell

3. **Run the cell** - The system will ask you to:
   - Upload images of the match
   - Select the player marking the offside line

4. **View the results** - The system will display:
   - Preprocessed image
   - Detected and numbered players
   - Team classification
   - Detection confidence histogram
   - Team distribution
   - Dominant jersey colors
   - Final result with offside line

## Processing Pipeline
```
1. Preprocessing
   └─> Adaptive image quality enhancement

2. Homography
   └─> Calculation of field-image perspective transformation

3. YOLO detection
   └─> Identification of all players
   └─> Generation of confidence histogram

4. Initial visualization
   └─> Unclassified numbered players

5. CLIP classification
   └─> Automatic grouping into teams
   └─> Distribution and balance analysis
   └─> Visualization of dominant colors

6. Position Calculation
   └─> Projection to actual field coordinates
   └─> Identification of goalkeepers
   └─> Generation of position heatmap

7. Manual selection
   └─> User indicates player marking the offside line

8. Offside analysis
   └─> Automatic determination of players in an advanced position
   └─> Analysis by direction of attack

9. Final result
   └─> Complete visualization with offside line
   └─> Statistics and final status
```

## Color Code

- **Green** - Team 0
- **Red** - Team 1
- **Orange** - Player marking the offside line
- **Magenta** - Players in an offside position
- **Gray** - Unclassified players

## Generated Metrics

1. **Confusion Matrix** - Confusion Matrix showing the precision of the model.
2. **Accuracy** - Show the accuracy of the model with a specific example.

## Configuration

### Adjust detection confidence

To adjust confidence, simply change the value

```python
persons = detector.detect(frame, conf=0.25)  
```

### Modify field dimensions
```python
homography = RobustHomography(field_length=105.0, field_width=68.0)
```

### Change YOLO model
```python
detector = PlayerDetector(model_path=“/path/to/your/model.pt”)
```
## Notes

- The `best.pt` model is a retrained version of YOLO11 used for this program
- The classes detected depend on how the model used was trained
- Homography works best with images from high angles
- CLIP classification works best with T-shirts in contrasting colors

## Ablation Studies and Preliminary Experiments

Before implementing the final version of the Offside Detection System, several ablation experiments were conducted to evaluate the impact of different detection and team-classification strategies. These tests provided key insights into the limitations of simpler approaches and guided the design of the full pipeline.

1. **YOLOv8-Only Detection** (No Team Classification)

The initial prototype relied exclusively on a YOLOv8 model for player detection without any additional team-classification mechanism. While YOLOv8 delivered strong detection accuracy and performed well in standard scenarios, it exhibited a consistent limitation:
- Frequent confusion between players from different teams, as the model does not infer team identity from visual appearance.
- Incorrect grouping of players, which significantly affected the reliability of offside calculations, especially in crowded scenes or when jersey colors were similar.

This experiment demonstrated that robust offside analysis requires an explicit and automated team-classification step beyond object detection alone.

2. **YOLOv11  + Manual Team Assignment**

A second experiment used YOLOv11 in its base, non-retrained version, combined with a manual procedure in which the user was required to assign each detected player to a team. This configuration produced moderately improved results:
- YOLOv11 yielded more stable detections and better localization than YOLOv8 in some cases.
- Manual team assignment reduced team-mixing errors, producing cleaner team separation.

However, this approach introduced a critical usability drawback:
- Manual labeling was slow, repetitive, and impractical, especially for sequences with many players or multiple frames.

These limitations highlighted the need for an automated solution that combines high-quality detection with reliable and fully autonomous team classification.




