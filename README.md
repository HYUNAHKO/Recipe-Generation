# 🍳 방구석 흑백요리사 (Armchair Chef)

## Introduction
"Armchair Chef" leverages AI technologies like image detection and knowledge graphs to recommend high-quality recipes based on limited ingredients. This project reminds us data collection is crucial part of making advanced AI. 

## Pipeline
![Pipeline](/image/pipeline.jpg)

## Features
### Image Detection
Utilizes YOLOv5 for recognizing ingredients from images, with fine-tuning performed on a specialized dataset.
![Image Detection](/path/to/image_detection_gif.gif)

### Knowledge Graphs with Neo4j
Manages complex relationships between ingredients and recipes, enhancing the recommendation system.
![Neo4j Graph](/path/to/graph_screenshot.png)

### KAPING for Recipe Recommendation
Integrates knowledge and context to provide tailored recipe suggestions.
![KAPING Flow](/path/to/kaping_flow_diagram.png)

### Prompt Engineering
Generates detailed and context-aware prompts for recipe generation.
![Prompt Engineering](/path/to/prompt_engineering_screenshot.png)

## Model Training and Data
The fine-tuning of the YOLOv5 model was conducted using a custom dataset available on Hugging Face:
- **Model**: [YOLOv5 Fine-tuned Model on Hugging Face](https://huggingface.co/yourusername/yolov5-finetuned)
- **Dataset**: [Labeled Image Data on Hugging Face](https://huggingface.co/datasets/yourusername/yourdataset)

## Pipeline Overview
Our project follows a comprehensive pipeline from data gathering to final recipe suggestions:
1. **Dataset Preparation**
2. **Image Detection with YOLOv5**
3. **Knowledge Graph Construction**
4. **KAPING Integration**
5. **Prompt Engineering**
6. **Result Compilation**

![Pipeline Diagram](/path/to/pipeline_diagram.png)

## Citations
This project utilizes several external libraries and models:
- **KAPING**: @article{yourKapingCitationHere, title={Title of KAPING Paper}, author={Author Names}, journal={Journal Name}, year={Year} }
- **CompVis Stable Diffusion**: @article{Rombach2022HighResolutionIS, title={High-Resolution Image Synthesis with Latent Diffusion Models}, author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer}, journal={ArXiv}, year={2022}, volume={abs/2112.10752} }

## Getting Started
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourgithubrepo/Recipe-Generation.git
pip install -r requirements.txt
