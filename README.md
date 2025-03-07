# 🍳 방구석 흑백요리사 (Armchair Chef) - YAICON 5th

## Introduction
"Armchair Chef" leverages AI technologies like image detection and knowledge graphs to recommend high-quality recipes based on limited ingredients. This project reminds us data collection is crucial part of making advanced AI. 

## Related Works
- **Image Detection**:
  - We utilize YOLOv5 image detection model to identify ingredients from user-uploaded photos. Our approach is based on fine-tuning the YOLOv5m model, which leverages its excellent compatibility with our server infrastructure(Elice Company) and its proven effectiveness in training. It provides robust and efficient detection of ingredients from user-uploaded photos, which is essential for accurately identifying the components of complex dishes. This fine-tuning process not only enhances the model's accuracy but also tailors it specifically for culinary applications, significantly improving our system's ability to provide precise recipe suggestions and dietary insights based on the identified ingredients. This ensures a seamless and intuitive user experience, making our application a valuable tool for culinary enthusiasts and health-conscious individuals alike.

- **Retrieval-Augmented Generation (RAG)**:
  - Our project integrates the RAG model for enhancing recipe recommendations. This method combines the robustness of retrieval-based techniques with the generative capabilities of a transformer. By leveraging indexed data for immediate lookup and contextual relevance, RAG significantly improves the specificity and quality of recipe suggestions based on user inputs and detected ingredients.

- **Text Generation**:
  - For generating customized recipes and culinary instructions, we employ advanced text generation models. These models are fine-tuned on a diverse dataset of recipes, enabling them to generate creative and contextually relevant culinary directions tailored to the user's dietary preferences and available ingredients.

For further details on each component, refer to the [YOLOv5 official repository](https://github.com/ultralytics/yolov5), [Hugging Face's RAG documentation](https://huggingface.co/docs/transformers/model_doc/rag), and the [CompVis/stable-diffusion GitHub page](https://github.com/CompVis/stable-diffusion) for insights into the cutting-edge image synthesis used in our project.
  
## Pipeline
![Pipeline](/image/pipeline.jpg)

## Pipeline Overview
Our project follows a comprehensive pipeline from data gathering to final recipe suggestions:
1. **Dataset Preparation**
2. **Image Detection with YOLOv5**
3. **Knowledge Graph Construction**
4. **KAPING Integration**
5. **Prompt Engineering**
6. **Result Compilation**
   
## Features
### Image Detection
Utilizes YOLOv5m for recognizing ingredients from images, with fine-tuning performed on a specialized dataset.
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
- **Model**: [YOLOv5 Fine-tuned Model on Hugging Face](https://huggingface.co/HYUNAHKO/Ingredients_object_detection)
- **Dataset**: [Labeled Image Data on Hugging Face](https://huggingface.co/datasets/HYUNAHKO/ingredients_dataset)


## Citations
This project utilizes several external libraries and models:
- **KAPING**: @article{baek2023knowledge,
  title={Knowledge-augmented language model prompting for zero-shot knowledge graph question answering},
  author={Baek, Jinheon and Aji, Alham Fikri and Saffari, Amir},
  journal={arXiv preprint arXiv:2306.04136},
  year={2023}
}
- **CompVis Stable Diffusion**: @article{Rombach2022HighResolutionIS, title={High-Resolution Image Synthesis with Latent Diffusion Models}, author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer}, journal={ArXiv}, year={2022}, volume={abs/2112.10752} }

## Getting Started
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourgithubrepo/Recipe-Generation.git
pip install -r requirements.txt
