# About the project

MSc Computing Group Project - An Explainable AI (XAI) Interface for Multimodal Classification Models, Department of Computing, Imperial College London.

This project aims to explore ways to explain predicition decisions by *Multimodal* machine Learning models and build a web application for users to interact with. A machine learning task is *multimodal* if the input to the model has multiple data forms, e.g. both image and text. Given a user model and an input example, this software will provide interpretations for the model’s behaviour. The project started out by targeting classification problems, specifically identifying whether memes are hateful or not. The underlying algorithms are built on popular model-agnostic XAI methods, and are packaged up as a library named MMXAI. We welcome contributors to extend it to more ML tasks / more explanation methods!

<!-- The below links need changing if porting to gitlab / github README page -->
Interested? play with the [interface](/ ":ignore") yourself!

Meet the [team](/about) behind this.
# About the repository

The repository contains the code for the following components:
- Implementations of XAI methods [Lime](https://github.com/marcotcr/lime), [Shap](https://github.com/slundberg/shap), [Torchray](https://github.com/facebookresearch/TorchRay) extended to multimodal classification context: **mmxai/interpretability/classification/**

- Implementation of image inpainting: **mmxai/text_removal/**

- Our integrated XAI interface application: **web_app/**

- Extended [MMF](https://github.com/facebookresearch/mmf) source code to allow easy interfacing with pretrained models for Facebook AI's [the hateful memes challenge](https://github.com/facebookresearch/mmf/tree/master/projects/hateful_memes): **mmf_modified/**

## Modularity
The components above are well-separated. Future developers can easily expand this project further:

- Supporting more machine learning tasks: The multimodal XAI implementations currently work for multimodal binary classification tasks, specifically the MMF hateful memes classification. If there is a need for explaining other tasks like VQA, image captioning with reading, just add a directory under mmxai/interpretability specifying the type of the task (multilabel classification, multiclass classification, regression etc.) and work on it.

- Adding more XAI methods: We mainly focused on feature importance branch of XAI methods. There are many other types of XAI methods to explore. If anyone feels like doing that for any machine learning task, create a folder under /mmxai/interpretability/[replaced_by_task_name]/ for the method.

- Enabling extra functionalities: For the hateful memes task, we incorporated image inpainting as an option for our web app users to remove text chunks from their uploaded meme image to avoid potential occlusion of any image information. Add any extra tricks under /mmxai/.

- Adding more features to the web interface: Add to /mmxai. We have packed the /mmxai/* as a python library. Following the below installation steps and then import mmxai in the codes under /web_app where needed. 
