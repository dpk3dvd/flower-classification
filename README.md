# Flower-classification
## 📊 Dataset

We are using the famous **Iris dataset** from Kaggle:

🔗 [Download the dataset from Kaggle](https://www.kaggle.com/datasets/uciml/iris)

You can also download it using the Kaggle CLI:



## 📁 Folder Structure

flower_classification/
├── flowers/ # Dataset folder (each class in its own subfolder)
│ ├── roses/
│ ├── tulips/
│ ├── sunflowers/
│ └── ...
├── README.md # Project overview and instructions
└── .gitignore # Git ignore rules

## 🧠 Model Summary

The model is a **Convolutional Neural Network** with the following architecture:

- `Conv2D` → `MaxPooling2D`  
- `Conv2D` → `MaxPooling2D`  
- `Flatten`  
- `Dense (128 units)`  
- `Dense (softmax)` for multi-class classification

Compiled with:
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


Install Requirements

pip install tensorflow matplotlib
```bash
kaggle datasets download -d uciml/iris



