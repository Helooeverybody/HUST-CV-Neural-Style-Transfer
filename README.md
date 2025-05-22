# Neural Style Transfer
Data is available at: [data](https://github.com/victorkitov/style-transfer-dataset)

Group 9 members:
- Nguyen Nhat Minh 20225510
- Doi Sy Thang - 20225528
- Ngo Duy Dat - 20225480
- Nguyen Hoang Son Tung - 20225536
- Vu Tuan Truong - 20225535

## Project Description

Our project focuses on the neural style transfer problem using 4 models, ranging from simple to complex 
— from pretrained ResNet and VGG, to Transformer-based models and the well-known Adain model. 
Importantly, the evaluation metrics for these models are also a critical aspect that must be carefully 
considered and compared.


Our project will consist of two phases. In the first phase, we tackle the neural style transfer
problem by leveraging established models to learn content and style representations from a dataset
of content and style images, and subsequently generate stylized images.The second phase
functions as a standard regression task with three outputs corresponding to three users: here, we
freeze the weights of a pretrained ResNet and append several fully connected layers to produce
three regression outputs. This second phase is extremely straightforward, as we only use the
dataset’s ratings to demonstrate that a model with strong metric performance does not necessarily
achieve high subjective ratings, highlighting the challenge of evaluating metrics for applications
that are inherently aesthetic and subject to individual taste.

## 🧪 Model Performance Comparison

| Model       | ArtFID   |   FID   | LPIPS  | G-LPIPS |  CFSD  |
|-------------|----------|---------|--------|---------|--------|
| Adain       |  39.31   |  24.28  | 0.55   | 0.43    |  0.26  |
| WCT         |  39.67   |  24.53  | 0.55   | 0.58    |  0.34  |
| Patch_st    |  49.52   |  28.76  | 0.66   | 0.5344  |  1.02  |
| Transformer |  40.12   |  26.11  | 0.48   | 0.51    |  0.23  |

## Infer
To infer styled image and get ratings, please type
```
python infer.py --model transformer --content_img_dir content_5 --style_img_dir style_10 --retain_color True
```
You can easily change model transformer with adain, wct, patch_st, and retain_color= False to ensure
color styled following the style image. The stylised image will be save in folder output/modelname
## UI guide


![image](https://github.com/user-attachments/assets/2d55fc14-712b-49d7-8c4b-753f2b033028)


To run the UI, type 
```
streamlit run app.py
```
