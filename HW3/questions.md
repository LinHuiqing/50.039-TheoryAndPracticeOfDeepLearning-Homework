# HW3 Questions

## Part 1 - TestingPretrainedResNet

### Task

Modify the code above, to peform data augmentation for the testing sample (averaging the scores of 5 crops: center crop, upper left crop, lower left crop, lower right crop, upper right crop).

Pls briefly discuss the advantages and disadvantages of using testing data augmentation.

## Part 2 - FinetuningNetwork

### Tasks

(Note: In this task, if you are adapting the code based on the open-source projects, pls acknowledge the original source in your code files, and also clearly mention it in your report. Also you need to clearly highlight which parts are done by yourself)

1. Replace the used base model (densenet169) to another model (refer to https://pytorch.org/vision/0.8/models.html for more types of models). Pls compare the performance of these two models on the validation set.
2. Please try different training methods that use densenet169 as the base model (i.e., training the whole model from scratch, finetuning the model but only updating the top layers, finetuning the whole model), and compare their performance on the validation set. Please also draw the curves of training/validation losses over training steps for these methods, and give your analysis based on the observed curves.
3. For the model based on densenet169, please also report its performance (when you use the training method of finetuning the model but only updating the top layers) on the testing set.
4. Please replace the base model to a new model which contains some convolutional layers. You need to implement this new model by yourselves, and then report its performance on the validation set. Note, pls try different numbers of convolutional layers for your model, and compare their results, and give analysis for the results. You need to try at least 2 different numbers of conv layers.

Extra tasks (not included in Homework 3):

5. Please try using two different learning rate scheduling schemes for densenet169, and compare the performance on the validation set.
6. Please try using two different optimizers for densenet169, and compare the performance on the validation set.

### Hints

1. For the densenet169 model, the final layer with parameters is the 'classifier'. Thus, we replaced 'model.classifier' to a new layer when doing transfer learning. But for other models, the name of the final layer with parameters may be different (e.g., 'fc' for the ResNet model, and thus you need to replace 'model.fc' to another layer if you want to use ResNet. You may refer to the "Transfer Learning
" section of https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/)

    Note: you need to modify the make_NN function in model_ic.py

2. If you want to finetune the whole model, you need to do two steps:
    1. you should not freeze any parameter in the model;
    2. in the optimizer (optim.Adam), you need to optimize all the parameters. You can refer to the "Transfer Learning" section of https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/, where the whole model is finetuned in that example. You can check the optim.Adam method used in that example.

3. You need to modify the make_NN function to complete this task.

4. You can refer to https://colab.research.google.com/drive/1zhDmMfSFBy3clH-NRp9nXruQXnckZ3X1#scrollTo=KZd049wKyFT8, where a new model is implemented, instead of loading a pre-trained model.
