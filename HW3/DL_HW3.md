# 50.039 Theory and Practice of Deep learning - Homework 3

## Part 1 - Testing Pretrained ResNet

Note: Full code and answers to Part 1 are also in the Jupyter notebook `HW3_TestingPretrainedResNet.ipynb`.

### Task

> Modify the code above, to peform data augmentation for the testing sample (averaging the scores of 5 crops: center crop, upper left crop, lower left crop, lower right crop, upper right crop).

I used the following image for this task:

![](https://i.imgur.com/dFUfhMs.jpg)

The results from the original code are as follows:

![](https://i.imgur.com/viRVBtk.png)

The top 2 labels which have significantly higher confidence scores than the other labels are "tiger cat" (282) and "tabby" (281) respectively. Both of these labels are correct as this is a mackeral tabby cat, which is also known as a tiger cat.

To modify the code to perform the 5 crops, I first commented out the CenterCrop from the transform function as we want to get the full image after transforming.

``` python
transform = torchvision.transforms.Compose([
  torchvision.transforms.Resize(256),
  # torchvision.transforms.CenterCrop(224),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  )])
```

To debug our crops, we can use the `matplotlib` package. We do so, we use the code below to display the image from the sliced tensors.

``` python
import matplotlib.pyplot as plt
plt.imshow(img_t[:, :224, :224].permute(1, 2, 0))
```

The output of the above code, which is the upper left crop, is the following:

![](https://i.imgur.com/tb84tsP.png)

We will then be able to replace the tensor in the code above to "test" our other crops.

As such, we were able to confidently determine that the crops are as follows.

``` python
crops = [
  img_t[:, (img_t.shape[1]-224)//2:(img_t.shape[1]+224)//2, (img_t.shape[2]-224)//2:(img_t.shape[2]+224)//2], # center
  img_t[:, :224, :224], # upper left
  img_t[:, (img_t.shape[1]-224):, :224], # lower left
  img_t[:, (img_t.shape[1]-224):, (img_t.shape[2]-224):], # lower right
  img_t[:, :224, (img_t.shape[2]-224):], # upper right
]
```

Next, we modify the portion where the classification is made, looping over each crop to make a prediction, then averaging over the confidence scores, as seen below.

``` python
for idx, crop_t in enumerate(crops):
    batch_t = torch.unsqueeze(crop_t, 0)

    # perform inference
    out = resnet(batch_t)

    # print top-5 classes predicted by model
    _, indices = torch.sort(out, descending=True)
    if idx == 0:
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100 / 5
    else:
        percentage += torch.nn.functional.softmax(out, dim=1)[0] * 100 / 5
    for idx in indices[0][:5]:
        print('Label:', idx, '. Confidence Score:', percentage[idx].item(), '%')
```

With this modified code, we then get the following result:

![](https://i.imgur.com/JgvqF7V.png)

As seen, there is a wider margin between the correct labels and the wrong labels, meaning that the 5 crop technique allows the model to better classify images.

> Pls briefly discuss the advantages and disadvantages of using testing data augmentation.

Advantages:

* Makes test data more similar to data which the model is trained on, possibly making predictions better.
* Testing data augmentation allows us to fix dimension issues when the model takes in fixed dimensions of input.
* Testing data augmentation can help increase the quality of testing inputs for better predictions.

Disadvantages:

* Augmented data might not be representative of real-world input (e.g. mirroring an image with text).

## Part 2 - Finetuning Network

Note: Full code for part 2 can be found in `flowers-image-classifier/`.

> (Note: In this task, if you are adapting the code based on the open-source projects, pls acknowledge the original source in your code files, and also clearly mention it in your report. Also you need to clearly highlight which parts are done by yourself)

### Task 1

> Replace the used base model (densenet169) to another model (refer to https://pytorch.org/vision/0.8/models.html for more types of models). 

For this task, the ResNet-18 model was chosen to replace the default model.

As ResNet-18 has a `.fc` layer instead of a `.classifier` layer, the code is modified so as to accomodate models with their last layers being either `.fc` or `.classifier`. This can be done with a `try ... except ...` block to catch the `AttributeError` which would occur otherwise. The code below shows the blocks where the code has been modified to do so.

``` python
def make_NN(n_hidden, n_epoch, labelsdict, lr, device, model_name, trainloader, validloader, train_data):
    ...
    # Make classifier
    try:
        n_in = next(model.classifier.modules()).in_features
        n_out = len(labelsdict)
        model.classifier = NN_Classifier(input_size=n_in, output_size=n_out, hidden_layers=n_hidden)
    except AttributeError:
        n_in = next(model.fc.modules()).in_features
        n_out = len(labelsdict)
        model.fc = NN_Classifier(input_size=n_in, output_size=n_out, hidden_layers=n_hidden)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    try:
        optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
    except AttributeError:
        optimizer = optim.Adam(model.fc.parameters(), lr = lr)
    ...
    # Add model info 
    try:
        model.classifier.n_in = n_in
        model.classifier.n_hidden = n_hidden
        model.classifier.n_out = n_out
        model.classifier.labelsdict = labelsdict
        model.classifier.lr = lr
        model.classifier.optimizer_state_dict = optimizer.state_dict
        model.classifier.model_name = model_name
        model.classifier.class_to_idx = train_data.class_to_idx
    except AttributeError:
        model.fc.n_in = n_in
        model.fc.n_hidden = n_hidden
        model.fc.n_out = n_out
        model.fc.labelsdict = labelsdict
        model.fc.lr = lr
        model.fc.optimizer_state_dict = optimizer.state_dict
        model.fc.model_name = model_name
        model.fc.class_to_idx = train_data.class_to_idx
    ...

# Define function to save checkpoint
def save_checkpoint(model, path):
    try:
        checkpoint = {'c_input': model.classifier.n_in,
                    'c_hidden': model.classifier.n_hidden,
                    'c_out': model.classifier.n_out,
                    'labelsdict': model.classifier.labelsdict,
                    'c_lr': model.classifier.lr,
                    'state_dict': model.state_dict(),
                    'c_state_dict': model.classifier.state_dict(),
                    'opti_state_dict': model.classifier.optimizer_state_dict,
                    'model_name': model.classifier.model_name,
                    'class_to_idx': model.classifier.class_to_idx
                    }
    except AttributeError:
        checkpoint = {'c_input': model.fc.n_in,
                    'c_hidden': model.fc.n_hidden,
                    'c_out': model.fc.n_out,
                    'labelsdict': model.fc.labelsdict,
                    'c_lr': model.fc.lr,
                    'state_dict': model.state_dict(),
                    'c_state_dict': model.fc.state_dict(),
                    'opti_state_dict': model.fc.optimizer_state_dict,
                    'model_name': model.fc.model_name,
                    'class_to_idx': model.fc.class_to_idx
                    }
    torch.save(checkpoint, path)
...
# Define function to load model
def load_model(path):
    try:
        # Make classifier
        model.classifier = NN_Classifier(input_size=cp['c_input'], output_size=cp['c_out'], \
                                        hidden_layers=cp['c_hidden'])
        
        # Add model info 
        model.classifier.n_in = cp['c_input']
        model.classifier.n_hidden = cp['c_hidden']
        model.classifier.n_out = cp['c_out']
        model.classifier.labelsdict = cp['labelsdict']
        model.classifier.lr = cp['c_lr']
        model.classifier.optimizer_state_dict = cp['opti_state_dict']
        model.classifier.model_name = cp['model_name']
        model.classifier.class_to_idx = cp['class_to_idx']
    except AttributeError:
        # Make classifier
        model.fc = NN_Classifier(input_size=cp['c_input'], output_size=cp['c_out'], \
                                        hidden_layers=cp['c_hidden'])
        
        # Add model info 
        model.fc.n_in = cp['c_input']
        model.fc.n_hidden = cp['c_hidden']
        model.fc.n_out = cp['c_out']
        model.fc.labelsdict = cp['labelsdict']
        model.fc.lr = cp['c_lr']
        model.fc.optimizer_state_dict = cp['opti_state_dict']
        model.fc.model_name = cp['model_name']
        model.fc.class_to_idx = cp['class_to_idx']
```

To train a ResNet-18 model, we can simply use the `--arch` flag provided. As such, 5 epochs of the models can be trained with the following commands:

``` shell
# train base model (densenet-169) for 5 epochs
python train.py "./flowers" --gpu --epochs=5

# train chosen model (resnet-18) for 5 epochs
python train.py "./flowers" --gpu --arch=resnet18 --epochs=5
```

> Pls compare the performance of these two models on the validation set. 

Below are screenshots for the performance of the 2 models on the validation set.

Raw results for base model:
<!-- ![](https://i.imgur.com/eCWgeTf.png) -->
![](https://i.imgur.com/DefHbdr.png)

Raw results for chosen model (ResNet-18):
<!-- ![](https://i.imgur.com/ipva2Mu.png) -->
![](https://i.imgur.com/XWsz8Be.png)

Note: Each statement is printed after 40 batches are trained.

With the numbers from the raw results are then plotted on graphs as shown below.

![](https://i.imgur.com/JR5p1VN.png)
![](https://i.imgur.com/jZUVwlz.png)

Based on the graphs, the following observations can be made:

* As both models are trained more batches, the validation loss decreases and validation accuracy increases.
    As the models are trained relevant features are learnt, resulting in more accurate predictions and thus validation loss is decreased while validation accuracy is increased.
* Densenet-169 has lower validation loss and higher validation accuracy than ResNet-18.
    As Densenet-169 is deeper than ResNet-18, Densenet-169 is able to learn higher level features compared to ResNet-18. This allows Densenet-169 to make better predictions with these higher level features, leading to lower validation loss and higher validation accuracy.

### Task 2

> Please try different training methods that use densenet169 as the base model (i.e., training the whole model from scratch, finetuning the model but only updating the top layers, finetuning the whole model), and compare their performance on the validation set. Please also draw the curves of training/validation losses over training steps for these methods, and give your analysis based on the observed curves.

There are 3 models which we need to train for this task, which are:

* whole model from scratch
* finetune the model but only the top layers
* finetune the whole model

From these models, we can see mainly 2 variables - pretrained / not pretrained and the number of layers that are trainable.

To this end, the variables `pretrained` and `no_of_trainable` were added as parameters for the `make_NN` function, and used to determine if the model is to get pretrained weights and if so the number of top layers to be trainable respectively.

``` python
def make_NN(n_hidden, n_epoch, labelsdict, lr, device, model_name, trainloader, validloader, train_data, no_of_trainable, pretrained):

    # import NN model 
    model = getattr(models, model_name)(pretrained=pretrained)
    
    if pretrained:
        no_of_frozen = len(list(model.parameters())) - no_of_trainable
        # freeze parameters that we don't need to re-train 
        for index, param in enumerate(model.parameters()):
            if index >= no_of_frozen:
                break
            param.requires_grad = False
```

To pass these variables as arguments from the command line, I made the following changes to the `train.py` file as well:

``` python
parser.add_argument("--unfreeze", type=int, default=0, help="unfreeze top n layers")
pretrained_group = parser.add_mutually_exclusive_group()
pretrained_group.add_argument('--pretrained', dest='pretrained', action='store_true', help="use pretrained weights")
pretrained_group.add_argument('--no-pretrained', dest='pretrained', action='store_false', help="do not use pretrained weights")
parser.set_defaults(pretrained=True)
...
model = make_NN(..., \
                no_of_trainable=args.no_of_trainable, \
                pretrained=args.pretrained)
```

However, running the modified scripts at this point would give a `RuntimeError` as shown below:

![](https://i.imgur.com/p49IpxU.png)

To overcome this issue, I decreased the batch size from the initial 64 to 32 in `utils_ic.py`, as shown below.

``` python
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
```

Now we can train 5 epochs of the required models with the following commands:

``` shell
# finetune top layer
python train.py "./flowers" --gpu --epochs=5 --no_of_trainable=0

# finetune whole model
python train.py "./flowers" --gpu --epochs=5 --no_of_trainable=508

# train from scratch
python train.py "./flowers" --gpu --epochs=5 --no-pretrained
```

Note: 508 is the number of layers in total excluding the classifier. This value was obtained from printing the model out.

After running the commands as listed above, we get the following results:

Raw results from finetuning the top layer:

![](https://i.imgur.com/dk9Qjqv.png)

Raw results from finetuning the whole model:

![](https://i.imgur.com/DYorrWW.png)

Raw results from training from scratch:

![](https://i.imgur.com/u4KaUcp.png)

Model is evaluated every 40 batches. Using the raw results, the following graph of loss against batch number can be plotted:

<!-- ![](https://i.imgur.com/X8h7nW1.png) -->
![](https://i.imgur.com/YqbJ37y.png)

"Finetune Top" means that the pretrained model is used, and we only finetuned the top layer. "Finetune All" means that the pretrained model is used, and all the layers were finetuned. "From Scratch" means that the model used was loaded without the pretrained weights.

From the graph, the following observations can be made:

* Both training and validation loss decrease as the models are trained with more batches.
    As more batches are trained, the model learns more relevant features, and thus loss is decreased for both training and validation.
* Training loss is consistently higher than the validation loss for all 3 models. 
    This is expected as dropout layers are being used by the Densenet-169 model. The dropout layers lead to the model not being able to use all the features learnt previously during training, resulting in a higher loss, as compared to during validation when all the features can be used by the model. This also means that the model is currently underfitting.
* Training loss and validation loss converge as more batches are trained.
    This is expected as the more the model is trained with dropout, the better each individual feature that is being learnt is. As training loss is calculated with some neurons excluded due to dropout, better individual features learnt allows the training loss to converge with validation loss.
* Training and validation loss for the model trained from scratch is significantly higher than those of the pretrained models.
    This is expected as the pretrained models are able to leverage on previously trained weights obtained from training the model on large datasets, while training the model from scratch would not have this advantage. This means that the losses would be much higher for the model being trained from scratch as the model tries to learn useful features from scratch.

While it is not obvious from the graph, the losses of the pretrained model with fine-tuning only done on the top layers are higher than when the model with fine-tuning for all layers. This is because fine-tuning on all layers allows the model to better learn features which are relevant to the current classification problem throughout the model.


### Task 3

> For the model based on densenet169, please also report its performance (when you use the training method of finetuning the model but only updating the top layers) on the testing set.

The following command is run to make a new directory `models` to save models.

``` shell
mkdir models
```

Next, the following command is run to train a pretrained model with fine-tuning only on the top layer for 5 epochs and save it in a file `pretrained_top_5ep` in the `models` directory.

``` shell
python train.py "./flowers/" --gpu --epochs=5 --save-dir=models/pretrained_top_5ep
```

The output of the command is as follows.

![](https://i.imgur.com/thDkKo6.png)

After which, the following script `test.py` is written to load a saved model and test it.

``` python
import argparse
from utils_ic import load_data, read_jason
from model_ic import load_model, test_model

parser = argparse.ArgumentParser(description="Test image classifier model")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--load_dir", help="load model")

args = parser.parse_args()

cat_to_name = read_jason(args.category_names)

trainloader, testloader, validloader, train_data = load_data(args.data_dir)

model = load_model(args.load_dir)

test_model(model, testloader, device=args.gpu)
```

To test the saved model `pretrained_top_5ep`, the following command is run:

``` shell
python test.py "./flowers/" --gpu --load_dir=model/pretrained_top_5ep
```

The resulting output is as follows:

![](https://i.imgur.com/gkDrBY0.png)

This means that the testing accuracy of the pretrained model with fine-tuning only on the top layer is **0.923**.

### Task 4

> Please replace the base model to a new model which contains some convolutional layers. You need to write this new model by yourselves, and then report its performance on the validation set. Note, pls try different numbers of convolutional layers for your model, and compare their results, and give analysis for the results. You need to try at least 2 different numbers of conv layers.

For this task, the following custom models were made:

<!-- <table>
    <thead>
        <tr>
            <th>layer name</th>
            <th>output size</th>
            <th>1-layer</th>
            <th>2-layer</th>
            <th>3-layer</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>conv1</td>
            <td>82x82</td>
            <td colspan=3 style="text-align:center"> 7x7, 64, stride 2, padding 3 </td>
        </tr>
        <tr>
            <td> layerx </td>
            <td> layer1 </td>
            <td> [ 3 x 3, 64 ] </td>
        </tr>
    </tbody>
</table> -->

Drawing inspiration from ResNet, 3 custom models were coded from scratch in `custom_models.py` as seen below. This file includes classes for the custom models `ConvNet1`, `ConvNet2` and `ConvNet3`, where there are 3, 5 and 7 convolution layers in total respectively.

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.batchnorm = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.batchnorm = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.batchnorm = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output
```

Next, the `model_ic.py` file was modified to allow the script to use the custom models.

``` python
# import custom models
from custom_models import ConvNet1, ConvNet2, ConvNet3

...

def make_NN(n_hidden, n_epoch, labelsdict, lr, device, model_name, trainloader, validloader, train_data, no_of_trainable, pretrained):
    # use custom models if specified by model_name
    if model_name == "ConvNet1":
        model = ConvNet1()
    elif model_name == "ConvNet2":
        model = ConvNet2()
    elif model_name == "ConvNet3":
        model = ConvNet3()
    else:
        # load other models according to parameters
        ...
```

To train the models for 5 epochs, the following commands can be run:

``` shell
# train ConvNet1
python train.py "./flowers/" --gpu --arch=ConvNet1 --epochs=5

# train ConvNet2
python train.py "./flowers/" --gpu --arch=ConvNet2 --epochs=5

# train ConvNet3
python train.py "./flowers/" --gpu --arch=ConvNet3 --epochs=5
```

<!-- ![](https://i.imgur.com/iFAQSz3.png)

![](https://i.imgur.com/VnXzAKX.png)

![](https://i.imgur.com/vQEE8fT.png) -->

<!-- ![](https://i.imgur.com/xv003Fd.png)

![](https://i.imgur.com/ljB4bRW.png)

![](https://i.imgur.com/2uv1lcH.png) 

![](https://i.imgur.com/XE7SVwT.png) -->

The following are the raw results from when training the custom models:

Raw results for ConvNet1:

![](https://i.imgur.com/2NE29zd.png)

Raw results for ConvNet2:

![](https://i.imgur.com/uhJuHRn.png)

Raw results for ConvNet3:

![](https://i.imgur.com/rId25KB.png)

Using the raw results the above, using Densenet-169 training from scratch as a benchmark, the following graph can be plotted:

![](https://i.imgur.com/RBjbxbY.png)

From the graph, the following observations can be made:

* All 3 custom models have consistently higher validation accuracy than Densenet-169 when training 5 epochs.
    As Densenet-169 is a much deeper network as compared to the custom networks, gradient updates for the lower layers will be much smaller, and thus it will take a much longer time to learn relevant features. This leads to Densenet-169 performing worse than the custom models when only 5 epochs are trained.
* The validation accuracy of ConvNet2 is higher than that of ConvNet1.
    As ConvNet2 has more convolution layers than ConvNet1, ConvNet2 is able to learn higher level features compared to ConvNet1. This allows better predictions to be made and thus ConvNet2 has higher validation accuracy compared to ConvNet1.
* The validation accuracy of ConvNet3 is lower than that of ConvNet2.
    While ConvNet3 has more convolution layers than ConvNet2 and is expected to perform better as it is able to learn higher level features, the number of epochs trained is low. As such, a possible reason causing the validation accuracy of ConvNet3 to be lower than ConvNet2 is that ConvNet3 is deeper, meaning that gradients at lower levels are updated more slowly, leading to the model being slower at learning relevant features. If the models are train on more epochs, ConvNet3 is expected to perform better than ConvNet2.

As only 5 epochs are trained, it is difficult to tell how the models will perform with more training.
