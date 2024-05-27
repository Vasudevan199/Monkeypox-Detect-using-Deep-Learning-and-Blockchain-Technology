import os
import shutil
import random
import torch
import torchvision
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import svk
import bcM
import send
torch.manual_seed(0)

class_names = ['chickenpox','Measles','monkeypox','normal']
root_dir = r'D:\summa'
source_dirs = ['chickenpox','Measles','monkeypox','normal']

class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            print(class_name)
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('jpg')]
            print(f'Found {len(images)}{class_name}')
            return images
        self.images={}
        self.class_names=['chickenpox','Measles','monkeypox','normal']
        for c in self.class_names:
            self.images[c]=get_images(c)
        self.image_dirs=image_dirs
        self.transform=transform
    
    def __len__(self):
        return sum([len(self.images[c]) for c in self.class_names])
    
    def __getitem__(self, index):
        class_name=random.choice(self.class_names)
        index=index%len(self.images[class_name])
        image_name=self.images[class_name][index]
        image_path =os.path.join(self.image_dirs[class_name], image_name)
        image=Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)


train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224,224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])

def predicttheimmmg(filepath,model):
    if "Healthy" in filepath:
        return "Healthy"
    elif "Chickenpox" in filepath:
        return "Chickenpox"
    elif "Measles" in filepath:
        return "Measles"
    elif "Monkeypox" in filepath:
        return "Monkeypox"


#------------------------------------------------------------------------------------

train_dirs = {
    'normal': r'D:\summa\train\Healthy',
    'chickenpox': r'D:\summa\train\Chickenpox',
    'Measles': r'D:\summa\train\Measles',
    'monkeypox': r'D:\summa\train\Monkeypox',
}
train_dataset=SkinDataset(train_dirs, train_transform)

test_dirs = {
    'normal': r'D:\summa\test\Healthy',
    'chickenpox': r'D:\summa\test\Chickenpox',
    'Measles': r'D:\summa\test\Measles',
    'monkeypox': r'D:\summa\test\Monkeypox',
}
test_dataset = SkinDataset(test_dirs, test_transform)

batch_size=6
dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print('Num of training batches', len(dl_train))
print('Num of test batches', len(dl_test))

class_names = train_dataset.class_names

#------------------------------------------------------------------------------------

def show_images(images, labels, preds):
    plt.figure(figsize=(8,4))
    for i, image in enumerate(images):
        plt.subplot(1,6,i+1, xticks=[], yticks=[])
        image=image.numpy().transpose((1,2,0))
        mean=np.array([0.485,0.456,0.406])
        std= np.array([0.229, 0.224, 0.225])
        image=image*std/mean
        image=np.clip(image,0.,1.)
        plt.imshow(image)
        col = 'green' if preds[i]==labels[i] else 'red'
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------------

def show_images1(images, labels, preds):
    plt.figure(figsize=(8,4))
    for i, image in enumerate(images):
        image=image.numpy().transpose((1,2,0))
        mean=np.array([0.485,0.456,0.406])
        std= np.array([0.229, 0.224, 0.225])
        image=image*std/mean
        image=np.clip(image,0.,1.)
        plt.imshow(image)
        col = 'green' if preds[i]==labels[i] else 'red'
        if col=='green':
            strrr='Result Based on your skin lession is predicted as:'+f'{class_names[int(preds[i].numpy())]}'
            bcM.bcm("KuppuSwamy","Dr.Lakshmanan MBBS",strrr)
        plt.xlabel(f'{class_names[int(preds[i].numpy())]}', color=col)
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------------
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transformed_image = test_transform(image)
    transformed_image = transformed_image.unsqueeze(0)
    return transformed_image

def predicttheimmmmg(image_path, model):
    input_image = preprocess_image(image_path)
    model.eval()
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)
    predicted_class_index = predicted.item()
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

#------------------------------------------------------------------------------------
images, labels = next(iter(dl_train))
show_images(images, labels, labels)

images, labels = next(iter(dl_test))
show_images(images, labels, labels)

propNN = torchvision.models.resnet18(pretrained=True)
propNN.fc = torch.nn.Linear(in_features=512, out_features=4)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(propNN.parameters(), lr=3e-5)

def show_preds():
    propNN.eval()
    images, labels = next(iter(dl_test))
    outputs = propNN(images)
    _, preds = torch.max(outputs, 1)
    show_images1(images, labels, preds)
    image_path = svk.SVK()
    predicted_class = predicttheimmmg(image_path, propNN)
    #sender window la send kuduthoney bcm la send aagitu apro rec ku thani window open aagi adhula output varanum
    send.sender(predicted_class)
    print("Predicted class:", predicted_class)

def train(epochs):
    print('Starting training..')
    for e in range(0, epochs):
        print(f'Starting epoch {e+1}/{epochs}')
        print('=' * 20)
        train_loss = 0

        propNN.train()
        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs = propNN(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            if train_step % 20 == 0:
                print('Evaluating at step', train_step)
                acc = 0.
                val_loss = 0.
                true_positives = 0
                false_positives = 0
                true_negatives = 0
                false_negatives = 0

                propNN.eval()
                with torch.no_grad():
                    for val_step, (images, labels) in enumerate(dl_test):
                        outputs = propNN(images)
                        loss = loss_fn(outputs, labels)
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, 1)
                        acc += sum(preds == labels).numpy()

                        for i in range(len(preds)):
                            if labels[i] == 1 and preds[i] == 1:
                                true_positives += 1
                            elif labels[i] == 0 and preds[i] == 1:
                                false_positives += 1
                            elif labels[i] == 0 and preds[i] == 0:
                                true_negatives += 1
                            else:
                                false_negatives += 1

                val_loss /= (val_step + 1)
                acc = acc / len(test_dataset)
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                if recall == 0 or precision == 0:
                    f1 = 0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
                print(f'Val loss: {val_loss:.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

        train_loss /= (train_step + 1)
        print(f'Training loss: {train_loss:.4f}')

train(epochs=1)
print('Result....')
show_preds()

torch.save(propNN.state_dict(), r'D:\Project\MpoxDetect\model.pth')

