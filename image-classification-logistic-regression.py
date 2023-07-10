#!/usr/bin/env python
# coding: utf-8

# # image-classification-logistic-regression
# 
# Use the "Run" button to execute the code.

# In[1]:


import torch
import torchvision
from torchvision.datasets import MNIST


# In[2]:


dataset = MNIST(root='data/', download=True)


# In[3]:


len(dataset)


# In[4]:


test_dataset = MNIST(root='data/', train=False)
len(test_dataset)


# In[5]:


dataset[0]


# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label:', label)


# In[8]:


image, label = dataset[10]
plt.imshow(image, cmap='gray')
print('Label:', label)


# In[9]:


import torchvision.transforms as transforms


# In[10]:


dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor())
img_tensor, label = dataset[0]
print(img_tensor.shape, label)


# In[11]:


print(img_tensor[0,10:15,10:15])
print(torch.max(img_tensor), torch.min(img_tensor))


# In[12]:


# Plot the image by passing in the 28x28 matrix
plt.imshow(img_tensor[0,10:15,10:15], cmap='gray');


# In[13]:


from torch.utils.data import random_split

train_ds, val_ds = random_split(dataset, [50000, 10000])
len(train_ds), len(val_ds)


# In[14]:


from torch.utils.data import DataLoader

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)


# In[15]:


# Install the library
get_ipython().system('pip install jovian --upgrade --quiet')


# In[16]:


import jovian
jovian.commit(project='03-logistic-regression-live')


# In[17]:


import torch.nn as nn

input_size = 28*28
num_classes = 10

# Logistic regression model
model = nn.Linear(input_size, num_classes)


# In[18]:


print(model.weight.shape)
model.weight


# In[19]:


print(model.bias.shape)
model.bias


# In[20]:


for images, labels in train_loader:
    print(labels)
    print(images.shape)
    outputs = model(images)
    print(outputs)
    break


# In[21]:


images.shape


# In[22]:


images.reshape(128, 784).shape


# In[23]:


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
model = MnistModel()


# In[24]:


print(model.linear.weight.shape, model.linear.bias.shape)
list(model.parameters())


# In[25]:


for images, labels in train_loader:
    print(images.shape)
    outputs = model(images)
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)


# In[26]:


import torch.nn.functional as F


# In[27]:


# Apply softmax for each output row
probs = F.softmax(outputs, dim=1)

# Look at sample probabilities
print("Sample probabilities:\n", probs[:2].data)

# Add up the probabilities of an output row
print("Sum: ", torch.sum(probs[0]).item())


# In[28]:


max_probs, preds = torch.max(probs, dim=1)
print(preds)
print(max_probs)


# In[29]:


labels


# In[30]:


torch.sum(preds == labels)


# In[31]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[32]:


accuracy(outputs, labels)


# In[33]:


probs


# In[34]:


loss_fn = F.cross_entropy
# Loss for current batch of data
loss = loss_fn(outputs, labels)
print(loss)


# In[35]:


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # for recording epoch-wise results
    
    for epoch in range(epochs):
        
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history


# In[36]:


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# In[37]:


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
model = MnistModel()


# In[38]:


result0 = evaluate(model, val_loader)
result0


# In[39]:


history1 = fit(5, 0.001, model, train_loader, val_loader)


# In[40]:


history2 = fit(5, 0.001, model, train_loader, val_loader)


# In[41]:


history3 = fit(5, 0.001, model, train_loader, val_loader)


# In[42]:


history4 = fit(5, 0.001, model, train_loader, val_loader)


# In[43]:


history = [result0] + history1 + history2 + history3 + history4
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');


# In[44]:


jovian.log_metrics(val_acc=history[-1]['val_acc'], val_loss=history[-1]['val_loss'])


# In[45]:


jovian.commit(project='03-logistic-regression', environment=None)


# In[46]:


# Define test dataset
test_dataset = MNIST(root='data/', 
                     train=False,
                     transform=transforms.ToTensor())


# In[47]:


img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Shape:', img.shape)
print('Label:', label)


# In[48]:


def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


# In[49]:


img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))


# In[50]:


img, label = test_dataset[10]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))


# In[51]:


img, label = test_dataset[193]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))


# In[52]:


img, label = test_dataset[1839]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))


# In[53]:


test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model, test_loader)
result


# In[54]:


torch.save(model.state_dict(), 'mnist-logistic.pth')


# In[55]:


model2 = MnistModel()
model2.state_dict()


# In[56]:


evaluate(model2, test_loader)


# In[57]:


jovian.commit(project='03-logistic-regression', environment=None, outputs=['mnist-logistic.pth'])


# In[ ]:




