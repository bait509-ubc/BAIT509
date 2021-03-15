from PIL import Image
from torchvision.models import vgg16
from torchvision import transforms
import torch
import pandas as pd

def classify_image(img, topn = 4):

    clf = vgg16(pretrained=True)
    preprocess = transforms.Compose([
                 transforms.Resize(299),
                 transforms.CenterCrop(299),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),])

    with open('data/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    clf.eval()
    output = clf(batch_t)
    _, indices = torch.sort(output, descending=True)
    #[print(f"{classes[idx]}, {percentage[0, idx].item():.2f}") for idx in indices[0][:n_results]]
    probabilities = torch.nn.functional.softmax(output, dim=1)
    d = {'Class': [classes[idx] for idx in indices[0][:topn]], 
         'Probability': [probabilities[0, idx].item() for idx in indices[0][:topn]]}
    df = pd.DataFrame(d, columns = ['Class','Probability Score'])
    return df
