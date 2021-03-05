from PIL import Image
from torchvision.models import alexnet
from torchvision import transforms
import torch

def classify_image(img, n_results):

    clf = alexnet(pretrained=True)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]
                                                        )
                                   ]
                                  )
    with open('data/imagenet_classes.txt') as f:
      classes = [line.strip() for line in f.readlines()]
    
    img_t = transform (img)
    batch_t = torch.unsqueeze(img_t, 0)
    clf.eval()
    out = clf(batch_t)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)
    print("Class, Probability")
    print("------------------")
    [print(f"{classes[idx]}, {percentage[0, idx].item():.2f}") for idx in indices[0][:n_results]]