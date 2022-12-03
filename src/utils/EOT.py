import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from image_process import rotate_img_ts
import cv2
from PIL import Image

img_A = cv2.imread('/home/imprints/Imprints/wm_adv_0.png')
img_A_ts = (torch.from_numpy(img_A).float().unsqueeze(0)/255.0).repeat(5,1,1,1).permute(0,3,1,2)
img_A_ts.requires_grad = True
print(img_A.shape, img_A_ts.size())


EOT = T.Compose([
    # T.RandomRotation(degrees=(0, 180)),
    T.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.3), scale=(0.5, 1)),
    T.RandomPerspective(distortion_scale=0.5, p=1.0),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
])
# perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=1.0)
eot_img_A_ts = [EOT(img_A_ts[i:i+1,:,:,:]) for i in range(img_A_ts.size(0))]
eot_img_A_ts = torch.vstack(eot_img_A_ts)

print(type(eot_img_A_ts))
print(eot_img_A_ts.size())

eot_img_A_ts = eot_img_A_ts.permute(0,2,3,1)
print(eot_img_A_ts.size())

loss = eot_img_A_ts.mean()
loss.backward()

print(img_A_ts.grad.size())

print(torch.sum(img_A_ts.grad!=0))
for i in range(5):
    cv2.imwrite(f'/home/imprints/Imprints/wm_adv_{i+1}.png', (eot_img_A_ts[i]*255).detach().numpy())