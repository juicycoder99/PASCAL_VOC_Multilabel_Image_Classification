"""PASCAL VOC 2007 multi-label dataset preparation for Assignment 3.

Builds, for each image, a 20-dimensional present/absent label vector from the VOC detection
annotations and a 224x224 uint8 image tensor, cached to .pt files for fast reuse.
"""
import os, torch, torchvision
import torchvision.transforms.functional as TF

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}


def build_split(image_set, root="./data", size=224):
    cache = os.path.join(root, f"voc2007_{image_set}_{size}.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        return d["X"], d["Y"]
    ds = torchvision.datasets.VOCDetection(root, year="2007", image_set=image_set, download=True)
    imgs, labels = [], []
    for img, ann in ds:
        objs = ann["annotation"]["object"]
        objs = objs if isinstance(objs, list) else [objs]
        y = torch.zeros(len(VOC_CLASSES))
        for o in objs:
            y[CLASS_TO_IDX[o["name"]]] = 1.0
        img = TF.resize(img, [size, size])
        imgs.append(TF.pil_to_tensor(img))      # uint8 CHW
        labels.append(y)
    X = torch.stack(imgs)            # (N, 3, size, size) uint8
    Y = torch.stack(labels)          # (N, 20) float
    torch.save({"X": X, "Y": Y}, cache)
    return X, Y


if __name__ == "__main__":
    for split in ["trainval", "test"]:
        X, Y = build_split(split)
        print(f"{split}: X={tuple(X.shape)} ({X.dtype}), Y={tuple(Y.shape)}, "
              f"avg labels/img={Y.sum(1).mean():.2f}")
