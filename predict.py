import os
from tqdm import tqdm
import torch
from torchvision import transforms
from vit_model import vit_base_patch16_224_in21k as create_model
from my_dataset import MyDataset
from utils import parse, accuracy

args = parse()

def test(model, data_loader):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    total_test_acc = 0
    total_test_loss = 0

    data_loader = tqdm(data_loader)
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            pred = model(images)

            acc = accuracy(pred, labels)[0]
            loss = loss_function(pred, labels)
            total_test_acc += acc.item()
            total_test_loss += loss.item()

            data_loader.desc = "loss: {:.3f}, acc: {:.2f}%".format(total_test_loss / (step + 1),
                                                                   total_test_acc / (step + 1))
    return total_test_loss / (step + 1), total_test_acc / (step + 1)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = os.listdir(args.model_weights_dir)
    model.sort(key=lambda x: int(x.split(".")[0].split("--")[1]))
    model_weight_path = model[len(model)-1]
    model_weight_path = os.path.join(args.model_weights_dir,model_weight_path)
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_dataset = MyDataset(root=args.root, train=False, val=False, transform=data_transform)
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))



    test(model=model,data_loader=test_loader)

if __name__ == '__main__':
    main()
