import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataset,Cub2011
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate, parse


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    #train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    #import pdb
    #pdb.set_trace()
    data_transform = {
            "train":
            transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            #    transforms.Compose([
            #        transforms.Resize((256,256)),
            #        transforms.RandomCrop(224),
            #        transforms.RandomHorizontalFlip(),
            #        transforms.AutoAugment(),
            #        transforms.ToTensor(),
            #        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val":
            # transforms.Compose([transforms.Resize(256),
            #                        transforms.CenterCrop(224),
            #                        transforms.ToTensor(),
            #                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

            transforms.Compose([transforms.Resize((256,256)),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    #train_dataset = MyDataSet(images_path=train_images_path,images_class=train_images_label,transform=data_transform["train"])
    #val_dataset = MyDataSet(images_path=val_images_path,images_class=val_images_label,transform=data_transform["val"])
    train_dataset = MyDataset(root = args.root,train=True,val=False,transform=data_transform["train"])
    val_dataset = MyDataset(root = args.root,train=False,val=True,transform=data_transform["val"])
    # train_dataset = Cub2011(root=args.root, train=True, transform=data_transform["train"])
    # val_dataset = Cub2011(root=args.root, train=False, transform=data_transform["val"])


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    total_iters = len(train_loader) * args.epochs
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)
    best_acc = -1
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                scheduler=scheduler,
                                                epoch=epoch)


        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),"./weights/model-{}.pth".format(epoch))
        print("best acc = {:.2f}%".format(best_acc))
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

if __name__ == '__main__':
    main(args)
