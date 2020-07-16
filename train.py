import argparse
import time

import numpy as np
import torch

from models import render4cnn
from util import SoftmaxVPLoss, Paths, get_data_loaders, kp_dict


def main(args):
    initialization_time = time.time()

    print("#############  Read in Database   ##############")
    train_loader, valid_loader = get_data_loaders(dataset=args.dataset,
                                                  batch_size=args.batch_size,
                                                  num_workers=args.num_workers,
                                                  model=args.model)

    print("#############  Initiate Model     ##############")

    assert args.model == 'r4cnn'
    assert Paths.render4cnn_weights != None, "Error: Set render4cnn weights path in util/Paths.py."
    model = render4cnn(weights_path=Paths.render4cnn_weights)
    args.no_keypoint = True

    # Loss functions
    criterion = SoftmaxVPLoss()

    # Parameters to train

    params = list(model.parameters())

    # Optimizer
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # train/evaluate on GPU
    model.cuda()

    print("Time to initialize take: ", time.time() - initialization_time)
    print("#############  Start Training     ##############")
    total_step = len(train_loader)

    for epoch in range(0, args.num_epochs):
        # 到了规定的eval_epoch周期,进行验证:eval_step
        if epoch % args.eval_epoch == 0:
            eval_step(model=model,
                      data_loader=valid_loader,
                      criterion=criterion,
                      step=epoch * total_step,
                      datasplit="valid")

        train_step(model=model,
                   train_loader=train_loader,
                   criterion=criterion,
                   optimizer=optimizer,
                   epoch=epoch,
                   step=epoch * total_step)


def train_step(model, train_loader, criterion, optimizer, epoch, step):
    model.train()
    total_step = len(train_loader)
    # loss_sum每个epoch都会清零
    loss_sum = 0.

    for i, (images, azim_label, elev_label, tilt_label, obj_class, kp_map, kp_class, key_uid) in enumerate(
            train_loader):

        # Set mini-batch dataset
        images = images.cuda()
        azim_label = azim_label.cuda()
        elev_label = elev_label.cuda()
        tilt_label = tilt_label.cuda()
        obj_class = obj_class.cuda()

        # Forward, Backward and Optimize
        model.zero_grad()

        if args.no_keypoint:  # r4cnn
            # def forward(self, x, obj_class):
            # return azim, elev, tilt
            azim, elev, tilt = model(images, obj_class)
        else:
            kp_map = kp_map.cuda()
            kp_class = kp_class.cuda()
            azim, elev, tilt = model(images, kp_map, kp_class, obj_class)

        loss_a = criterion(azim, azim_label)
        loss_e = criterion(elev, elev_label)
        loss_t = criterion(tilt, tilt_label)
        loss = loss_a + loss_e + loss_t

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        # Print log info
        # 针对这个epoch,每训练args.log_rate个batch后输出平均损失
        if i % args.log_rate == 0 and i > 0:
            print(f'Epoch [{epoch}/{args.num_epochs}] Step [{i}/{total_step}]: Training Loss={loss_sum / (i + 1):2.5f}')


def eval_step(model, data_loader, criterion, step, datasplit):
    model.eval()

    total_step = len(data_loader)
    epoch_loss_a = 0.
    epoch_loss_e = 0.
    epoch_loss_t = 0.
    epoch_loss = 0.
    results_dict = kp_dict()

    for i, (images, azim_label, elev_label, tilt_label, obj_class, kp_map, kp_class, key_uid) in enumerate(data_loader):
        # 每验证args.log_rate个batch,输出log
        if i % args.log_rate == 0:
            # print("Evaluation of %s [%d/%d] " % (datasplit, i, total_step))
            print(f'Evaluation of {datasplit} [{i}/{total_step}]')
        # Set mini-batch dataset
        images = images.cuda()
        # ground_truth的shape都是batch_size
        azim_label = azim_label.cuda()
        elev_label = elev_label.cuda()
        tilt_label = tilt_label.cuda()
        obj_class = obj_class.cuda()

        if args.no_keypoint:  # r4cnn
            # 验证也会将obj_class的ground_truth输入
            azim, elev, tilt = model(images, obj_class)
        else:
            kp_map = kp_map.cuda()
            kp_class = kp_class.cuda()
            azim, elev, tilt = model(images, kp_map, kp_class, obj_class)

        # embed()
        # 将此epoch的所有loss累加
        # azim:64*360
        # azi_label:64
        epoch_loss_a += criterion(azim, azim_label).item()
        epoch_loss_e += criterion(elev, elev_label).item()
        epoch_loss_t += criterion(tilt, tilt_label).item()

        results_dict.update_dict(key_uid,
                                 [azim.data.cpu().numpy(), elev.data.cpu().numpy(), tilt.data.cpu().numpy()],
                                 [azim_label.data.cpu().numpy(), elev_label.data.cpu().numpy(),
                                  tilt_label.data.cpu().numpy()])
    # 返回:
    # 1.每类数据的正确率(距离<self.threshold = np.pi / 6.)
    # 2.每类的测试总数
    # 3.所有测试结果的几何距离,size:12*n
    type_accuracy, type_total, type_geo_dist = results_dict.metrics()

    # 取每个类别所有距离的中值*(180/np.pi)
    geo_dist_median = [np.median(type_dist) * 180. / np.pi for type_dist in type_geo_dist if type_dist != []]
    # 每类数据的正确率*100
    type_accuracy = [type_accuracy[i] * 100. for i in range(0, len(type_accuracy)) if type_total[i] > 0]
    # 所有种类正确率*100再求平均
    w_acc = np.mean(type_accuracy)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Type Acc_pi/6 : ", np.around(type_accuracy, 2), " -> ", np.around(w_acc, 2), " %")
    print("Type Median   : ", [np.around((int(1000 * a_type_med) / 1000.), 2) for a_type_med in geo_dist_median],
          " -> ",
          np.around((int(1000 * np.mean(geo_dist_median)) / 1000.), 2), " degrees")
    print("Type Loss     : ",
          np.around((epoch_loss_a / total_step, epoch_loss_e / total_step, epoch_loss_t / total_step), 2), " -> ",
          np.around(((epoch_loss_a + epoch_loss_e + epoch_loss_t) / total_step), 2))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # logging parameters
    parser.add_argument('--eval_epoch', type=int, default=5)
    parser.add_argument('--log_rate', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=7)

    # training parameters
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--optimizer', type=str, default='sgd')

    # experiment details
    parser.add_argument('--dataset', type=str, default='pascalFull')
    parser.add_argument('--model', type=str, default='r4cnn')
    parser.add_argument('--experiment_name', type=str, default='Test')
    parser.add_argument('--just_attention', action="store_true", default=False)

    args = parser.parse_args()
    main(args)
