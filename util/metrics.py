import numpy as np

from scipy import linalg as linAlg


# 计算两个旋转矩阵的距离
def compute_angle_dists(preds, labels):
    # Get rotation matrices from prediction and ground truth angles
    predR = angle2dcm(preds[0], preds[1], preds[2])
    gtR = angle2dcm(labels[0], labels[1], labels[2])

    # Get geodesic distance
    return linAlg.norm(linAlg.logm(np.dot(predR.T, gtR)), 2) / np.sqrt(2)


# 视角->旋转矩阵
def angle2dcm(xRot, yRot, zRot, deg_type='deg'):
    if deg_type == 'deg':
        xRot = xRot * np.pi / 180.0
        yRot = yRot * np.pi / 180.0
        zRot = zRot * np.pi / 180.0

    xMat = np.array([
        [np.cos(xRot), np.sin(xRot), 0],
        [-np.sin(xRot), np.cos(xRot), 0],
        [0, 0, 1]
    ])

    yMat = np.array([
        [np.cos(yRot), 0, -np.sin(yRot)],
        [0, 1, 0],
        [np.sin(yRot), 0, np.cos(yRot)]
    ])

    zMat = np.array([
        [1, 0, 0],
        [0, np.cos(zRot), np.sin(zRot)],
        [0, -np.sin(zRot), np.cos(zRot)]
    ])

    return np.dot(zMat, np.dot(yMat, xMat))


class kp_dict(object):

    def __init__(self, num_classes=12):
        # self.keypoint_dict用来储存预测值和ground_truth,以便下一步预测
        self.keypoint_dict = dict()
        self.num_classes = num_classes
        # [0, 360, 720, 1080, 1440, 1800, 2160, 2520, 2880, 3240, 3600, 3960, 4320]
        self.class_ranges = list(range(0, 360 * (self.num_classes + 1), 360))
        # compute_angle_dists(curr_pred, curr_label) <= self.threshold = np.pi / 6.
        self.threshold = np.pi / 6.

    """
        Updates the keypoint dictionary
        params:     unique_id       unique id of each instance (NAME_objc#_kpc#)
                    predictions     the predictions for each vector
                    labels          ground_truth
        eg:
        results_dict.update_dict(key_uid,
                                 [azim.data.cpu().numpy(), elev.data.cpu().numpy(), tilt.data.cpu().numpy()],
                                 [azim_label.data.cpu().numpy(), elev_label.data.cpu().numpy(),tilt_label.data.cpu().numpy()])
        key_uid={tuple:64}
        azim={Tensor:64*360}
        azim_label={Tensor:64}
    """

    def update_dict(self, unique_id, predictions, labels):
        """Log a scalar variable."""
        if type(predictions) == int:
            predictions = [predictions]
            labels = [labels]
        # 记录每个图片的uid,预测值,ground_truth
        for i in range(0, len(unique_id)):  # 64
            image = unique_id[i].split('_objc')[0]
            obj_class = int(unique_id[i].split('_objc')[1].split('_kpc')[0])
            kp_class = int(unique_id[i].split('_objc')[1].split('_kpc')[1])

            start_index = self.class_ranges[obj_class]
            end_index = self.class_ranges[obj_class + 1]

            pred_probs = (predictions[0][i], predictions[1][i], predictions[2][i])

            label_probs = (labels[0][i], labels[1][i], labels[2][i])

            # 如果image(uid:str)已经在keypoint_dict.keys()中,则只赋值self.keypoint_dict[image][kp_class]
            if image in list(self.keypoint_dict.keys()):
                self.keypoint_dict[image][kp_class] = pred_probs
            # 否则更新self.keypoint_dict[image]的所有键值
            else:
                self.keypoint_dict[image] = {'class': obj_class, 'label': label_probs, kp_class: pred_probs}

    def calculate_geo_performance(self):
        for image in list(self.keypoint_dict.keys()):
            curr_label = self.keypoint_dict[image]['label']
            self.keypoint_dict[image]['geo_dist'] = dict()
            self.keypoint_dict[image]['correct'] = dict()
            for kp in list(self.keypoint_dict[image].keys()):
                if type(kp) != str:
                    # kp 只有一个值:0
                    # kp=0 keypoint_dict[image][kp]是一个size=3的tuple,里面存放了3个角度的预测值 ndarray:(360,)
                    # np.argmax(): Returns the indices of the maximum values along an axis.
                    # curr_pred 为预测的视角中概率最大的
                    curr_pred = [np.argmax(self.keypoint_dict[image][kp][0]),
                                 np.argmax(self.keypoint_dict[image][kp][1]),
                                 np.argmax(self.keypoint_dict[image][kp][2])]
                    # compute_angle_dists(): return linAlg.norm(linAlg.logm(np.dot(predR.T, gtR)), 2) / np.sqrt(2)
                    self.keypoint_dict[image]['geo_dist'][kp] = compute_angle_dists(curr_pred, curr_label)
                    # self.threshold = np.pi / 6.
                    self.keypoint_dict[image]['correct'][kp] = \
                        1 if (self.keypoint_dict[image]['geo_dist'][kp] <= self.threshold) else 0

    def metrics(self, unique=False):
        '''
        type_accuracy, type_total, type_geo_dist = results_dict.metrics()
        '''
        self.calculate_geo_performance()
        # 每个type的几何距离记录
        type_geo_dist = [[] for x in range(0, self.num_classes)]
        # 每个type的正确个数
        type_correct = np.zeros(self.num_classes, dtype=np.float32)
        # 每个type的总数
        type_total = np.zeros(self.num_classes, dtype=np.float32)

        # 遍历每张图片结果
        for image in list(self.keypoint_dict.keys()):
            # 对于每个图片结果都重新初始化
            object_type = self.keypoint_dict[image]['class']
            curr_correct = 0.
            curr_total = 0.
            curr_geodist = []
            # kp 只取 0 ?
            for kp in list(self.keypoint_dict[image]['correct'].keys()):
                # 如果预测的结果和ground_truth的compute_angle_dists()结果小于self.threshold=np.pi/6,则curr_correct加1
                curr_correct += self.keypoint_dict[image]['correct'][kp]
                # 每遍历一个结果curr_total加1
                curr_total += 1.
                # curr_geodist储存每一个预测值与其对应的ground_truth的compute_angle_dists()结果
                curr_geodist.append(self.keypoint_dict[image]['geo_dist'][kp])
            # unique default:False
            if unique:
                # curr_correct是所有预测正确(距离小于阈值)的个数与总个数的比值
                curr_correct = curr_correct / curr_total
                curr_total = 1.
                curr_geodist = [np.median(curr_geodist)]
            # 此类数据的正确数+1
            type_correct[object_type] += curr_correct
            # 此类数据的总数+1
            type_total[object_type] += curr_total
            for dist in curr_geodist:
                # 将每个结果与其对应的真实结果的距离添加到对应的list中
                type_geo_dist[object_type].append(dist)
        # 记录了每个种类的正确率
        type_accuracy = np.zeros(self.num_classes, dtype=np.float16)
        for i in range(0, self.num_classes):
            if type_total[i] > 0:
                type_accuracy[i] = float(type_correct[i]) / type_total[i]

        self.calculate_performance_baselines()
        # 返回每类数据的:1.正确率 2.测试总数 3.所有测试结果的几何距离
        return type_accuracy, type_total, type_geo_dist

    def calculate_performance_baselines(self, mode='real'):
        # 记录所有结果的距离,按照类别分成不同的list
        mean_baseline = [[] for x in range(0, self.num_classes)]
        # 记录所有结果的距离,按照类别分成不同的list
        total_baseline = [[] for x in range(0, self.num_classes)]
        # mean_baseline和total_baseline其实是一样的
        # mean_baseline==total_baseline:True

        # iterate over batch
        # 遍历所有结果
        for image in list(self.keypoint_dict.keys()):
            obj_cls = self.keypoint_dict[image]['class']
            # performance? 记录image结果与对应ground_truth的距离
            # perf只有一个元素?
            perf = [self.keypoint_dict[image]['geo_dist'][kp] \
                    for kp in list(self.keypoint_dict[image]['geo_dist'].keys())]

            # Append baselines
            mean_baseline[obj_cls].append(np.mean(perf))
            for p in perf:
                total_baseline[obj_cls].append(p)

        # embed()
        # np.around(*,2):四舍五入到小数点后两位
        # accuracy_mean是所有距离结果中小于self.threshold的平均值*100,保留两位小数,按照物体种类排序
        accuracy_mean = np.around(
            [100. * np.mean([num < self.threshold for num in mean_baseline[i]]) for i in range(0, self.num_classes)],
            decimals=2)
        # accuracy_total == accuracy_mean
        accuracy_total = np.around(
            [100. * np.mean([num < self.threshold for num in total_baseline[i]]) for i in range(0, self.num_classes)],
            decimals=2)

        # 取每个种类的所有距离结果的中值*180/np.pi(~57.32) ,按照种类储存
        medError_mean = np.around([(180. / np.pi) * np.median(mean_baseline[i]) for i in range(0, self.num_classes)],
                                  decimals=2)
        # medError_total == medError_mean
        medError_total = np.around([(180. / np.pi) * np.median(total_baseline[i]) for i in range(0, self.num_classes)],
                                   decimals=2)

        if np.isnan(accuracy_mean[0]):  # False
            accuracy_mean = accuracy_mean[[4, 5, 8]]
            accuracy_total = accuracy_total[[4, 5, 8]]
            medError_mean = medError_mean[[4, 5, 8]]
            medError_total = medError_total[[4, 5, 8]]

        # *_baseline:所有测试结果的compute_angle_dists(),size:12*n
        # accuracy_*:所有距离结果中小于self.threshold的平均值*100,保留两位小数,按照物体种类排序,size:12
        # medError_*:取每个种类的所有距离结果的中值*180/np.pi(~57.32),保留两位小数,按照种类储存,size:12
        # print("--------------------------------------------")
        # print("Accuracy ")
        # print("mean      : ", accuracy_mean, " -- mean : ", np.round(np.mean(accuracy_mean), decimals=2))
        # print("total     : ", accuracy_total, " -- mean : ", np.round(np.mean(accuracy_total), decimals=2))
        # print("")
        # print("Median Error ")
        # print("mean      : ", medError_mean, " -- mean : ", np.round(np.mean(medError_mean), decimals=2))
        # print("total     : ", medError_total, " -- mean : ", np.round(np.mean(medError_total), decimals=2))
        # print("--------------------------------------------")
