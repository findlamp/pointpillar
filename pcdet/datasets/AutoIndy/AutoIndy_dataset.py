import copy
import pickle

import numpy as np
from skimage import io
from . import AutoIndy_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils

from ..dataset import DatasetTemplate

class AutoIndyDataset(DatasetTemplate):

    def __init__(self, dataset_cfg, class_names, training = True, root_path = None, logger = None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training = training, root_path=root_path,logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_autoIndy_data(self.mode)


    def include_autoIndy_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading AutoIndy dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)
        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for AutoIndy datasets: %d' % (len(kitti_infos)))
    
    def set_split(self,split):

        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None


    def get_lidar(self,idx):
        lidar_file = self.root_split_path / 'lidar' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype = np.float32).reshape(-1,4)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label' / ('%s.txt' % idx)
        assert label_file.exists()
        return AutoIndy_utils.get_objects_from_label(label_file)

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures
        def process_single_scene(sample_idx):
            print("%s sample_idx: %s" % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features' : 4, 'lidar_idx' : sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                #annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                #annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                #annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                #annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = annotations['location']
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                #loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype = np.int32)
                    
                    for k in range(num_objects):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt
            return info
        
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
        

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('autoIndy_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:,:3] -= gt_boxes[i,:3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))
                    db_info = {
                        'name':names[i],'path':db_path,'image_idx':sample_idx,
                        'box3d_lidar':gt_boxes[i], 'num_points_in_gt':gt_points.shape[0],
                        'difficulty':difficulty[i]
                    }
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print("Database %s: %d" % (k, len(v)))
        
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos,f)
    
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dict, class_names, output_path = None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['dimensions'] = pred_boxes[:,3:6]
            pred_dict['location'] = pred_boxes[:,0:3]
            pred_dict['rotation_y'] = pred_boxes[:,6]
            pred_dict['score'] = pred_scores

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dict):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl
                    
                    for idx in range(len(loc)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx],
                                 loc[idx][0], loc[idx][2], loc[idx][1], dims[idx][1],
                                 dims[idx][2], dims[idx][0], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)
        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}


        #TODO modify
        from .AutoIndy_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        print(eval_det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)
    

    def __getitem__(self, index):

        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id' : sample_idx,
        }

        if 'annos' in info:
            annos = info['annos']
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_lidar = np.concatenate([loc,dims,rots[..., np.newaxis]], axis=1).astype(np.float32)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            input_dict['points'] = points

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    

def create_AutoIndy_infos(dataset_cfg, class_names, data_path, save_path, workers = 4):
    dataset = AutoIndyDataset(dataset_cfg = dataset_cfg, class_names = class_names, root_path = data_path, training = False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('autoIndy_infos_%s.pkl' % train_split)
    val_filename = save_path / ('autoIndy_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'autoIndy_infos_trainval.pkl'
    test_filename = save_path / 'autoIndy_infos_test.pkl'

    print('---------------Start to generate data infos---------------')
    dataset.set_split(train_split)
    AutoIndy_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(AutoIndy_infos_train, f)
    print('AutoIndy info train file is saved to %s' % train_filename)


    dataset.set_split(val_split)
    AutoIndy_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(AutoIndy_infos_val, f)
    print('AutoIndy info val file is saved to %s' % val_filename)


    with open(trainval_filename, 'wb') as f:
        pickle.dump(AutoIndy_infos_train + AutoIndy_infos_val, f)
    print('AutoIndy info trainval file is saved to %s' % trainval_filename)


    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_autoindy_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()


        create_AutoIndy_infos(
            dataset_cfg=dataset_cfg,
            class_names=['car'],
            data_path=ROOT_DIR / 'data' / 'AutoIndy',
            save_path=ROOT_DIR / 'data' / 'AutoIndy'
        )
