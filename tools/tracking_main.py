import os
from collections import defaultdict

import cv2
import torch
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracker.byte_tracker import ByteTracker
from yolox.utils import post_process
from yolox.utils.visualize import plot_tracking_sc, \
    plot_tracking_mc


COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush']


class Opt:
    
    '''
    minimal class to house tracking params
    '''
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


def get_image_list(path):
    """
    :param path:
    :return:
    """
    image_names = []
    for main_dir, sub_dir, file_name_list in os.walk(path):
        for file_name in file_name_list:
            apath = os.path.join(main_dir, file_name)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results_dict(f_path,
                       results_dict,
                       data_type,
                       num_classes=5):
    """
    :param f_path:
    :param results_dict:
    :param data_type:
    :param num_classes:
    :return:
    """
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,{cls_id},1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(f_path, "w", encoding="utf-8") as f:
        for cls_id in range(num_classes):  # process each object class
            cls_results = results_dict[cls_id]
            for fr_id, tlwhs, track_ids in cls_results:  # fr_id starts from 1
                if data_type == 'kitti':
                    fr_id -= 1

                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue

                    x1, y1, w, h = tlwh
                    # x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=fr_id,
                                              id=track_id,
                                              x1=x1, y1=y1, w=w, h=h,
                                              cls_id=cls_id)
                    # if fr_id == 1:
                    #     print(line)

                    f.write(line)
                    # f.flush()

    logger.info('Save results to {}.\n'.format(f_path))


def write_results_mcmot(f_path,
                        frame_id,
                        online_tr_ids_dict,
                        online_tlwhs_dict,
                        id2cls):

    lines = []
    # iterate over class
    for cls_id, tracks in online_tr_ids_dict.items():
        for track_id, track in enumerate(tracks):
            tlwh = online_tlwhs_dict[cls_id][track_id]
            x1, y1, w, h = tlwh
            lines.append(f'{frame_id},{id2cls[cls_id]},{track},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f}\n') # write_line
    with open(f_path, "a+", encoding="utf-8") as f:
        f.writelines(lines)


def write_detection_crops(crop_dir,
                          frame,
                          frame_id,
                          online_tr_ids_dict,
                          online_tlwhs_dict,
                          id2cls):

    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    # iterate over class
    for cls_id, tracks in online_tr_ids_dict.items():
        for track_id, track in enumerate(tracks):
            tlwh = online_tlwhs_dict[cls_id][track_id]
            x1, y1, w, h = tlwh
            # crop specific frame
            crop = frame[int(y1):int(y1+h), int(x1):int(x1+w)]
            crop_path = os.path.join(crop_dir, f'{frame_id}_{id2cls[cls_id]}_{track}.jpg')
            cv2.imwrite(crop_path, crop)

def apply_run_thresholds_to_exp(exp, exp_params):
    # apply run-specific params to exp definition
    for param, value in exp_params.items():
        setattr(exp, param, value)

    return exp



def write_results(file_path, results):
    """
    :param file_path:
    :param results:
    :return:
    """
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(file_path, "w", encoding="utf-8") as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):

                if track_id < 0:
                    continue

                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id,
                                          id=track_id,
                                          x1=round(x1, 1),
                                          y1=round(y1, 1),
                                          w=round(w, 1),
                                          h=round(h, 1),
                                          s=round(score, 2))
                f.write(line)

    logger.info('save results to {}'.format(file_path))


def exists(var):
    return var in globals()


class Predictor(object):
    def __init__(self,
                 model,
                 conf_thresh=0.5,
                 input_size=(448, 768),  # (608, 1088)
                 nms_thresh=0.5,
                 n_classes=80,
                 device="cpu"):
        """
        :param model:
        :param exp:
        :param trt_file:
        :param decoder:
        :param device:
        :param fp16:
        :param reid:
        """
        self.model = model
        self.num_classes = n_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.device = device
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        """
        :param img:
        :param timer:
        :return:
        """
        img_info = {"id": 0}

        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.input_size, self.mean, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        if self.device == "cuda":
            img = img.cuda()

        with torch.no_grad():

            ## ----- forward
            outputs = self.model.forward(img)
            ## -----

            ## post process detections
            outputs = post_process(prediction=outputs, num_classes=self.num_classes,
                                   conf_thre=self.conf_thresh, nms_thre=self.nms_thresh)

            return outputs, img_info


def track_video(predictor, cap, save_path, save_video, exp_params):
    """
    online or offline tracking
    :param predictor:
    :param cap:
    :param vid_save_path:
    :param opt:
    :return:
    """

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # int

    vid_save_path = os.path.abspath(save_path['video'])

    if save_video:
        vid_writer = cv2.VideoWriter(vid_save_path,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps,
                                     (int(width), int(height)))

    ## ---------- initialize tracker
    # tracker = ByteTracker(opt, frame_rate=30)
    opt = Opt(exp_params)

    # Todo: Enable external frame rate modification
    tracker = ByteTracker(frame_rate=30, delta_t=3, opt=opt)

    ## ----------

    ## ----- class name to class id and class id to class name
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)

    # change cls_id, cls_na
    for cls_id, cls_name in enumerate(tracker.class_names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    net_size = opt.input_size

    frame_id = 0
    results = []

    while True:

        ## ----- read the video
        ret_val, frame = cap.read()

        if ret_val:
            outputs, img_info = predictor.inference(frame)

            dets = outputs[0]

            if dets is not None:

                ## ----- update the frame
                input_size = [img_info['height'], img_info['width']]

                # update byte
                online_dict = tracker.update_byte_enhance2(dets, input_size, net_size)

                ## ----- plot single-class multi-object tracking results
                if tracker.n_classes == 1:
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for track in online_targets:
                        tlwh = track.tlwh
                        tid = track.track_id

                        if tlwh[2] * tlwh[3] > opt.min_box_area:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(track.score)

                    results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))

                    online_img = plot_tracking_sc(img_info['raw_img'],
                                                  online_tlwhs,
                                                  online_ids,
                                                  frame_id=frame_id + 1)

                ## ----- plot multi-class multi-object tracking results
                elif tracker.n_classes > 1:
                    ## ---------- aggregate current frame's results for each object class
                    online_tlwhs_dict = defaultdict(list)
                    online_tr_ids_dict = defaultdict(list)
                    for cls_id in range(tracker.n_classes):  # process each object class
                        online_targets = online_dict[cls_id]
                        for track in online_targets:

                            # TODO: Check if min_box_size works
                            if track.tlwh[-2] * track.tlwh[-1] > opt.min_box_area:
                                online_tlwhs_dict[cls_id].append(track.tlwh)
                                online_tr_ids_dict[cls_id].append(track.track_id)

                    online_img = plot_tracking_mc(img=img_info['raw_img'],
                                                  tlwhs_dict=online_tlwhs_dict,
                                                  obj_ids_dict=online_tr_ids_dict,
                                                  num_classes=tracker.n_classes,
                                                  frame_id=frame_id + 1,
                                                  id2cls=id2cls)

            else:
                online_img = img_info['raw_img']

            if save_video:
                # Write video of tracks
                vid_writer.write(online_img)



            # write detections to log
            write_results_mcmot(f_path=save_path['log'],
                                frame_id=frame_id,

                                # Todo: clean up this hack to allow for writing missing rows
                                online_tr_ids_dict=online_tr_ids_dict if exists('online_tr_ids_dict') else defaultdict(list),
                                online_tlwhs_dict=online_tlwhs_dict if exists('online_tlwhs_dict') else defaultdict(list),
                                id2cls=id2cls)

                # TODO: configure writing to disk
                # # write detection crops to disk

                # write_detection_crops(crop_dir=os.path.join(vid_save_path.split('.')[0],'crops'),
                #                       frame=img_info['raw_img'],
                #                       frame_id=frame_id,
                #                       online_tr_ids_dict=online_tr_ids_dict,
                #                       online_tlwhs_dict=online_tlwhs_dict,
                #                       id2cls=id2cls)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            print("Read frame {:d} failed!".format(frame_id))
            break

        ## ----- update frame id
        frame_id += 1

    print("{:s} saved.".format(vid_save_path))


def tracker_wrapper(predictor, input_path, output_dir, save_video, exp_params):
    """
    :param predictor:
    :param vis_dir:
    :param current_time:
    :param opt:
    :return:
    """

    mode = 'videos' if os.path.isdir(input_path) else 'video'

    if mode == "videos":
        if os.path.isdir(input_path):

            # TODO: Check if support for other video types is necessary
            mp4_path_list = [input_path + "/" + x for x in os.listdir(input_path)
                             if x.endswith(".mp4")]
            mp4_path_list.sort()
            if len(mp4_path_list) == 0:
                logger.error("empty mp4 video list.")
                exit(-1)

            for video_path in mp4_path_list:
                if os.path.isfile(video_path):

                    # TODO: Check if this is the right structure for determining video_name
                    video_name = os.path.split(video_path)[-1][:-4]
                    print(f"\nStart tracking video {video_name} offline...")

                    ## ----- video capture
                    cap = cv2.VideoCapture(video_path)
                    ## -----

                    save_dir = os.path.join(output_dir, video_name)
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)

                    save_path = {'log': os.path.join(save_dir, "log.txt"),
                                 'video': os.path.join(save_dir, "video.mp4")}

                    ## ---------- Get tracking results
                    track_video(predictor=predictor,
                                cap=cap,
                                exp_params=exp_params,
                                save_path=save_path,
                                save_video=save_video)

                    ## ----------
                    print(f"{video_name} tracking offline done")

    elif mode == "video":
        input_path = os.path.abspath(input_path)

        if os.path.isfile(input_path):
            # Todo: check if this gets the video name correctly
            video_name = input_path.split("/")[-1][:-4]
            print(f"Start tracking video {video_name} offline...")

            ## ----- video capture
            cap = cv2.VideoCapture(input_path)
            ## -----

            save_dir = os.path.join(output_dir, video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            save_path = {'log': os.path.join(save_dir, "log.txt"),
                         'video': os.path.join(save_dir, "video.mp4")}

            ## ---------- Get tracking results
            track_video(predictor=predictor,
                        cap=cap,
                        exp_params=exp_params,
                        save_path=save_path,
                        save_video=save_video)

            ## ----------

            print(f"{video_name} tracking done offline.")
            logger.info(f'Save results to {save_path["log"]}')

        else:
            logger.error(f"invalid video path: {input_path}, exit now!")
            exit(-1)


def run_tracker(input_path,
                output_dir,
                exp_file='/Users/dschmidt/PycharmProjects/temp-video-tracking/tracking_utils/yolox_tiny_det.py',
                ckpt_path='/Users/dschmidt/PycharmProjects/temp-video-tracking/pretrained/yolox_tiny_32.8.pth',
                save_video=False,
                n_classes=80,
                class_names=COCO_CLASSES,
                input_size=(608, 1088),
                conf_thresh=0.5,
                nms_thresh=0.5,
                low_det_thresh=0.1,
                match_thresh=0.8,
                low_match_thresh=0.5,
                unconfirmed_match_thresh=0.7,
                iou_thresh=0.2,
                track_thresh=0.6,
                track_buffer=240,
                min_box_area=10000,
                device='cuda' if torch.cuda.is_available() else 'cpu'):

    """

    :return:
    """


    # initialize model

    ## ----- Define the network
    logger.info("Accessing exp...")
    exp = get_exp(exp_file, None)

    # ----- update exp base based off of run_tracker args
    exp_params = {'n_classes': n_classes,
                  'class_names': class_names,
                  'input_size': input_size,
                  'iou_thresh': iou_thresh,
                  'conf_thresh': conf_thresh,
                  'nms_thresh': nms_thresh,
                  'track_thresh': track_thresh,
                  'track_buffer': track_buffer,
                  'min_box_area': min_box_area,
                  'low_det_thresh': low_det_thresh,
                  'match_thresh': match_thresh,
                  'low_match_thresh': low_match_thresh,
                  'unconfirmed_match_thresh': unconfirmed_match_thresh}

    exp = apply_run_thresholds_to_exp(exp, exp_params)
    logger.info("Getting model...")

    net = exp.get_model()
    if device == "cuda":
        net.cuda()

    net.eval()
    ## ------ apply pretrained ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # load the model state dict
    logger.info("Loading checkpoint...")
    net.load_state_dict(ckpt["model"])
    logger.info(f"Checkpoint {ckpt_path} has been sucessfully loaded")

    ## ---------- Define the predictor
    predictor = Predictor(model=net,
                          conf_thresh=conf_thresh,
                          input_size=input_size,
                          nms_thresh=nms_thresh,
                          n_classes=n_classes,
                          device=device)

    ## ----------
    # run tracker
    tracker_wrapper(predictor=predictor,
                    input_path=input_path,
                    output_dir=output_dir,
                    save_video=save_video,
                    exp_params=exp_params)


if __name__ == "__main__":

    # Examples
    ## ----- run tracking (directory)

    run_tracker(

        # Required Args
        input_path='/Users/dschmidt/PycharmProjects/temp-video-tracking/input/347098902.mp4', # either folder for multiple videos or single video
        output_dir='/Users/dschmidt/PycharmProjects/temp-video-tracking/output',

        # Optional args
        exp_file='/Users/dschmidt/PycharmProjects/temp-video-tracking/tracking_utils/yolox_tiny_det.py',
        ckpt_path='/Users/dschmidt/PycharmProjects/temp-video-tracking/pretrained/yolox_tiny_32.8.pth',
        save_video=True,
        n_classes=80,
        class_names=COCO_CLASSES,
        input_size=(448, 768),

        # detection thresholds
        conf_thresh=0.5,
        nms_thresh=0.5,
        low_det_thresh=0.1,

        # tracking thresholds
        match_thresh=0.8,
        low_match_thresh=0.5,
        unconfirmed_match_thresh=0.7,
        iou_thresh=0.2,
        track_thresh=0.6,
        track_buffer=240,
        min_box_area=10000,
        device='cuda' if torch.cuda.is_available() else 'cpu')