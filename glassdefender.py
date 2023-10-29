import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import torch
import numpy as np

# Flask 통신을 하기 위한 라이브러리
import requests

# alarm 파일 불러오기
from edge_detection import alarm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

# 합성곱 신경망을 구현하기 위한 라이브러리
import torch.nn as nn
import torch.optim as optim

# CNN 모델로 사전학습된 모델을 사용하기 위한 라이브러리 가져오기
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import transforms
from PIL import Image

# mobilenetv2에 넣기 위해 이미지 리사이즈 및 정규화
def preprocess_mobilenetv2_image_from_numpy(image_np):
    # 넘파이 배열을 PIL 이미지로 변환
    img = Image.fromarray(np.uint8(image_np))

    # MobilenetV2 모델을 위한 전처리
    preprocess = transforms.Compose([
        transforms.Resize(256),  # 이미지 크기 조정
        transforms.CenterCrop(224),  # 중앙 부분을 잘라냄 (MobilenetV2의 입력 크기)
        transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
    ])
    # 전처리 적용
    img = preprocess(img)

    return img


#사전학습된 mobilenetv2를 이용하여 자른 이미지를 판별한다

model = models.mobilenet_v2(pretrained=True)
checkpoint = torch.load('mobilenetv2.pt', map_location = torch.device('cpu'))
num_classes = 3  # 새 클래스 수에 맞게 설정
model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
model.load_state_dict(checkpoint)
model.eval()
def mobilenetv2(image):

    input_data = image.unsqueeze(0)

    with torch.no_grad():
        output = model(input_data)

    _, predicted = output.max(1)
    label = predicted.item()

    return label

server_url = "http://192.168.0.161:5000"

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    # stream_url 가져오기
    stream_url = 'http://[192.168.0.161]:8080/?action=stream'
    source = stream_url
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # 여기서부터 반복문 계속 반복
    for path, im, im0s, vid_cap, s in dataset:

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                is_danger = False
                for *xyxy, conf, cls in reversed(det):

                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    #print(f'x좌표 : {x1}, {x2}')
                    #print(f'y좌표 : {y1}, {y2}')


                    if c == 39 or c == 40 or c == 41:
                        # x좌표와 y좌표 추출
                        x1, y1, x2, y2 = map(int, xyxy)

                        # 무게 중심 추출
                        center_x = x1 + (x2 - x1) / 2
                        center_y = y2

                        # 잘린 이미지를 담는다
                        cut_image = im0[y1:y2, x1:x2]
                        cut_image0 = cut_image.copy()

                        if save_csv:
                            write_to_csv(p.name, label, confidence_str)

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            cnt = 0  # 위험 물체에 대한 알람을 주기 위한 함수
                            if c == 39:
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                print(cls)
                                cnt += 1
                            else:
                                # mobilenetv2의 결과에 따라 label을 준다(40, 41일 때)
                                cut_img = preprocess_mobilenetv2_image_from_numpy(cut_image0)
                                label = mobilenetv2(cut_img)
                                if label == 0:
                                    continue
                                elif label == 1:
                                    label = str(label)
                                    label = 'beer_cup'
                                    cnt += 1
                                elif label == 2:
                                    label == str(label)
                                    label = 'soju_cup'
                                    cnt += 1
                            # bottle(병), 소주컵, 맥주컵일 경우 위험 라인에 걸쳐져 있을 경우 알람을 줌
                            if cnt == 1:
                                # 무게중심이 일정반경안에 들어와 있을 경우 알람을 주기 위한 코드
                                if alarm(center_x, center_y) == True:
                                    is_danger = True


                            # 바운딩 박스 주변에 레이블을 추가한다
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            # 바운딩 박스에 해당하는 이미지를 저장한다 이 함수의 결과를 save_dir / 'crops'/ names[c]/ f'{p.stem}
                            # 경로로 저장한다.
                            save_one_box(xyxy, im0, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                if is_danger == True:
                    # LED를 켜는 요청을 보냄
                    print("빨간불이 켜졌습니다.")
                    response = requests.post(server_url + '/control_led', json={'action': 'on'})
                    print(response.text)
                else:
                    print("알람이 꺼졌습니다.")
                    # LED를 끄는 요청을 보냄
                    response = requests.post(server_url + '/control_led', json={'action': 'off'})
                    print(response.text)
            lines = [
                [(0, 449), (156, 255)],
                [(156, 255), (470, 255)],
                [(470, 255), (623, 444)]
            ]

            lines_2 = [
                [(57, 449), (183, 270)],
                [(183, 270), (441, 270)],
                [(441, 270), (566, 444)]
            ]

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # 책상 윤곽선을 실시간 스트림으로 나타내기 위한 코드
                for line in lines:
                    pt1, pt2 = line
                    cv2.line(im0, pt1, pt2, (255, 0, 0), 2)
                for line in lines_2:
                    pt1, pt2 = line
                    cv2.line(im0, pt1, pt2, (0, 0, 255), 2)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
