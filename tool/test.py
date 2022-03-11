import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import argparse
import os
import cv2
import glob
import copy
import numpy as np
import torch
import imutils
import imageio
from PIL import Image
import scipy.ndimage
from skimage.feature import canny
import torchvision.transforms.functional as F

from RRRR import utils
from RRRR import RAFT

import utils.region_fill as rf
from utils.Poisson_blend import Poisson_blend
from utils.Poisson_blend_img_test import Poisson_blend_img
from get_flowNN import get_flowNN
from get_flowNN_gradient_test import get_flowNN_gradient
from utils.common_utils import flow_edge
from spatial_inpaint_test import spatial_inpaint
from frame_inpaint import DeepFillv1
from edgeconnecttest.networks import EdgeGenerator_


















def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
        pass
    print(dir)

def initialize_RAFT(args):
    """Initializes the RAFT model.
    """
    # 利用多GPU加速，只有input数据并行，本地只有一个其实用不上 https://zhuanlan.zhihu.com/p/102697821
    model = torch.nn.DataParallel(RAFT(args))
    #state_dict就是一个简单的Python dictionary，其功能是层与 层的参数张量之间一一映射
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda')
    model.eval()

    return model


def infer_flow(args, mode, filename, image1, image2, imgH, imgW, model, homography=False):

    if not homography:
        _, flow = model(image1, image2, iters=20, test_mode=True)
#         print(_)
        print(flow.shape)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        print(flow.shape)
    else:
        print("infer_flow homography -----------------------------------------------------")

        pass
    
    return flow



def calculate_flow(args, model, video):
    """Calculates optical flow.
    """
    nFrame, _, imgH, imgW = video.shape
    FlowF = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    FlowB = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    FlowNLF = np.empty(((imgH, imgW, 2, 3, 0)), dtype=np.float32)
    FlowNLB = np.empty(((imgH, imgW, 2, 3, 0)), dtype=np.float32)
    # FlowNLF0 = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    # FlowNLF1 = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    # FlowNLF2 = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    # FlowNLB0 = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    # FlowNLB1 = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    # FlowNLB2 = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    print(nFrame, _, imgH, imgW)
    
    if args.Nonlocal:
        mode_list = ['forward', 'backward', 'nonlocal_forward', 'nonlocal_backward']
    else:
        mode_list = ['forward', 'backward']
        pass
    print(mode_list)
    
    for mode in mode_list:
        create_dir(os.path.join(args.outroot, 'flow', mode + '_flo'))
        create_dir(os.path.join(args.outroot, 'flow', mode + '_png'))
        
        """# 内部计算不会被跟踪记录，梯度
        单张 'forward', 'backward' 不执行
        with torch.no_grad():
            # 合成一张，单张此步骤去掉？
            for i in range(nFrame):
                if mode == 'forward':
                    if i == nFrame - 1:
                        continue
                    print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
                    image1 = video[i, None]
                    image2 = video[i + 1, None]
                    flow = infer_flow(args, mode, '%05d'%i, image1, image2, imgH, imgW, model, homography=False)
        """
    return FlowF, FlowB, FlowNLF, FlowNLB

def gradient_mask(mask):
    # np.logical_or.reduce 或 对应值中有真即真，全假则假
    # 去掉第一行，第一列
    # (124, 305) zeros(1, 305) axis=0 一行 0 放到最后一行
    # (125, 304) zeros(125, 1) axis=1 一列 0 拼到每行结尾
    # 补入的 0 原本是 1 的改成 1，最后一个值改不了 
    gradient_mask = np.logical_or.reduce((mask,
        np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=bool)), axis=0),
        np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=bool)), axis=1)))

    return gradient_mask

def complete_flow(args, corrFlow, flow_mask, mode, edge=None):
    """Completes flow.
    """
    if mode not in ['forward', 'backward', 'nonlocal_forward', 'nonlocal_backward']:
        raise NotImplementedError
        pass
    print(mode)
    print(corrFlow)
    sh = corrFlow.shape
    imgH = sh[0]
    imgW = sh[1]
    nFrame = sh[-1]
    
    print(nFrame)

    create_dir(os.path.join(args.outroot, 'flow_comp', mode + '_flo'))
    create_dir(os.path.join(args.outroot, 'flow_comp', mode + '_png'))

    compFlow = np.zeros(((sh)), dtype=np.float32)
    
    
    
    return compFlow


def video_completion_seamless(args):

    # Flow model.
    RAFT_model = initialize_RAFT(args)
    # Loads frames.
    filename_list = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))

    # Obtains imgH, imgW and nFrame. Image.open(‘.jpg’)读取的格式为RGBA（其中A表示图像的alpha通道，即RGBA共四个通道）
    # (2991, 2000, 3)
    imgH, imgW = np.array(Image.open(filename_list[0])).shape[:2]
    print(filename_list[0])
    nFrame = len(filename_list)
    # 高，宽 pixels
    print(imgH, imgW)
    
    # Loads video.
    # permute(2, 0, 1) 维度换位，将通道换到前面 (2991, 2000, 3) -> (3, 2991, 2000)
    video = []
    for filename in sorted(filename_list):
        video.append(torch.from_numpy(np.array(Image.open(filename)).astype(np.uint8)[..., :3]).permute(2, 0, 1).float())
        pass
    # 维度拼接，只有单张图片拼不了
    video = torch.stack(video, dim=0)
    print(video.shape)
#     video = video[0].to('cuda')
#     print(video.shape)
    video = video.to('cuda')
    
    # Calcutes the corrupted flow.
    corrFlowF, corrFlowB, corrFlowNLF, corrFlowNLB = calculate_flow(args, RAFT_model, video)
    
    # Makes sure video is in BGR (opencv) format.
    # ::-1 通道翻转，/ 255.归一化
    video = video.permute(2, 3, 1, 0).cpu().numpy()[:, :, ::-1, :] / 255.
    print(video.shape) # (2991, 2000, 3, 1)
    
    # 暂时只做 object_removal，所以不需要判断，也有可能扣除后做修复
    # Loads masks.
    filename_list = glob.glob(os.path.join(args.path_mask, '*.png')) + \
                    glob.glob(os.path.join(args.path_mask, '*.jpg'))
    mask = []
    mask_dilated = []
    flow_mask = []
    for filename in sorted(filename_list):
        # 灰度图像，每个像素用8个bit表示，0表示黑，255表示白
        mask_img = np.array(Image.open(filename).convert('L'))
        # Dilate 15 pixel so that all known pixel is trustworthy 元素膨胀 https://zhuanlan.zhihu.com/p/362042756
        flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=15)
        # Close the small holes inside the foreground objects erode https://www.jianshu.com/p/ea253fff2289
        # cv2.MORPH_ERODE(腐蚀)：它沿着物体边界移除像素并缩小物体的大小，会增强图像的暗部。
        # cv2.MORPH_DILATE(膨胀)：通过将像素添加到该图像中的对象的感知边界，扩张放大图像中的明亮白色区域。
        # cv2.MORPH_OPEN 先腐蚀，后膨胀。能够排除小黑点。cv2.MORPH_CLOSE ：先膨胀，后腐蚀。能够排除小亮点。
        flow_mask_img = cv2.morphologyEx(flow_mask_img.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(bool) # 填充孔洞
        flow_mask.append(flow_mask_img)
        
        mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=5) # 元素膨胀
        mask_img = scipy.ndimage.binary_fill_holes(mask_img).astype(bool) # 填充孔洞
        mask.append(mask_img)
        
        mask_dilated.append(gradient_mask(mask_img))

    # mask indicating the missing region in the video.           array([[[1, 4],[2, 5],[3, 6]],
    #>>> a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])                [[1, 4],[2, 5],[3, 6]],
    #>>> b = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])                [[1, 4],[2, 5],[3, 6]]])
    #>>> np.stack((a, b), axis=-1)
    # 如果有多个标记时拼接，如果只有一个这就没执行
    mask = np.stack(mask, -1).astype(bool)
    mask_dilated = np.stack(mask_dilated, -1).astype(bool)
    flow_mask = np.stack(flow_mask, -1).astype(bool)

    print(args.edge_guide)
#     if args.edge_guide:
#         # Edge completion model.
#         EdgeGenerator = EdgeGenerator_()
    FlowF_edge, FlowB_edge = None, None
    # Completes the flow.
    # corrFlowF 来自于 calculate_flow RRRR，单张没值，先跳过看看
    videoFlowF = complete_flow(args, corrFlowF, flow_mask, 'forward', FlowF_edge)
    videoFlowB = complete_flow(args, corrFlowB, flow_mask, 'backward', FlowB_edge)
    print(args.Nonlocal)
#     if args.Nonlocal:
    videoNonLocalFlowF = None
    videoNonLocalFlowB = None
    print('\nFinish flow completion.')
    
    # Prepare gradients
    gradient_x = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)
    gradient_y = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)
    print(gradient_x, gradient_y)
    
    for indFrame in range(nFrame):
        img = video[:, :, :, indFrame]
        print(img.shape)
        print(indFrame)
        print(mask.shape)
        # print(mask[:, :, indFrame])
        img[mask[:, :, indFrame], :] = 0
        # cv2.inpaint 用邻近的像素替换那些坏标记，使其看起来像是邻居  创建一个与输入图像大小相同的掩码，其中非零像素对应于要修复的区域
        # https://www.cnblogs.com/lfri/p/10618417.html
        img = cv2.inpaint((img * 255).astype(np.uint8), mask[:, :, indFrame].astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32)  / 255.
        
        gradient_x_ = np.concatenate((np.diff(img, axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1)
        gradient_y_ = np.concatenate((np.diff(img, axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)
        gradient_x = np.concatenate((gradient_x, gradient_x_.reshape(imgH, imgW, 3, 1)), axis=-1)
        gradient_y = np.concatenate((gradient_y, gradient_y_.reshape(imgH, imgW, 3, 1)), axis=-1)

        gradient_x[mask_dilated[:, :, indFrame], :, indFrame] = 0
        gradient_y[mask_dilated[:, :, indFrame], :, indFrame] = 0
        pass
    
    iter = 0
    mask_tofill = mask
    gradient_x_filled = gradient_x # corrupted gradient_x, mask_gradient indicates the missing gradient region
    gradient_y_filled = gradient_y # corrupted gradient_y, mask_gradient indicates the missing gradient region
    mask_gradient = mask_dilated
    video_comp = video
    
    # Image inpainting model. https://zhuanlan.zhihu.com/p/47919251
    deepfill = DeepFillv1(pretrained_model=args.deepfill_model, image_shape=[imgH, imgW])

    
    
    # 当 mask 未变化，循环不会结束，先继续，最后加循环
    create_dir(os.path.join(args.outroot, 'frame_seamless_comp_' + str(iter)))
        
    # Gradient propagation.
    gradient_x_filled, gradient_y_filled, mask_gradient = get_flowNN_gradient(args,
                            gradient_x_filled,
                            gradient_y_filled,
                            mask,
                            mask_gradient,
                            videoFlowF,
                            videoFlowB,
                            videoNonLocalFlowF,
                            videoNonLocalFlowB)


    # if there exist holes in mask, Poisson blending will fail. So I did this trick. I sacrifice some value. Another solution is to modify Poisson blending.
    for indFrame in range(nFrame):
        mask_gradient[:, :, indFrame] = scipy.ndimage.binary_fill_holes(mask_gradient[:, :, indFrame]).astype(bool)
        pass

    # After one gradient propagation iteration
    # gradient --> RGB
    print(range(nFrame))
    for indFrame in range(nFrame):
        print("Poisson blending frame {0:3d}".format(indFrame))
        # print(mask[:, :, indFrame].sum())
        if mask[:, :, indFrame].sum() > 0:
            try:
                frameBlend, UnfilledMask = Poisson_blend_img(video_comp[:, :, :, indFrame], gradient_x_filled[:, 0 : imgW - 1, :, indFrame], gradient_y_filled[0 : imgH - 1, :, :, indFrame], mask[:, :, indFrame], mask_gradient[:, :, indFrame])
                # UnfilledMask = scipy.ndimage.binary_fill_holes(UnfilledMask).astype(bool)
            except:
                frameBlend, UnfilledMask = video_comp[:, :, :, indFrame], mask[:, :, indFrame]
                print(frameBlend, UnfilledMask)
                pass
            print(frameBlend.shape, UnfilledMask.shape)
            
            # 小于 0 的 =0，大于1的 = 2
            frameBlend = np.clip(frameBlend, 0, 1.0)
            tmp = cv2.inpaint((frameBlend * 255).astype(np.uint8), UnfilledMask.astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.
            
            frameBlend[UnfilledMask, :] = tmp[UnfilledMask, :]
            video_comp[:, :, :, indFrame] = frameBlend
            mask[:, :, indFrame] = UnfilledMask
            frameBlend_ = copy.deepcopy(frameBlend)
            # Green indicates the regions that are not filled yet.
            frameBlend_[mask[:, :, indFrame], :] = [0, 1., 0]
        else:
            frameBlend_ = video_comp[:, :, :, indFrame]
            pass
        
        # 用绿色覆盖了标记区域
        cv2.imwrite(os.path.join(args.outroot, 'frame_seamless_comp_' + str(iter), '%05d.png'%indFrame), frameBlend_ * 255.)
        pass
    
#     torch.cuda.empty_cache()
#     print(torch.cuda.memory_stats())
    print(torch.cuda.memory_summary())
    print("==========================", torch.cuda.memory_allocated())
    mask, video_comp = spatial_inpaint(deepfill, mask, video_comp)
    iter += 1
    
    # Re-calculate gradient_x/y_filled and mask_gradient
    for indFrame in range(nFrame):
        mask_gradient[:, :, indFrame] = gradient_mask(mask[:, :, indFrame])
        gradient_x_filled[:, :, :, indFrame] = np.concatenate((np.diff(video_comp[:, :, :, indFrame], axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1)
        gradient_y_filled[:, :, :, indFrame] = np.concatenate((np.diff(video_comp[:, :, :, indFrame], axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)
        
        gradient_x_filled[mask_gradient[:, :, indFrame], :, indFrame] = 0
        gradient_y_filled[mask_gradient[:, :, indFrame], :, indFrame] = 0
        pass
    
    create_dir(os.path.join(args.outroot, 'frame_seamless_comp_' + 'final'))
    video_comp_ = (video_comp * 255).astype(np.uint8).transpose(3, 0, 1, 2)[:, :, :, ::-1]
    for i in range(nFrame):
        img = video_comp[:, :, :, i] * 255
        cv2.imwrite(os.path.join(args.outroot, 'frame_seamless_comp_' + 'final', '%05d.png'%i), img)

    
    """
     # We iteratively complete the video.
    while(np.sum(mask) > 0):
        print(np.sum(mask))
        create_dir(os.path.join(args.outroot, 'frame_seamless_comp_' + str(iter)))
        
        # Gradient propagation.
        """

import PIL
from torch.utils.collect_env import get_pretty_env_info

def get_pil_version():
    return "\n        Pillow ({})".format(PIL.__version__)


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += get_pil_version()
    return env_str



def main(args):
    
    print (collect_env_info())

    assert args.mode in ('object_removal', 'video_extrapolation'), (
        "Accepted modes: 'object_removal', 'video_extrapolation', but input is %s"
    ) % mode

    if args.seamless:
        video_completion_seamless(args)
    else:
        video_completion(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # video completion
    parser.add_argument('--seamless', action='store_true', help='Whether operate in the gradient domain')
    parser.add_argument('--edge_guide', action='store_true', help='Whether use edge as guidance to complete flow')
    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--path', default='../data/tennis', help="dataset for evaluation")
    parser.add_argument('--path_mask', default='../data/tennis_mask', help="mask for object removal")
    parser.add_argument('--outroot', default='../result/', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=np.inf, type=float, help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--Nonlocal', action='store_true', help='Whether use edge as guidance to complete flow')

    # RAFT
    parser.add_argument('--model', default='../weight/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # Deepfill
    parser.add_argument('--deepfill_model', default='../weight/imagenet_deepfill.pth', help="restore checkpoint")

    # Edge completion
    parser.add_argument('--edge_completion_model', default='../weight/edge_completion.pth', help="restore checkpoint")

    # extrapolation
    parser.add_argument('--H_scale', dest='H_scale', default=2, type=float, help='H extrapolation scale')
    parser.add_argument('--W_scale', dest='W_scale', default=2, type=float, help='W extrapolation scale')

    args = parser.parse_args()

    main(args)