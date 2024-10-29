import torch
import numpy as np
import cv2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def read_image(image_path, mask_path = None):
    img = cv2.imread(image_path)[...,::-1]  # RGB

    scale_factor = np.min([1024 / img.shape[1], 1024 / img.shape[0]])  
    img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)))  

    if mask_path:
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

    return img, mask


# 在输入掩码内采样点
def get_points(mask, num_points): 
    points=[]
    for i in range(num_points):
        coords = np.argwhere(mask > 0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return np.array(points)



def main():
    # torch.autocast(device_type='cuda', dtype=torch.bfloat16).enter()

    image_path = "D:/File/Research/Code/Grounded-SAM-2/data/plateau/part_tex_poly_TK040997_p11030_6.jpg"
    image, mask = read_image(image_path)

    num_samples = 50 # 要采样的点/分段数量
    input_points = get_points(mask, num_samples)

    sam2_checkpoint = 'D:/File/Research/Code/Grounded-SAM-2/checkpoints/sam2_hiera_small.pt'
    model_cfg="D:/File/Research/Code/Grounded-SAM-2/sam2_configs/sam2_hiera_s.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    predictor.model.load_state_dict(torch.load("model.torch"))

    with torch.no_grad(): # 防止网络计算梯度(更高效的推理)
        predictor.set_image(image) # 图像编码器
        masks, scores, logits = predictor.predict(  # prompt编码器 + mask解码器
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
    )

    if isinstance(masks, torch.Tensor):
        np_masks = masks[:,0].detach().cpu().numpy()
        np_scores = scores[:,0].detach().float().cpu().numpy()
    else:
        np_masks = masks[:,0]
        np_scores = scores[:,0]
    
    # np_masks = np.array(masks[:,0].cpu().numpy()) # 从torch转换为numpy  
    # np_scores = scores[:,0].float().cpu().numpy() # 从torch转换为numpy  

    shorted_masks = np_masks[np.argsort(np_scores)][::-1] # 根据分数排列掩码
    seg_map = np.zeros_like(shorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(shorted_masks[0], dtype=bool)

    for i in range(shorted_masks.shape[0]):
        mask = shorted_masks[i].astype(bool)
        if(mask * occupancy_mask).sum() / mask.sum() > 0.15: 
            continue 
        mask[occupancy_mask] = 0
        seg_map[mask] = i + 1
        occupancy_mask[mask] = 1

    rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for id_class in range(1,seg_map.max()+1):
        rgb_image[seg_map == id_class] = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

    cv2.imshow("annotation",rgb_image)
    cv2.imshow("mix",(rgb_image / 2 + image / 2).astype(np.uint8))
    cv2.imshow("image",image)
    cv2.waitKey()


if __name__ == "__main__":
    main()








