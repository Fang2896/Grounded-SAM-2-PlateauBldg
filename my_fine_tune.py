import os
import cv2
import torch
import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor



# 读取随机图像及其标注
def read_batch(data):
    ent = data[np.random.randint(len(data))]
    img = cv2.imread(ent['image'])[...,::-1] # BGR -> RBG
    mask_img = cv2.imread(ent['mask'], 0)

    scale_factor = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)))
    mask_img = cv2.resize(mask_img, (int(mask_img.shape[1] * scale_factor), int(mask_img.shape[0] * scale_factor)), interpolation=cv2.INTER_NEAREST)

    # 目前其实label只有一个255，代表main facade
    inds = np.unique(mask_img)[1:]
    points = []
    masks = []
    for ind in inds:
        mask = (mask_img == ind).astype(np.uint8)
        masks.append(mask)
        coords = np.argwhere(mask > 0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    
    return img, np.array(masks), np.array(points), np.ones([len(masks), 1])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取数据集
    data = []

    image_data_dir = './data/plateau/'
    mask_data_dir = './data/plateau_mask/'

    for ff, name in enumerate(os.listdir(mask_data_dir)):
        data.append({"image" : image_data_dir + name[:-4] + ".jpg", "mask" : mask_data_dir + name})

    sam2_checkpoint = 'D:/File/Research/Code/Grounded-SAM-2/checkpoints/sam2_hiera_small.pt'
    model_cfg="D:/File/Research/Code/Grounded-SAM-2/sam2_configs/sam2_hiera_s.yaml"
    sam2_model=build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor=SAM2ImagePredictor(sam2_model)    # 加载网络
    
    predictor.model.sam_mask_decoder.train(True)    # 启用掩码解码器的训练
    predictor.model.sam_prompt_encoder.train(True)  # 启用提示编码器的训练

    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
    scaler = torch.amp.GradScaler('cuda') # 设置混合精度

    # 训练主循环
    for iter in range(30000):
        with torch.amp.autocast('cuda'):
            image, mask, input_point, input_label = read_batch(data)
            
            if mask.shape[0] == 0:
                continue
            
            # 对图像应用SAM图像编码器
            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

            batched_mode = unnorm_coords.shape[0] > 1   # multi mask prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
            # 网络预测
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # Upscale the masks to the original image resolution

            # 1. 分割损失
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            gt_mask = torch.tensor(mask, dtype=torch.float32).to(device)

            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

            # 2. 分数损失
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            
            loss = seg_loss + score_loss*0.05  # 混合损失

            # 反向传播和保存模型
            predictor.model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if iter % 1000 == 0:
                torch.save(predictor.model.state_dict(), "model.torch") # 保存模型
            
            if iter == 0: 
                mean_iou=0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print("step)", iter, "Accuracy(IOU) = ", mean_iou)


if __name__ == "__main__":
    main()


