import torch 
import numpy as np

# def normalize_bbox(bboxes, pc_range):

#     cx = bboxes[..., 0:1]
#     cy = bboxes[..., 1:2]
#     cz = bboxes[..., 2:3]
#     w = bboxes[..., 3:4].log()
#     l = bboxes[..., 4:5].log()
#     h = bboxes[..., 5:6].log()

#     rot = bboxes[..., 6:7]
#     if bboxes.size(-1) > 7:
#         vx = bboxes[..., 7:8] 
#         vy = bboxes[..., 8:9]
#         normalized_bboxes = torch.cat(
#             (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
#         )
#     else:
#         normalized_bboxes = torch.cat(
#             (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
#         )
#     return normalized_bboxes

def normalize_bbox(bboxes, pc_range):
    # all_bbox_preds: (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy)
    # gt_bboxes_list: (cx, cy, cz, l, w, h, rot, vx, vy)
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    rot = bboxes[..., 6:7]
    
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]

        if int(mmdet3d.__version__[0]) >= 1: 
            normalized_bboxes = torch.cat(
                (cx, cy, l, w, cz, h, (- rot - np.pi / 2).sin(), (- rot - np.pi / 2).cos(), vx, vy), dim=-1
                )
        else:
            normalized_bboxes = torch.cat(
                (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
            )
    else:
        if int(mmdet3d.__version__[0]) >= 1: 
            normalized_bboxes = torch.cat(
                (cx, cy, l, w, cz, h, (- rot - np.pi / 2).sin(), (- rot - np.pi / 2).cos()), dim=-1
                )
        else:
            normalized_bboxes = torch.cat(
                (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
            )
    return normalized_bboxes    

def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes