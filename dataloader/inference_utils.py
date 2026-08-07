# dataloader/inference_utils.py

import torch
import torch.nn.functional as F

def sliding_window_predict(model, lq: torch.Tensor,
                            patch_size: int = 128,
                            stride: int = 64,
                            return_patches: bool = False,
                            hq: torch.Tensor | None = None):

    device = lq.device
    C, H, W = lq.shape

    if return_patches:
        assert hq is not None, 'hq must be provided when return_patches=True'

    pred_sum   = torch.zeros((C, H, W), device=device)
    count_map  = torch.zeros((1, H, W), device=device)
    patch_list = []

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    lq_padded = F.pad(lq, (0, pad_w, 0, pad_h), mode='reflect')
    _, H_pad, W_pad = lq_padded.shape

    model.eval()
    with torch.no_grad():
        for y in range(0, H_pad - patch_size + 1, stride):
            for x in range(0, W_pad - patch_size + 1, stride):
                patch = lq_padded[:, y:y+patch_size, x:x+patch_size]
                patch_input = patch.unsqueeze(0)
                patch_pred  = model(patch_input).squeeze(0)

                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                pred_sum [:, y:y_end, x:x_end] += patch_pred[:, :y_end-y, :x_end-x]
                count_map[0, y:y_end, x:x_end] += 1

                if return_patches and y < H and x < W:
                    lq_patch = patch[:, :y_end-y, :x_end-x]
                    pred_patch = patch_pred[:, :y_end-y, :x_end-x]
                    hq_patch = hq[:, y:y_end, x:x_end]
                    if (lq_patch.shape[1] >= patch_size // 2
                            and lq_patch.shape[2] >= patch_size // 2):
                        patch_list.append((lq_patch, hq_patch, pred_patch, y, x))

    count_map = count_map.clamp(min=1)
    stitched = pred_sum / count_map

    if return_patches:
        return stitched, patch_list
    return stitched