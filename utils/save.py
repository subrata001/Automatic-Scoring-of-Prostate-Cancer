import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
from typing import Dict, Tuple, Any

SAVE_PATH = './result/'

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_join(base: str, *parts: str) -> str:
    
    return os.path.normpath(os.path.join(base, *parts))


def save(
    data: Any,
    wsi_name: str,
    pattern_color: Dict[str, str],
    func_mode: str,
    save_path: str = SAVE_PATH,
    **kwargs: Any,
) -> None:

    figsize: Tuple[int, int] = kwargs.get("figsize", (15, 10))
    dpi: int = kwargs.get("dpi", 100)

    mode = func_mode.lower().strip()
    if mode not in {"json", "image", "count"}:
        raise ValueError(f"Unsupported func_mode: {func_mode!r}")

    _ensure_dir(save_path)

    if mode == "json":
        if not isinstance(data, dict):
            raise TypeError("For func_mode='json', data must be dict")
        save_json(data, wsi_name, save_path, pattern_color)
        return

    if mode == "image":
        if not isinstance(data, np.ndarray):
            raise TypeError("For func_mode='image', data must be numpy.ndarray")
        save_img(data, wsi_name, save_path, pattern_color, figsize, dpi)
        return

    # mode == "count"
    if not isinstance(data, dict):
        raise TypeError("For func_mode='count', data must be dict")
    out_file = _safe_join(save_path, f"{wsi_name}_count.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f)


# def save(data, 
#          wsi_name:str,
#          pattern_color:Dict[str, str],
#          func_mode:str,
#          save_path=SAVE_PATH,
#          **kwargs):
    
#     try:figsize = kwargs['figsize']
#     except KeyError:figsize = (15,10)
    
#     try:dpi = kwargs['dpi']
#     except KeyError:dpi = 100
        
#     if func_mode.lower() == 'json':
#         assert type(data) == dict
#         # save_path += 'json/'
#         save_json(data, wsi_name, save_path, pattern_color)
#     elif func_mode.lower() == 'image':
#         assert type(data) == np.ndarray
#         # save_path += 'image/'
#         save_img(data, wsi_name, save_path, pattern_color, figsize, dpi)
#     elif func_mode.lower() == 'count':
#         assert type(data) == dict:
#         with open(SAVE_PATH+wsi_name+'_count.json', 'w') as f:
#             json.dump(data, f)


def save_img(
    overlap_img: np.ndarray,
    wsi_name: str,
    save_path: str,
    label_and_color: Dict[str, str],
    figsize: Tuple[int, int],
    dpi: int,
) -> None:
    _ensure_dir(save_path)

    handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in label_and_color.items()
    ]

    plt.figure(figsize=figsize)
    if handles:  # why: matplotlib errors if ncols=0
        plt.legend(handles=handles, loc="best", ncols=len(handles))
    plt.imshow(overlap_img)  # let matplotlib infer colormap
    plt.axis("off")

    out_png = _safe_join(save_path, f"{wsi_name}.png")
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()

# def save_img(overlap_img: np.ndarray, 
#              wsi_name:str,
#              save_path:str,
#              label_and_color:Dict[str, str],
#              figsize: Tuple[int, int], 
#              dpi: int):
    
#     handles = [mpatches.Patch(color=color, label=label) for label, color in label_and_color.items()]
    
#     plt.figure(figsize=figsize)
#     plt.legend(handles=handles, loc='best', ncols=len(handles)) # loc='upper right'
#     plt.imshow(overlap_img, cmap='gray')
#     plt.axis('off')

#     plt.savefig(save_path + wsi_name + '.png', dpi=dpi) # format choice?

# def save_json(pattern_json:dict, 
#               wsi_name:str, 
#               save_path:str, 
#               label_and_color:dict):
    
#     label_and_color = label_and_color.copy()
#     del label_and_color['normal']

#     if not os.path.exists(save_path + wsi_name) or os.path.isfile(save_path + wsi_name):
#         os.mkdir(save_path + wsi_name)
    
#     for pattern, data in pattern_json.items():
#         json_data = {'color':label_and_color[pattern]}
#         json_data.update(data)
#         with open(save_path + wsi_name + '/' + pattern + '.json', 'w') as f:
#             json.dump(json_data, f)

def save_json(
    pattern_json: Dict[str, Any],
    wsi_name: str,
    save_path: str,
    label_and_color: Dict[str, str],
) -> None:
    # copy to avoid caller mutation; ignore 'normal' if missing
    label_and_color = dict(label_and_color)
    label_and_color.pop("normal", None)

    folder = _safe_join(save_path, wsi_name)
    _ensure_dir(folder)

    for pattern, pdata in pattern_json.items():
        color = label_and_color.get(pattern)
        json_data = {"color": color}
        if isinstance(pdata, dict):
            json_data.update(pdata)
        out_file = _safe_join(folder, f"{pattern}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f)


            