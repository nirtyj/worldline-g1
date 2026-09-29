"""Build the PolicyServer observation for the N1.7 G1 checkpoint from what the runtime has.

Inputs are the robot-side quantities (docs/groot_arms_design.md §2.3-2.4): the `ego_view` frame (P1's Arena-matched
head camera, 640x480 RGB) and the measured state from SONIC's g1_debug stream (`body_q` 29 in MuJoCo order,
`left/right_hand_q` 7 each in Dex3 order). No ground truth.

Output, exactly what `Gr00tPolicy.check_observation` (strict mode, Isaac-GR00T @ 4b1dca9d
`gr00t/policy/gr00t_policy.py:208-369`) accepts for this checkpoint's modality config (B = 1, T = 1):

    {"video":    {"ego_view": uint8 (1, 1, 480, 640, 3)},
     "state":    {"left_arm": f32 (1, 1, 7), "right_arm": f32 (1, 1, 7), "left_hand": f32 (1, 1, 7),
                  "right_hand": f32 (1, 1, 7), "waist": f32 (1, 1, 3)},
     "language": {"annotation.human.task_description": [[prompt]]}}

GR00T does its own image preprocessing and lower-cases the prompt and strips punctuation (`formalize_language`,
`processing_gr00t_n1d7.py:421`). Never send a crop. The image preprocessing (this checkpoint's processor has
`use_albumentations: true`, so the eval transform is `build_image_transformations_albumentations`,
`gr00t/model/gr00t_n1d7/image_augmentations.py:479-486`):

    LetterBoxPad (black bars to a square: 640x480 -> 640x640, 80 rows top and bottom)
    -> SmallestMaxSize(256, cv2.INTER_AREA) -> FractionalCenterCrop(0.95) -> SmallestMaxSize(256, cv2.INTER_AREA)

The strict check tests dtype, rank, T and C but not H x W (`gr00t_policy.py:243-283`), so the client may do the
first resize itself: `request_image="area256"` sends INTER_AREA(640x480 -> 256x192) (0.15 MB instead of 0.92 MB per
request). The server then pads it to 256x256 (32 rows top and bottom) and its SmallestMaxSize(256) is a copy; because
a 2.5x reduction maps 5 input pixels onto 2 output pixels and the 80-row bars are exactly 32 output rows, that equals
what the server computes from the full frame (`area_downscale_2p5`; checked against the server's own transform on the
dev box: docs/groot_serving.md §9).
"""

from __future__ import annotations

import numpy as np

from . import joint_order as jo

# The instruction this checkpoint was trained and is evaluated with: Arena's HDF5 -> LeRobot converter config
# (`$ARENA/docs/pages/example_workflows/static_apple/step_3_policy_training.rst:96`, `language_instruction`, task_index 3),
# Arena's closed-loop eval config (arena_spike/g1_static_apple_gr00t_closedloop_config.yaml) and the checkpoint's ONNX
# export (`exports/g1-static-apple-b1-480x640/onnx/leapp-0.5.2/README.md:11`). The released dataset's tasks.jsonl
# carries a longer sentence for the same task_index (DATASET_PROMPT), and the model card names that dataset as the
# training data. Open loop the dataset sentence reproduces the demos' arm motion better in 3 of 3 runs
# (docs/groot_serving.md §6.2.1), so it is the default; the closed-loop comparison is open (G2).
ARENA_PROMPT = "move the apple to the plate"
DATASET_PROMPT = "Pick up the apple from the shelf and place it onto the plate on the same shelf next to it."
DEFAULT_PROMPT = DATASET_PROMPT


# What the request carries as `ego_view` (build_observation's request_image):
#   full     the 640x480 frame as rendered (0.92 MB on the wire)
#   area256  the server's first INTER_AREA resize done here: 256x192 (0.15 MB); the model input is the same
REQUEST_IMAGES = ("full", "area256")


def area_downscale_2p5(img: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 with H, W multiples of 5 -> (H*2/5)x(W*2/5)x3 by exact pixel-area averaging: the weights of
    cv2.resize(..., INTER_AREA) for a 2.5x reduction (output pixel j covers source [2.5 j, 2.5 j + 2.5), so per axis
    5 source pixels give 2 outputs with weights (2, 2, 1, 0, 0) / 5 and (0, 0, 1, 2, 2) / 5).

    Integer arithmetic: the 2-D sum is v / 25 with v <= 6375 (uint16), and v / 25 is never a .5 tie (v would have to
    be 25 n + 12.5), so (v + 12) // 25 is the exact rounding; any exact area resize gives the same uint8."""
    a = np.asarray(img)
    h, w, c = a.shape
    if h % 5 or w % 5:
        raise ValueError(f"area_downscale_2p5 needs H and W divisible by 5, got {w}x{h}")
    x = a.reshape(h, w // 5, 5, c).astype(np.uint16)
    cols = np.stack([2 * (x[:, :, 0] + x[:, :, 1]) + x[:, :, 2],
                     x[:, :, 2] + 2 * (x[:, :, 3] + x[:, :, 4])], axis=2).reshape(h // 5, 5, 2 * w // 5, c)
    v = np.stack([2 * (cols[:, 0] + cols[:, 1]) + cols[:, 2],
                  cols[:, 2] + 2 * (cols[:, 3] + cols[:, 4])], axis=1).reshape(2 * h // 5, 2 * w // 5, c)
    return ((v + 12) // 25).astype(np.uint8)


def request_frame(img: np.ndarray, request_image: str = "full") -> np.ndarray:
    """The ego frame as it goes on the wire (REQUEST_IMAGES)."""
    if request_image == "full":
        return img
    if request_image == "area256":
        if tuple(img.shape[:2]) != jo.IMAGE_HW:
            raise ValueError(f"request_image area256 needs the {jo.IMAGE_HW[1]}x{jo.IMAGE_HW[0]} frame, "
                             f"got {img.shape[1]}x{img.shape[0]}")
        return area_downscale_2p5(img)
    raise ValueError(f"request_image must be one of {REQUEST_IMAGES}, got {request_image!r}")


def build_observation(ego_rgb_uint8_HxWx3, body_q_mj29, left_hand_q7, right_hand_q7, prompt: str, *,
                      expect_hw: tuple[int, int] | None = jo.IMAGE_HW, request_image: str = "full") -> dict:
    """Ego frame + g1_debug state + prompt -> the server's observation dict (B = 1, T = 1).

    body_q_mj29: 29 values, MuJoCo/Unitree order (g1_debug `body_q`, absolute rad).
    left_hand_q7 / right_hand_q7: 7 values each, Dex3 order (g1_debug `left_hand_q` / `right_hand_q`).
    expect_hw: the frame size the checkpoint was trained on; None accepts any size (GR00T resizes).
    request_image: "full" sends the frame as it is; "area256" sends the server's own first resize (256x192, the same
    model input, a sixth of the bytes; REQUEST_IMAGES).
    """
    img = np.asarray(ego_rgb_uint8_HxWx3)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"ego frame must be uint8 HxWx3 RGB, got {img.dtype} {img.shape}")
    if expect_hw is not None and tuple(img.shape[:2]) != tuple(expect_hw):
        raise ValueError(f"ego frame is {img.shape[1]}x{img.shape[0]}, the checkpoint expects "
                         f"{expect_hw[1]}x{expect_hw[0]} (Arena head camera, OD1)")
    q = np.asarray(body_q_mj29, dtype=np.float64).reshape(-1)
    lh = np.asarray(left_hand_q7, dtype=np.float64).reshape(-1)
    rh = np.asarray(right_hand_q7, dtype=np.float64).reshape(-1)
    if q.size != 29 or lh.size != jo.N_HAND or rh.size != jo.N_HAND:
        raise ValueError(f"need body q 29 + hands 7 + 7, got {q.size} + {lh.size} + {rh.size}")
    if not (np.isfinite(q).all() and np.isfinite(lh).all() and np.isfinite(rh).all()):
        raise ValueError("non-finite joint state")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    state = jo.groot_state_from_body(q, lh, rh)
    return {
        "video": {jo.VIDEO_KEY: np.ascontiguousarray(request_frame(img, request_image)[None, None])},
        "state": {k: v.astype(np.float32).reshape(1, 1, -1) for k, v in state.items()},
        "language": {jo.LANGUAGE_KEY: [[prompt]]},
    }
