
import torch
import hydra
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from utils.clip_scribe_logging import logger


def build_sam2_video_predictor(
        config_file:str,
        ckpt_path:str,
        device:str="cpu",
        mode:str="eval",
        apply_postprocessing=True,
):

    with hydra.initialize_config_module(config_module="src.sam2.configs", version_base=None):

        cfg = compose(config_name=config_file)

        OmegaConf.set_struct(cfg, False)
        OmegaConf.resolve(cfg)

        # Post-processing Overrides (Manual logic instead of complex Hydra overrides)
        if apply_postprocessing:
            if "sam_mask_decoder_extra_args" not in cfg.model:
                cfg.model.sam_mask_decoder_extra_args = {}

            cfg.model.sam_mask_decoder_extra_args["dynamic_multimask_via_stability"] = True
            cfg.model.sam_mask_decoder_extra_args["dynamic_multimask_stability_delta"] = 0.05
            cfg.model.sam_mask_decoder_extra_args["dynamic_multimask_stability_thresh"] = 0.98
            cfg.model.binarize_mask_from_pts_for_mem_enc = True
            cfg.model.fill_hole_area = 8

        # This looks at the '_target_' keys in YAML
        model = instantiate(cfg.model, _recursive_=True)

        _load_checkpoint(model, ckpt_path)
        model = model.to(device)

        if mode == "eval":
            model.eval()

        return model


def _load_checkpoint(model, ckpt_path):

    if ckpt_path:

        logger.info(f"Loading weights from {ckpt_path}...")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]

        missing_keys, unexpected_keys = model.load_state_dict(sd)

        if missing_keys:
            logger.error(f"Missing keys: {missing_keys}")
            raise RuntimeError("Architecture mismatch: Keys missing")
        if unexpected_keys:
            logger.error(f"Unexpected keys: {unexpected_keys}")
            raise RuntimeError("Architecture mismatch: Unexpected keys found")

        logger.info("Loaded checkpoint successfully")
