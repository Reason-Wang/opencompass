import copy

from mmengine.config import ConfigDict

from opencompass.registry import LOAD_DATASET, MODELS


def build_dataset_from_cfg(dataset_cfg: ConfigDict):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_cfg.pop('infer_cfg', None)
    dataset_cfg.pop('eval_cfg', None)
    return LOAD_DATASET.build(dataset_cfg)


def build_model_from_cfg(model_cfg: ConfigDict):
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg.pop('run_cfg', None)
    model_cfg.pop('max_out_len', None)
    model_cfg.pop('batch_size', None)
    model_cfg.pop('abbr', None)
    model_cfg.pop('summarizer_abbr', None)
    model_cfg.pop('pred_postprocessor', None)
    model_cfg.pop('min_out_len', None)
    return MODELS.build(model_cfg)


def build_model_from_cfg_with_context(model_cfg: ConfigDict, work_dir: str = None):
    """Build model from config with work directory context injection.
    
    Args:
        model_cfg (ConfigDict): Model configuration
        work_dir (str, optional): Work directory for the evaluation context
    
    Returns:
        Model instance with work directory context set
    """
    # Set work directory context if provided
    if work_dir:
        try:
            from opencompass.models.custom.custom_api_model import set_work_dir_context
            set_work_dir_context(work_dir)
        except ImportError:
            # If custom module is not available, just continue without setting context
            pass
    
    return build_model_from_cfg(model_cfg)
