#
# Copyright (C) 2024, Gaga
# Gaga research group, https://github.com/weijielyu/Gaga
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco
#

from argparse import ArgumentParser, Namespace
import sys
import os


# === 根目录推断：以当前文件为基准，向上一层就是 repo 根（workspace/Gaga） ★★★
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_ROOT = os.path.join(REPO_ROOT, "dataset")
MODEL_ROOT   = os.path.join(REPO_ROOT, "model")
OUTPUT_ROOT  = os.path.join(REPO_ROOT, "output")


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3

        # ===== 新增的“纯名字”参数（用户输入）——注意这里没有下划线！！！ =====
        self.scene = ""   # -> 只注册 --scene，没有 -s
        self.model = ""   # -> 只注册 --model，没有 -m
        self.output = ""  # -> 只注册 --output，没有 -o

        # ===== 内部使用的真实路径（保持兼容） =====
        self._source_path = ""          # -> --source_path / -s
        self._model_path = ""           # -> --model_path / -m
        self._trained_model_path = ""   # -> --trained_model_path / -t 之类

        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.n_views = 100
        self.random_init = False
        self.train_split = False
        self._object_path = "fused_mask"
        self.num_classes = 200
        self.lift = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)

        # ========== 统一路径规则（只在这里拼路径） ★★★ ==========
        #
        #   scene  -> source_path = <REPO_ROOT>/dataset/<scene>
        #   model  -> model_path  = <REPO_ROOT>/model/<model>
        #   output -> trained_model_path = <REPO_ROOT>/output/<output>
        #
        #   同时保留老的用法：
        #   - 如果 cfg_args 里已经有 source_path / model_path / trained_model_path，
        #     而且命令行没传 scene/model/output，就直接用 cfg_args 里的绝对路径。

        # ---- source_path / scene ----
        scene_name = getattr(g, "scene", "")
        if scene_name:  # 用户 / cfg_args 里有 scene
            g.source_path = os.path.join(DATASET_ROOT, scene_name)
        else:
            # 保留 cfg_args 里的 source_path（如果有的话）
            if getattr(g, "source_path", ""):
                g.source_path = os.path.abspath(g.source_path)
            else:
                # 都没有就默认 dataset 根（不强制要求）
                g.source_path = DATASET_ROOT

        # ---- model_path / model ----
        model_name = getattr(g, "model", "")
        if model_name:
            g.model_path = os.path.join(MODEL_ROOT, model_name)
        else:
            if getattr(g, "model_path", ""):
                g.model_path = os.path.abspath(g.model_path)
            # 否则保持为空，由上层决定要不要强制要求

        # ---- trained_model_path / output ----
        output_name = getattr(g, "output", "")
        if output_name:
            g.trained_model_path = os.path.join(OUTPUT_ROOT, output_name)
        else:
            if getattr(g, "trained_model_path", ""):
                g.trained_model_path = os.path.abspath(g.trained_model_path)
            # 否则保持为空

        # 最终确保 source_path 是绝对路径（你原来的行为）
        g.source_path = os.path.abspath(g.source_path)

        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 10_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        self.reg3d_interval = 2
        self.reg3d_k = 5
        self.reg3d_lambda_val = 2
        self.reg3d_max_points = 300000
        self.reg3d_sample_size = 1000

        super().__init__(parser, "Optimization Parameters")


class RenderParams(ParamGroup):
    def __init__(self, parser):
        self.iteration = -1
        self.skip_train = False
        self.skip_test = False
        self.render_video = False
        self.fps = 30
        super().__init__(parser, "Rendering Parameters")


def get_combined_args(parser: ArgumentParser):
    """
    先解析命令行，然后尝试从 <model 根>/model/<model>/cfg_args 读配置，
    再用命令行覆盖 cfg_args 中的同名参数。
    """
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    cfgfilepath = None

    # ==== 优先级：显式 model_path > model 名字 ★★★ ====
    try:
        model_dir = None

        # 1) 老用法：直接给 --model_path（绝对路径或相对路径）
        if hasattr(args_cmdline, "model_path") and args_cmdline.model_path:
            model_dir = args_cmdline.model_path
            if not os.path.isabs(model_dir):
                model_dir = os.path.abspath(model_dir)

        # 2) 新用法：只给 --model 名字，我们拼到 REPO_ROOT/model/<model>
        elif hasattr(args_cmdline, "model") and args_cmdline.model:
            model_dir = os.path.join(MODEL_ROOT, args_cmdline.model)

        if model_dir is not None:
            cfgfilepath = os.path.join(model_dir, "cfg_args")
            print("Looking for config file in", cfgfilepath)
            with open(cfgfilepath) as cfg_file:
                print("Config file found: {}".format(cfgfilepath))
                cfgfile_string = cfg_file.read()
        else:
            print("No model or model_path specified on command line; skip loading cfg_args.")
    except (TypeError, FileNotFoundError):
        print("Config file not found at", cfgfilepath)
        pass

    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
