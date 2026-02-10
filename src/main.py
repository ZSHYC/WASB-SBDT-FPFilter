import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from runners import select_runner
from utils import mkdir_if_missing

log = logging.getLogger(__name__)

# @hydra.main(version_base=None, config_name='root', config_path='configs')
@hydra.main(version_base=None, config_name='eval', config_path='configs')
def main(
        cfg: DictConfig
):
    # print(OmegaConf.to_yaml(cfg))
    # print(cfg)
    if cfg['output_dir'] is None:
        cfg['output_dir'] = HydraConfig.get().run.dir
    mkdir_if_missing(cfg['output_dir'])
    
    runner = select_runner(cfg)
    runner.run()

if __name__ == "__main__":
    main()


# # cd d:\Personal\Desktop\WASB-SBDT-FPFilter\src

#  python main.py --config-name=eval dataset=tennis_predict model=wasb detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar runner.split=test runner.vis_result=True detector.step=3  #步长为3
#  python main.py --config-name=eval dataset=tennis_predict model=wasb detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar runner.split=test runner.vis_result=True detector.step=1  #步长为1