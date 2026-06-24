import logging
import os

from pytorch_lightning.cli import LightningCLI

from cafa.core.common.logging import config_logging_json

logger = logging.getLogger(__name__)


class CLI(LightningCLI):
    def get_trainer_config(self):
        subcommand = self.config.get('subcommand')
        trainer_config = (
            self.config[subcommand]['trainer']
            if subcommand
            else self.config['trainer']
        )
        return trainer_config

    def set_data_checkpoint_path(self) -> None:
        subcommand = self.config.get('subcommand')
        if subcommand is None:
            return
        ckpt_path = self.config[subcommand].get('ckpt_path')
        datamodule_config = self.config[subcommand].get('data')
        if (
            datamodule_config is None
            or 'checkpoint_path' not in datamodule_config['init_args']
        ):
            return
        datamodule_config['init_args']['checkpoint_path'] = ckpt_path

    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        trainer_config = self.get_trainer_config()
        os.makedirs(trainer_config['default_root_dir'], exist_ok=False)
        config_logging_json(logs_dir=trainer_config['default_root_dir'])

        self.set_data_checkpoint_path()

        tb_logger_config = next(
            (
                logger
                for logger in trainer_config.logger
                if logger.class_path
                == 'pytorch_lightning.loggers.TensorBoardLogger'
            ),
            None,
        )
        tb_logger_config['init_args.save_dir'] = trainer_config[
            'default_root_dir'
        ]

    def _parse_ckpt_path(self) -> None:
        logger.info(
            'hyper_parameters in checkpoints are ignored to avoid overriding command line arguments'
        )
 