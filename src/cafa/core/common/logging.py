import json
import logging
import logging.config
import os
import traceback


def config_logging(logs_dir: str = None):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            fmt='%(asctime)s %(name)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )
    )

    handlers = [stream_handler]
    if logs_dir is not None:
        file_handler = logging.FileHandler(os.path.join(logs_dir, 'output.log'))
        file_handler.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s %(name)s  %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            )
        )
        handlers.append(file_handler)
    logging.basicConfig(level=logging.INFO, handlers=handlers)


def config_logging_json(config_path: str = None, logs_dir: str = None):
    config_path = config_path or os.environ.get('LOGGING_CONFIG')
    config_path = config_path or os.path.join(
        os.path.dirname(__file__), 'logging.json'
    )

    logs_dir = logs_dir or os.environ.get('LOGS_DIR')
    print(f'logging config path={config_path} logs_dir={logs_dir}')
    with open(config_path) as f:
        config_dict = json.load(f)
    file_handler_dict = config_dict['handlers'].get('file_handler')
    if file_handler_dict and logs_dir:
        file_handler_dict['filename'] = os.path.join(
            logs_dir, file_handler_dict['filename']
        )

    logging.config.dictConfig(config_dict)


def log_uncaught_exceptions(ex_cls, ex, tb):
    logging.critical(''.join(traceback.format_tb(tb)))
    logging.critical('{0}: {1}'.format(ex_cls, ex))
