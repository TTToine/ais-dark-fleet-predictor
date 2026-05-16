import logging, sys, json
from pathlib import Path
from logging.handlers import RotatingFileHandler
from .exceptions import AISPipelineError

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        if getattr(record, 'json_output', False):
            return json.dumps({"ts": self.formatTime(record), "lvl": record.levelname, "msg": record.getMessage(), "ctx": getattr(record, 'context', {})})
        return super().format(record)

def setup_logging(config: dict, log_dir: str = None, json_logs: bool = False) -> logging.Logger:
    logger = logging.getLogger(config.get('project', {}).get('name', 'ais-predictor'))
    logger.setLevel(getattr(logging, config.get('logging', {}).get('level', 'INFO')))
    if logger.hasHandlers(): logger.handlers.clear()
    fmt = StructuredFormatter(datefmt='%Y-%m-%dT%H:%M:%S') if json_logs else logging.Formatter(config.get('logging', {}).get('format', '%(asctime)s - %(levelname)s - %(message)s'))
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    if config.get('logging', {}).get('file'):
        log_path = Path(log_dir or '.') / config['logging']['file']
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_path, maxBytes=100*1024*1024, backupCount=5)
        fh.setFormatter(fmt); fh.setLevel(logging.DEBUG); logger.addHandler(fh)
    return logger
