import logging

# ============================================= 日志相关 ========================================================
def setup_logging(log_file=None, log_level=logging.INFO):
    # 创建一个日志记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 创建一个格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建一个控制台处理器并设置格式
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 如果指定了日志文件，创建一个文件处理器并设置格式
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)