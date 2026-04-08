import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.data_processing.process_gse164191 import process_gse164191
from scripts.data_processing.process_gse213516 import process_gse213516
from scripts.data_processing.process_gtex import process_gtex
from scripts.data_processing.merge_gse231409 import main as merge_gse231409
from scripts.data_processing.preprocess_and_merge import merge_datasets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_preprocessing():
    """按顺序执行所有数据预处理步骤"""
    logger.info("=== 开始数据预处理流程 ===")

    datasets_to_process = [
        ("GSE164191", process_gse164191),
        ("GSE213516", process_gse213516),
        ("GTEx", process_gtex),
    ]

    for dataset_name, process_func in datasets_to_process:
        logger.info(f"\n--- 处理 {dataset_name} 数据集 ---")
        try:
            result = process_func()
            if result is not None:
                logger.info(f"{dataset_name} 数据集处理成功")
            else:
                logger.warning(f"{dataset_name} 数据集处理失败，但继续执行")
        except Exception as e:
            logger.error(f"{dataset_name} 数据集处理出错: {e}")
            logger.warning("继续执行后续步骤")

    logger.info("\n--- 合并GSE231409数据集 ---")
    try:
        merge_gse231409()
        logger.info("GSE231409数据集合并成功")
    except Exception as e:
        logger.error(f"GSE231409数据集合并失败: {e}")
        logger.warning("继续执行后续步骤")

    logger.info("\n--- 合并所有数据集 ---")
    try:
        merged_df = merge_datasets()
        if merged_df is not None:
            logger.info("所有数据集合并成功")
            logger.info(f"最终数据集形状: {merged_df.shape}")
        else:
            logger.error("数据集合并失败")
    except Exception as e:
        logger.error(f"数据集合并出错: {e}")

    logger.info("\n=== 数据预处理流程完成 ===")


if __name__ == "__main__":
    run_preprocessing()