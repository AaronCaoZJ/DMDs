#!/bin/bash

# 查看训练数据集中的提示词示例
# 使用方法：bash scripts/view_laion_prompts.sh

CHECKPOINT_PATH=/data/sdv15

echo "=========================================="
echo "查看 LAION 训练数据集提示词"
echo "=========================================="
echo ""
echo "数据集: LAION (500k images, aesthetic score >= 6.25)"
echo "文件: $CHECKPOINT_PATH/captions_laion_score6.25.pkl"
echo ""

# 检查文件是否存在
if [ ! -f "$CHECKPOINT_PATH/captions_laion_score6.25.pkl" ]; then
    echo "错误: 提示词文件不存在！"
    echo "请先运行下载脚本:"
    echo "  bash scripts/download_sdv15.sh $CHECKPOINT_PATH"
    exit 1
fi

# 显示前20个提示词
python /home/zhijun/Code/DMDs/main/view_prompts.py \
    --pkl_path $CHECKPOINT_PATH/captions_laion_score6.25.pkl \
    --num_samples 20

echo ""
echo "=========================================="
echo "如果要查看随机样本，使用："
echo "  python /home/zhijun/Code/DMDs/main/view_prompts.py --pkl_path $CHECKPOINT_PATH/captions_laion_score6.25.pkl --num_samples 50 --random"
echo "=========================================="
