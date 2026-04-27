# Live2Diff
Live2Diff：基于点控制视频生成模型和缓存图片策略，模拟live2D效果的模型。

## 安装步骤
```bash
conda create -n live2diff python==3.10.11 
conda activate live2diff
git clone https://github.com/Lvshu6/Live2Diff.git
cd Live2Diff
pip install -e .
cd co-tracker
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
