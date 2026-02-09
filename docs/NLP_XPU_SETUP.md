# nlp 环境 + 核显加速（Windows 推荐 DirectML / 可选 Intel XPU）

在 **conda 环境 nlp** 下用核显做测试集验证。**Windows 强烈推荐用 DirectML（torch-directml）**，无需 Intel 专用包。

---

## 方案一：DirectML（推荐，Windows 核显）

适用于 **Windows 10/11 + Intel 核显（如 U7 155H）**，pip 可直接安装，无 “No matching distribution” 问题。

### 安装

```bash
conda activate nlp
pip install -r requirements-nlp-dml.txt
```

或直接：`pip install torch-directml`

### 运行测试集验证（用核显）

```bash
conda activate nlp
python experiments/train.py --data_source sound_api_cache --task hi --cache_dir datasets/sound_api/cache_npz_quick --eval_only --device dml
```

终端会显示使用 DirectML 设备（核显加速）。

---

## 方案二：Intel XPU（intel-extension-for-pytorch）

若你不在 Windows 或希望用 Intel 官方扩展，再考虑本方案。

### 安装（在 nlp 中）

```bash
conda activate nlp
pip install -r requirements-nlp-xpu.txt
```

若出现 **`No matching distribution found for intel-extension-for-pytorch`**：  
多为当前 **Python 版本** 在 Windows 下无对应 wheel（常见于 3.12/3.13）。可新建 Python 3.10/3.11 环境再装，或改用上方 **方案一 DirectML**。

若 nlp 中还没有 PyTorch：

```bash
conda activate nlp
pip install torch torchvision torchaudio
pip install intel-extension-for-pytorch
```

## 设备选择小结

| 设备 | 命令 | 说明 |
|------|------|------|
| **核显（Windows）** | `--device dml` | 安装 `torch-directml`，推荐 |
| Intel XPU | `--device xpu` | 需 `intel-extension-for-pytorch`，Windows 常无 wheel |
| NVIDIA GPU | `--device cuda` | 有独显时使用 |
| CPU | `--device cpu` | 不装任何加速包时使用 |

不指定 `--device` 时，脚本会按 **cuda → dml → xpu → cpu** 自动选择可用设备。
