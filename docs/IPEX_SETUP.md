# IPEX 环境配置（Intel 核显 / XPU）

使用 **Python 3.10** 的独立 conda 环境。**Windows 上 PyPI 常无 IPEX 的 wheel**，因此环境分两步：先创建环境（不含 IPEX），再单独安装 IPEX。

## 一、创建环境（必做）

在项目根目录执行（**不会安装 IPEX**，避免 “No matching distribution found”）：

```bash
conda env create -f environment_ipex.yml
```

或双击运行 **`setup_ipex_env.bat`**（会创建环境并自动尝试安装 IPEX）。

激活环境：

```bash
conda activate ipex
```

## 二、安装 IPEX（创建环境后执行）

在 **已激活 ipex** 的环境下，任选一种方式安装：

**方式 1：Intel 官方 wheel 源（优先尝试）**

```bash
conda activate ipex
pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

**方式 2：仅 PyPI（若方式 1 失败再试）**

```bash
pip install intel-extension-for-pytorch
```

**说明**：安装 IPEX 后若报 “needs to work with PyTorch 2.8.*, but PyTorch 2.10.x is found”，请在 ipex 环境中执行：`pip install "torch>=2.8,<2.9" torchvision torchaudio`，再重新验证。Windows 下若始终无 XPU wheel，可改用 **DirectML**（`--device dml`），见 `docs/NLP_XPU_SETUP.md`。

## 三、验证与运行

**检查 XPU 是否可用：**

```bash
conda activate ipex
python -c "import torch; import intel_extension_for_pytorch as ipex; print('XPU:', torch.xpu.is_available())"
```

**用核显做测试集验证：**

```bash
conda activate ipex
python experiments/train.py --data_source sound_api_cache --task hi --cache_dir datasets/sound_api/cache_npz_quick --eval_only --device xpu
```

成功时终端会显示：`使用设备: xpu`。

---

## 五、Windows 上报 “esimd_kernels.dll 找不到” (WinError 126)

IPEX 的 XPU 在 Windows 上依赖 **Intel oneAPI 运行库**，若未安装会报：

```text
OSError: [WinError 126] 找不到指定的模块。 Error loading "...\esimd_kernels.dll" or one of its dependencies.
```

**处理方式（二选一）：**

1. **安装 Intel oneAPI Base Toolkit（推荐，用核显时）**  
   - 打开：https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html  
   - 选 Windows、在线安装器，安装时至少勾选 **Intel oneAPI DPC++/C++ Compiler** 与 **Intel oneAPI runtime**。  
   - 安装完成后**重启终端**，再执行：`conda activate ipex`，然后重新跑 `python -c "import intel_extension_for_pytorch ..."` 或训练脚本。

2. **不装 oneAPI，改用 CPU 或 DirectML**  
   - 用 CPU：`python experiments/train.py ... --eval_only --device cpu`  
   - 用核显（DirectML）：先 `pip install torch-directml`，再 `... --device dml`（若模型能跑在 dml 上，见 `docs/NLP_XPU_SETUP.md`）。

## 四、脚本方式创建（Windows）

双击或在 cmd 中执行：

```
setup_ipex_env.bat
```

会创建 `ipex` 环境；若 IPEX 未装上，按脚本末尾提示在激活环境中再执行一次 `pip install intel-extension-for-pytorch` 或带 `--extra-index-url` 的命令。
