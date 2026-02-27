#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法示例:
    python tools/analyze_result_cross.py --model_path "D:\guzhangjiance\checkpoints\risk_index_small_0212" --data_path "D:\guzhangjiance\datasets\cwru\cwru_processed"
    （模型目录下必须有backbone和risk两个模型文件）
    # python tools/analyze_result_cross.py --model_path model.pth --data_path val.csv --x_keys feature1 feature2 --y_keys label
    # python tools/analyze_result_cross.py --model_path model.pth --data_path ./val_dir/
依赖:
    pip install torch numpy pandas openpyxl scikit-learn
"""

import argparse
import torch
import os
import sys
import importlib.util

def _match_keycase(keys, preferreds):
    # 支持忽略大小写的键匹配
    for want in (preferreds or []):
        for k in keys:
            if want.lower() == str(k).lower():
                return k
    return None

def load_data(data_path, x_key_options=None, y_key_options=None):
    if not x_key_options:
        x_key_options = ['X', 'features', 'feature', 'data', 'input', 'inputs']
    if not y_key_options:
        y_key_options = ['y', 'label', 'labels', 'target', 'targets', 'output']

    ext = os.path.splitext(data_path)[-1].lower()
    if ext in {'.npz'}:
        import numpy as np
        data = np.load(data_path, allow_pickle=True)
        keys = list(data.keys())
        # 做键名匹配
        x_key = _match_keycase(keys, x_key_options)
        y_key = _match_keycase(keys, y_key_options)
        X = data[x_key] if x_key is not None else None
        y = data[y_key] if y_key is not None else None
        if X is None or y is None:
            # 尝试fallback成0/1序号
            if 0 in keys and 1 in keys:
                X = data[0]
                y = data[1]
        if X is None or y is None:
            raise ValueError(f"npz文件格式识别失败，包含键: {keys}")
        y = y.reshape(-1)
        return X, y
    elif ext in {'.xlsx', '.xls'}:
        import pandas as pd
        df = pd.read_excel(data_path)
        label_col = _match_keycase(df.columns, y_key_options) or 'label'
        feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].values
        y = df[label_col].values
        return X, y
    elif ext in {'.csv'}:
        import pandas as pd
        df = pd.read_csv(data_path)
        label_col = _match_keycase(df.columns, y_key_options) or 'label'
        feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].values
        y = df[label_col].values
        return X, y
    elif ext == '.json':
        import json
        with open(data_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        keys = list(js.keys())
        x_key = _match_keycase(keys, x_key_options)
        y_key = _match_keycase(keys, y_key_options)
        X = js[x_key] if x_key is not None else None
        y = js[y_key] if y_key is not None else None
        if X is None or y is None:
            raise ValueError(f"json文件无有效X/y，keys: {keys}")
        return X, y
    elif os.path.isdir(data_path):
        import numpy as np
        # CWRU 风格：目录内 signals.npy + labels.npy
        signals_path = os.path.join(data_path, "signals.npy")
        labels_path = os.path.join(data_path, "labels.npy")
        if os.path.isfile(signals_path) and os.path.isfile(labels_path):
            X = np.load(signals_path)
            y = np.load(labels_path).reshape(-1)
            if X.ndim == 2:
                # (N, L) -> (N, 2, L)，backbone 需要 2 通道
                X = np.stack([X, X], axis=1)
            elif X.ndim == 3 and X.shape[1] == 1:
                X = np.concatenate([X, X], axis=1)
            return X, y
        files = [os.path.join(data_path, f) for f in sorted(os.listdir(data_path))]
        files = [f for f in files if not os.path.isdir(f) and not os.path.basename(f).startswith('.')]
        ext_sample = os.path.splitext(files[0])[-1].lower() if files else ''
        if ext_sample == '.npz':
            all_X, all_y = [], []
            for f in files:
                try:
                    Xi, yi = load_data(f, x_key_options, y_key_options)
                    all_X.append(Xi)
                    all_y.append(yi)
                except Exception as e:
                    print(f"读取{f}失败: {e}")
            if all_X:
                X = np.concatenate(all_X, axis=0) if hasattr(all_X[0], 'shape') else all_X
                y = np.concatenate(all_y, axis=0) if hasattr(all_y[0], 'shape') else all_y
                return X, y
            else:
                return [], []
        elif ext_sample in {'.csv', '.xlsx', '.xls'}:
            all_X, all_y = [], []
            for f in files:
                try:
                    Xi, yi = load_data(f, x_key_options, y_key_options)
                    all_X.append(Xi)
                    all_y.append(yi)
                except Exception as e:
                    print(f"读取{f}失败: {e}")
            if all_X:
                X = np.concatenate(all_X, axis=0) if hasattr(all_X[0], 'shape') else all_X
                y = np.concatenate(all_y, axis=0) if hasattr(all_y[0], 'shape') else all_y
                return X, y
            else:
                return [], []
        else:
            print(f"目录型数据不支持该格式（当前首扩展名: {ext_sample}）。支持: signals.npy+labels.npy、批量.npz、.csv、.xlsx。")
            return [], []
    else:
        print("不支持的数据格式：", ext)
        return [], []

def load_model_classes_from_dl():
    # 查找dl/model.py文件，引入模型结构
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_py = os.path.join(root_dir, "dl", "model.py")
    if not os.path.isfile(model_py):
        print(f"未找到模型结构定义文件: {model_py}")
        return None
    spec = importlib.util.spec_from_file_location("dl_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 查找Backbone类由此文件提供, RiskHead类由experiments/train.py提供（见下）
    backbone_cls = None
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if callable(obj):
            if ('backbone' in attr.lower() or 'feature' in attr.lower()):
                backbone_cls = obj
    # 加载risk head类
    risk_cls = None
    exp_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_py = os.path.join(exp_root, "experiments", "train.py")
    if os.path.isfile(train_py):
        train_spec = importlib.util.spec_from_file_location("exp_train", train_py)
        train_mod = importlib.util.module_from_spec(train_spec)
        train_spec.loader.exec_module(train_mod)
        # 风险头类名按照原有策略查找
        for attr in dir(train_mod):
            obj = getattr(train_mod, attr)
            if callable(obj) and ('risk' in attr.lower() or 'head' in attr.lower() or 'clshead' in attr.lower()):
                risk_cls = obj
                break
    if backbone_cls is None or risk_cls is None:
        print("未能从dl/model.py和experiments/train.py同时找到Backbone和RiskHead类。")
    return backbone_cls, risk_cls

def main():
    parser = argparse.ArgumentParser(
        description=(
            "使用指定模型文件在指定数据集上验证性能。\n"
            "支持的数据格式: .npz, .csv, .xlsx, .xls, .json, 目录(批量npz/csv/xlsx)。\n"
            "示例: python analyze_result_cross.py --model_path best_model.pt --data_path valid.npz\n"
            "依赖: pip install torch numpy pandas openpyxl scikit-learn"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model_path", type=str, required=True, help="PyTorch模型目录（需包含backbone和risk两个模型文件）")
    parser.add_argument("--data_path", type=str, required=True, help="验证集数据文件路径，可为npz/excel/json/csv/目录等")
    parser.add_argument("--x_keys", type=str, nargs='*', default=None, help="特征X的键（npz/json时可指定，如 features data X 等，支持多个）")
    parser.add_argument("--y_keys", type=str, nargs='*', default=None, help="标签y的键（npz/json时可指定，如 label y target 等，支持多个）")
    args = parser.parse_args()

    # 支持模型目录内有backbone和risk模型文件
    if os.path.isdir(args.model_path):
        candidate_files = [f for f in os.listdir(args.model_path) if not os.path.isdir(os.path.join(args.model_path, f))]
        backbone_file = None
        risk_file = None
        for f in candidate_files:
            lowerf = f.lower()
            if 'backbone' in lowerf and f.endswith(('.pt', '.pth', '.bin')):
                backbone_file = os.path.join(args.model_path, f)
            if 'risk' in lowerf and f.endswith(('.pt', '.pth', '.bin')):
                risk_file = os.path.join(args.model_path, f)
        if not backbone_file or not risk_file:
            print(f"目标文件夹必须包含backbone和risk两个模型文件, 当前: {[f for f in candidate_files]}")
            sys.exit(1)
        model_files = {'backbone': backbone_file, 'risk': risk_file}
    else:
        print(f"model_path建议直接传目录，目录下需有backbone和risk模型文件！")
        sys.exit(1)

    for k in model_files:
        if not os.path.exists(model_files[k]):
            print(f"指定模型文件不存在: {model_files[k]}")
            sys.exit(1)
    if not (os.path.exists(args.data_path) or os.path.isdir(args.data_path)):
        print(f"指定数据文件/目录不存在: {args.data_path}")
        sys.exit(1)

    # 加载模型权重（优先 weights_only=True；旧 checkpoint 可能为完整模型则回退）
    try:
        try:
            backbone_obj = torch.load(model_files['backbone'], map_location='cpu', weights_only=True)
            risk_obj = torch.load(model_files['risk'], map_location='cpu', weights_only=True)
        except (TypeError, ValueError):
            backbone_obj = torch.load(model_files['backbone'], map_location='cpu', weights_only=False)
            risk_obj = torch.load(model_files['risk'], map_location='cpu', weights_only=False)
        need_rebuild = (
            (isinstance(backbone_obj, dict) and any("." in k for k in backbone_obj))
            or (isinstance(risk_obj, dict) and any("." in k for k in risk_obj))
        )
        if need_rebuild:
            backbone_cls, risk_cls = load_model_classes_from_dl()
            if not backbone_cls or not risk_cls:
                print("无法从dl/model.py或experiments/train.py找到合适的模型结构类，无法重构模型。")
                sys.exit(1)
            backbone = backbone_cls()
            risk_model = risk_cls()
            if isinstance(backbone_obj, dict) and any("." in k for k in backbone_obj):
                backbone.load_state_dict(backbone_obj)
            else:
                print("backbone文件不是标准state_dict, 无法加载。")
                sys.exit(1)
            if isinstance(risk_obj, dict) and any("." in k for k in risk_obj):
                risk_model.load_state_dict(risk_obj)
            else:
                print("risk文件不是标准state_dict, 无法加载。")
                sys.exit(1)
        else:
            backbone = backbone_obj
            risk_model = risk_obj
        backbone.eval()
        risk_model.eval()
    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)

    try:
        X, y = load_data(args.data_path, x_key_options=args.x_keys, y_key_options=args.y_keys)
    except Exception as e:
        print(f"加载数据失败: {e}")
        sys.exit(1)

    if X is None or y is None or len(X) == 0 or len(y) == 0:
        print("未能加载有效数据，请检查格式与内容。")
        sys.exit(1)
    if len(X) != len(y):
        print(f"特征和标签数量不一致: {len(X)} vs {len(y)}")
        sys.exit(1)

    correct = 0
    total = 0
    y_true = []
    y_pred = []
    import numpy as np
    X_t = np.asarray(X)
    y_t = np.asarray(y)
    for i in range(len(X_t)):
        xi = X_t[i]
        yi = y_t[i]
        input_tensor = torch.tensor(xi).float().unsqueeze(0)
        with torch.no_grad():
            # 先通过backbone提取特征，再通过risk进行分类
            feature = backbone(input_tensor)
            output = risk_model(feature)
            if hasattr(output, 'ndim') and output.ndim == 2 and output.size(1) > 1:
                _, predicted = torch.max(output, 1)
                pred_val = predicted.item() if hasattr(predicted, "item") else int(predicted[0])
            else:
                output_val = output if hasattr(output, "detach") else torch.tensor(output)
                pred_val = (output_val > 0.5).long()
                pred_val = pred_val.item() if hasattr(pred_val, "item") else int(pred_val)
            correct += int(pred_val == int(yi))
            total += 1
            y_true.append(int(yi))
            y_pred.append(int(pred_val))

    print(f"Accuracy: {correct / total:.4f}" if total > 0 else "样本数量为0，无法计算准确率。")
    # 增加更多评估指标
    if total > 0:
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, digits=4))
            print("Confusion Matrix:")
            print(confusion_matrix(y_true, y_pred))
        except Exception as e:
            print(f"无法计算详细分类指标: {e}")

if __name__ == "__main__":
    main()