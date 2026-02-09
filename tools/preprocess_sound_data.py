#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理声音数据为NPZ格式
将所有sheet中的数据保存为更高效的二进制格式
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def preprocess_sound_data(sound_dir='声音能量曲线数据', output_dir='sound_data_processed'):
    """
    将所有sheet数据预处理为NPZ格式，加速加载
    
    Args:
        sound_dir: 声音数据目录
        output_dir: 输出目录
    """
    sound_path = Path(sound_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"预处理声音数据...")
    print(f"输入目录: {sound_path}")
    print(f"输出目录: {output_path}")
    
    data_dict = {}
    failed_count = 0
    
    xlsx_files = list(sound_path.glob('*.xlsx'))
    
    for xlsx_file in tqdm(xlsx_files, desc="处理xlsx文件"):
        try:
            xls = pd.ExcelFile(xlsx_file)
            
            for sheet_name in xls.sheet_names:
                base_name = sheet_name.replace('.wav', '')
                
                try:
                    df = pd.read_excel(xlsx_file, sheet_name=sheet_name, header=None, skiprows=2)
                    
                    # 保存为numpy数组（float32以节省空间）
                    data_dict[base_name] = np.array([
                        df.iloc[:, 0].values.astype(np.float32),  # frequency
                        df.iloc[:, 1].values.astype(np.float32),  # volume
                        df.iloc[:, 2].values.astype(np.float32)   # density
                    ], dtype=object)  # 使用object以支持不同长度
                    
                except Exception as e:
                    print(f"  Error processing {base_name}: {e}")
                    failed_count += 1
        
        except Exception as e:
            print(f"  Error reading {xlsx_file}: {e}")
            failed_count += 1
    
    # 保存为npz格式（压缩）
    output_file = output_path / 'sound_curves.npz'
    np.savez_compressed(output_file, **data_dict)
    
    print(f"\n预处理完成!")
    print(f"成功处理: {len(data_dict)} 个样本")
    print(f"失败: {failed_count} 个样本")
    print(f"保存到: {output_file}")
    print(f"文件大小: {output_file.stat().st_size / (1024**2):.2f} MB")
    
    return output_file

if __name__ == '__main__':
    preprocess_sound_data()
