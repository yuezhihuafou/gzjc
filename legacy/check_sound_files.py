import pandas as pd
import os

files = sorted([f for f in os.listdir('声音能量曲线数据') if f.endswith('.xlsx')])
print('声音文件夹中的11个xlsx文件:')
print('=' * 70)

for i, f in enumerate(files, 1):
    try:
        df = pd.read_excel(f'声音能量曲线数据/{f}', header=None, nrows=1)
        wav_name = df.iloc[0, 0]
        mat_name = wav_name.replace('.wav', '.mat')
        print(f'{i:2d}. {f:30s} -> {wav_name:25s} ({mat_name})')
    except Exception as e:
        print(f'{i:2d}. {f:30s} -> (读取失败: {e})')

print('\n' + '=' * 70)
print(f'总计: {len(files)} 个声音能量曲线文件')
print('\n这就是当前所有已转换的声音数据。')
print('如果需要更多覆盖，需要为其他CWRU样本生成对应的xlsx文件。')
