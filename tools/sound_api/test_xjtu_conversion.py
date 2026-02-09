# -*- coding: utf-8 -*-
"""
XJTU MC转换流程测试脚本
快速测试单个文件的转换流程
"""
import os
import sys

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from convert_mc_to_wav import load_binary_signal, convert_to_wav
from convert_sound_api import (
    get_default_config,
    test_sound_api,
    parse_api_response
)


def test_single_file_conversion():
    """测试单个文件的完整转换流程"""
    print("=" * 60)
    print("XJTU MC转换流程测试")
    print("=" * 60)
    
    # 测试文件路径
    mc_dir = r'D:\guzhangjiance\datasets\output_xjtu_mc\xjtu'
    test_base = 'XJTU-SY_1_0'
    
    binary_file = os.path.join(mc_dir, f'{test_base}.f')
    json_file = os.path.join(mc_dir, f'{test_base}.json')
    
    # 检查文件是否存在
    if not os.path.exists(binary_file):
        print(f"错误: 测试文件不存在 {binary_file}")
        print(f"请确认MC文件目录: {mc_dir}")
        return False
    
    if not os.path.exists(json_file):
        print(f"错误: 元数据文件不存在 {json_file}")
        return False
    
    print(f"\n[步骤1] 加载MC文件")
    print(f"  二进制文件: {binary_file}")
    print(f"  元数据文件: {json_file}")
    
    # 加载MC文件
    data, metadata = load_binary_signal(binary_file, json_file)
    
    if data is None:
        print("  ❌ 加载失败")
        return False
    
    print(f"  [OK] 加载成功")
    print(f"  数据形状: {data.shape}")
    print(f"  数据类型: {data.dtype}")
    print(f"  采样率: {metadata['sampling_rate']} Hz")
    print(f"  工况: {metadata['working_condition']}")
    print(f"  轴承: {metadata['bearing_name']}")
    print(f"  健康标签: {metadata['health_label']}")
    
    # 转换为WAV
    print(f"\n[步骤2] 转换为WAV音频")
    
    temp_dir = r'D:\guzhangjiance\datasets\output_xjtu_mc\test_temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    wav_file = os.path.join(temp_dir, f'{test_base}.wav')
    
    success = convert_to_wav(
        data=data,
        output_file=wav_file,
        sampling_rate=metadata['sampling_rate'],
        channel_mode='horizontal',
        normalize_method='minmax'
    )
    
    if not success:
        print("  [FAIL] 转换失败")
        return False
    
    print(f"  [OK] 转换成功")
    print(f"  输出文件: {wav_file}")
    print(f"  文件大小: {os.path.getsize(wav_file) / 1024:.2f} KB")
    
    # 通过API转换
    print(f"\n[步骤3] 调用API转换为能量密度曲线")
    
    # 获取API配置
    api_url, headers, form_data_params, file_param_name = get_default_config()
    print(f"  API URL: {api_url}")
    
    # 调用API
    result = test_sound_api(
        wav_file,
        api_url,
        headers=headers,
        file_param_name=file_param_name,
        form_data_params=form_data_params,
        timeout=60
    )
    
    if result is None:
        print("  [FAIL] API调用失败")
        print("\n提示:")
        print("  - 检查API服务是否运行")
        print("  - 检查网络连接")
        print("  - 检查API配置")
        return False
    
    print(f"  [OK] API调用成功")
    
    # 解析响应
    print(f"\n[步骤4] 解析API响应")
    
    data_result = parse_api_response(result, verbose=True)
    
    if data_result is None:
        print("  [FAIL] 解析失败")
        return False
    
    print(f"  [OK] 解析成功")
    
    # 清理临时文件
    print(f"\n[清理] 删除临时WAV文件")
    try:
        os.remove(wav_file)
        print(f"  [OK] 已删除: {wav_file}")
    except:
        print(f"  警告: 无法删除临时文件")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 测试完成！所有步骤都成功")
    print("=" * 60)
    print("\n可以开始批量转换了！")
    print("运行命令: python tools/batch_convert_xjtu_with_api.py")
    
    return True


if __name__ == '__main__':
    try:
        success = test_single_file_conversion()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
