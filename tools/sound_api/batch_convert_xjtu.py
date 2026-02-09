# -*- coding: utf-8 -*-
"""
XJTU数据集批量API转换完整流程脚本
整合MC文件转WAV和API转换两个步骤
"""
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 导入MC转WAV的函数
from convert_mc_to_wav import batch_convert_mc_to_wav

# 导入API转换的函数
from convert_sound_api import (
    get_default_config,
    test_sound_api,
    parse_api_response,
    save_to_xlsx_format
)


def convert_xjtu_mc_to_energy_density(
    mc_input_dir,
    wav_output_dir,
    api_output_dir,
    channel_mode='horizontal',
    normalize_method='minmax',
    api_config=None,
    save_json=True,
    save_xlsx=True,
    cleanup_wav=False
):
    """
    完整的XJTU MC双通道数据转能量密度曲线流程
    
    流程:
    1. MC文件(.f) -> WAV音频文件
    2. WAV文件 -> API处理 -> 能量密度曲线(JSON/XLSX)
    3. (可选) 删除中间WAV文件
    
    Args:
        mc_input_dir: MC文件输入目录
        wav_output_dir: WAV文件临时输出目录
        api_output_dir: API处理结果输出目录
        channel_mode: 通道模式
        normalize_method: 归一化方法
        api_config: API配置元组 (api_url, headers, form_data_params, file_param_name)
                   如果为None，使用默认配置
        save_json: 是否保存JSON格式
        save_xlsx: 是否保存XLSX格式
        cleanup_wav: 是否在API处理完成后删除WAV文件
    
    Returns:
        dict: 转换结果统计
    """
    results = {
        'mc_to_wav': {'success': 0, 'failed': 0},
        'wav_to_api': {'success': 0, 'failed': 0},
        'total_files': 0,
        'files': []
    }
    
    print("\n" + "=" * 60)
    print("XJTU MC数据 -> 能量密度曲线 完整转换流程")
    print("=" * 60)
    
    # ========================================================================
    # 步骤1: MC文件转WAV
    # ========================================================================
    print("\n[步骤 1/2] MC文件 -> WAV音频文件")
    print("-" * 60)
    
    mc_results = batch_convert_mc_to_wav(
        input_dir=mc_input_dir,
        output_dir=wav_output_dir,
        channel_mode=channel_mode,
        normalize_method=normalize_method
    )
    
    results['mc_to_wav']['success'] = mc_results['success']
    results['mc_to_wav']['failed'] = mc_results['failed']
    results['total_files'] = mc_results['success']
    
    if mc_results['success'] == 0:
        print("\n错误: 没有成功转换的WAV文件，流程终止")
        return results
    
    # ========================================================================
    # 步骤2: WAV文件通过API转换为能量密度曲线
    # ========================================================================
    print("\n[步骤 2/2] WAV文件 -> API处理 -> 能量密度曲线")
    print("-" * 60)
    
    # 获取API配置
    if api_config is None:
        api_url, headers, form_data_params, file_param_name = get_default_config()
    else:
        api_url, headers, form_data_params, file_param_name = api_config
    
    print(f"API URL: {api_url}")
    print(f"输出目录: {api_output_dir}\n")
    
    # 确保输出目录存在
    os.makedirs(api_output_dir, exist_ok=True)
    
    # 获取所有成功转换的WAV文件
    wav_files = []
    for item in mc_results['files']:
        if item['status'] == 'success' and 'output' in item:
            wav_files.append({
                'wav_path': item['output'],
                'mc_file': item['file'],
                'metadata': item.get('metadata', {})
            })
    
    # 批量处理WAV文件
    for wav_info in tqdm(wav_files, desc="API转换中"):
        wav_file = wav_info['wav_path']
        filename_base = os.path.splitext(os.path.basename(wav_file))[0]
        
        # 调用API
        result = test_sound_api(
            wav_file, 
            api_url, 
            headers=headers,
            file_param_name=file_param_name,
            form_data_params=form_data_params
        )
        
        file_result = {
            'mc_file': wav_info['mc_file'],
            'wav_file': wav_file,
            'metadata': wav_info['metadata']
        }
        
        if result:
            # 解析响应
            data = parse_api_response(result, verbose=False)
            
            if data:
                # 保存JSON格式
                if save_json:
                    json_file = os.path.join(api_output_dir, f"{filename_base}.json")
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'frequency': data['frequency'].tolist(),
                            'volume': data['volume'].tolist(),
                            'density': data['density'].tolist(),
                            'source_metadata': wav_info['metadata']
                        }, f, indent=2, ensure_ascii=False)
                    file_result['json_output'] = json_file
                
                # 保存XLSX格式
                if save_xlsx:
                    xlsx_file = os.path.join(api_output_dir, f"{filename_base}.xlsx")
                    save_to_xlsx_format(data, xlsx_file, filename_base)
                    file_result['xlsx_output'] = xlsx_file
                
                results['wav_to_api']['success'] += 1
                file_result['status'] = 'success'
                file_result['data_points'] = len(data['frequency'])
            else:
                results['wav_to_api']['failed'] += 1
                file_result['status'] = 'failed'
                file_result['reason'] = 'parse_error'
        else:
            results['wav_to_api']['failed'] += 1
            file_result['status'] = 'failed'
            file_result['reason'] = 'api_error'
        
        results['files'].append(file_result)
    
    # ========================================================================
    # 步骤3: (可选) 清理临时WAV文件
    # ========================================================================
    if cleanup_wav:
        print("\n[清理] 删除临时WAV文件...")
        cleanup_count = 0
        for wav_info in wav_files:
            try:
                if os.path.exists(wav_info['wav_path']):
                    os.remove(wav_info['wav_path'])
                    cleanup_count += 1
            except Exception as e:
                print(f"警告: 无法删除 {wav_info['wav_path']}: {e}")
        print(f"已删除 {cleanup_count} 个WAV文件")
        
        # 如果WAV目录为空，也删除它
        try:
            if not os.listdir(wav_output_dir):
                os.rmdir(wav_output_dir)
                print(f"已删除空目录: {wav_output_dir}")
        except:
            pass
    
    # ========================================================================
    # 生成最终报告
    # ========================================================================
    print("\n" + "=" * 60)
    print("转换完成汇总")
    print("=" * 60)
    print(f"步骤1 - MC转WAV: 成功 {results['mc_to_wav']['success']}, "
          f"失败 {results['mc_to_wav']['failed']}")
    print(f"步骤2 - API转换: 成功 {results['wav_to_api']['success']}, "
          f"失败 {results['wav_to_api']['failed']}")
    print(f"总计: {results['total_files']} 个MC文件")
    
    # 保存完整报告
    report_file = os.path.join(api_output_dir, 'complete_conversion_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n完整报告已保存: {report_file}")
    
    return results


def main():
    """主函数"""
    print("=" * 70)
    print("XJTU MC双通道数据 -> 能量密度曲线 批量转换工具")
    print("=" * 70)
    
    # 默认配置
    default_mc_dir = r'D:\guzhangjiance\datasets\output_xjtu_mc\xjtu'
    default_wav_dir = r'D:\guzhangjiance\datasets\output_xjtu_mc\wav_temp'
    default_api_dir = r'D:\guzhangjiance\datasets\output_xjtu_mc\api_output'
    
    print("\n默认路径:")
    print(f"  MC输入目录: {default_mc_dir}")
    print(f"  WAV临时目录: {default_wav_dir}")
    print(f"  API输出目录: {default_api_dir}")
    
    # 选择输入目录
    mc_dir = input("\n请输入MC文件目录（回车使用默认）: ").strip().strip('"').strip("'")
    if not mc_dir:
        mc_dir = default_mc_dir
    
    if not os.path.exists(mc_dir):
        print(f"错误: MC文件目录不存在 {mc_dir}")
        return
    
    # 选择WAV临时目录
    wav_dir = input("请输入WAV临时目录（回车使用默认）: ").strip().strip('"').strip("'")
    if not wav_dir:
        wav_dir = default_wav_dir
    
    # 选择API输出目录
    api_dir = input("请输入API输出目录（回车使用默认）: ").strip().strip('"').strip("'")
    if not api_dir:
        api_dir = default_api_dir
    
    # 选择通道模式
    print("\n通道模式选择:")
    print("1. horizontal - 只使用水平通道（推荐）")
    print("2. vertical - 只使用垂直通道")
    print("3. mix - 混合两个通道")
    
    channel_choice = input("请选择通道模式 (1/2/3，默认1): ").strip() or '1'
    channel_modes = {'1': 'horizontal', '2': 'vertical', '3': 'mix'}
    channel_mode = channel_modes.get(channel_choice, 'horizontal')
    
    # 选择是否清理WAV文件
    print("\n是否在API处理完成后删除临时WAV文件？")
    cleanup = input("删除WAV文件 (y/n，默认n): ").strip().lower() == 'y'
    
    # 询问是否继续
    print("\n" + "=" * 70)
    print("准备开始转换:")
    print(f"  输入: {mc_dir}")
    print(f"  临时WAV: {wav_dir}")
    print(f"  输出: {api_dir}")
    print(f"  通道模式: {channel_mode}")
    print(f"  清理WAV: {'是' if cleanup else '否'}")
    print("=" * 70)
    
    confirm = input("\n确认开始转换？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    # 执行完整转换流程
    results = convert_xjtu_mc_to_energy_density(
        mc_input_dir=mc_dir,
        wav_output_dir=wav_dir,
        api_output_dir=api_dir,
        channel_mode=channel_mode,
        normalize_method='minmax',
        cleanup_wav=cleanup
    )
    
    print("\n" + "=" * 70)
    print("全部完成！")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
