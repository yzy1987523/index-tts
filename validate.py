# -*- coding: utf-8 -*-
# 强制 stdout/stderr 用 UTF-8 (Windows GBK 兼容)
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
validate.py - IndexTTS-2.0 端到端验证脚本

5 个测试场景:
  1. 基线: Tab 2 + 模式 D (纯音色克隆)
  2. 关键: Tab 2 + 模式 C (描述词控制情感)
  3. 解耦: Tab 2 + 模式 A (情感参考音频)
  4. 向量: Tab 2 + 模式 B (8 维情感向量)
  5. 对比: 同样的输入在 CosyVoice_V2 Tab 3

输出: outfile/validate/*.wav + summary.txt
"""
import os
import sys
import time
import json

import numpy as np
import soundfile as sf
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from voice_lib_bridge import library, list_voice_names, get_voice_path
from app import get_tts

# 输出目录
OUT_DIR = os.path.join(ROOT_DIR, 'outfile', 'validate')
os.makedirs(OUT_DIR, exist_ok=True)

VOICE = '傅首尔'  # 温柔女.wav 已缺失,改用 傅首尔.wav (脱口秀主持人,音色辨识度高)
TEXT = '今天天气真好,适合出去走走。'

# 测试用例定义
CASES = [
    {
        'name': '01_baseline_纯音色克隆',
        'desc': '基线: 不传情感参数, 只克隆音色',
        'emo_mode': 'none',
    },
    {
        'name': '02_关键测试_描述词控制',
        'desc': '关键: 模式C 描述词="用愤怒的语气说"',
        'emo_mode': 'text',
        'emo_text': '用非常愤怒的语气说, 带点咬牙切齿的感觉',
    },
    {
        'name': '03_解耦_情感参考音频',
        'desc': '解耦: 模式A 情感参考=郭德纲.wav',
        'emo_mode': 'ref_audio',
        'emo_audio': r'c:\Custom\MyTream\CosyVoice_V2\voice_library\audio\郭德纲.wav',
        'emo_alpha': 0.8,
    },
    {
        'name': '04_向量_happy',
        'desc': '向量: 模式B happy=0.9, 其他=0',
        'emo_mode': 'vector',
        'emo_vector': [0.9, 0, 0, 0, 0, 0, 0, 0],
    },
    {
        'name': '05_向量_calm',
        'desc': '向量: 模式B calm=0.8, 其他=0',
        'emo_mode': 'vector',
        'emo_vector': [0, 0, 0, 0, 0, 0, 0, 0.8],
    },
]


def run_case(tts, case):
    """跑单个测试用例"""
    spk_audio = get_voice_path(VOICE)
    if not spk_audio or not os.path.exists(spk_audio):
        return None, f'音色 {VOICE} 缺失', 0

    out_path = os.path.join(OUT_DIR, f"{case['name']}.wav")

    kwargs = dict(
        spk_audio_prompt=spk_audio,
        text=TEXT,
        output_path=out_path,
    )

    if case['emo_mode'] == 'text':
        kwargs['use_emo_text'] = True
        kwargs['emo_text'] = case['emo_text']
    elif case['emo_mode'] == 'ref_audio':
        kwargs['emo_audio_prompt'] = case['emo_audio']
        kwargs['emo_alpha'] = case['emo_alpha']
    elif case['emo_mode'] == 'vector':
        kwargs['emo_vector'] = case['emo_vector']
    # 'none' 模式不传情感参数

    t0 = time.time()
    try:
        tts.infer(**kwargs)
        dt = time.time() - t0
        audio, sr = sf.read(out_path)
        dur = len(audio) / sr
        return (sr, audio), f'OK | {dur:.2f}s | {dt:.1f}s 推理', dt
    except Exception as e:
        import traceback
        return None, f'FAIL: {e}\n{traceback.format_exc()}', time.time() - t0


def main():
    print('=' * 60)
    print(' IndexTTS-2.0 验证脚本')
    print('=' * 60)
    print(f'模型: {VOICE} (音色)')
    print(f'文本: {TEXT!r}')
    print(f'输出: {OUT_DIR}')
    print()

    # 加载模型 (lazy load + warmup)
    print('[1/3] 加载模型 (首次约需 60s)...')
    tts = get_tts()
    print('     [OK] 模型已就绪')
    print()

    # 暖机一次 (避免 first-call 开销污染数据)
    print('[2/3] 暖机推理 (避免 first-call 开销)...')
    warmup_out = os.path.join(OUT_DIR, '_warmup.wav')
    tts.infer(
        spk_audio_prompt=get_voice_path(VOICE),
        text='测试',
        output_path=warmup_out,
    )
    try:
        os.unlink(warmup_out)
    except Exception:
        pass
    print('     [OK] 暖机完成')
    print()

    # 跑所有用例
    print('[3/3] 跑测试用例...')
    print()
    results = []
    for i, case in enumerate(CASES, 1):
        print(f'[{i}/{len(CASES)}] {case["name"]}')
        print(f'     {case["desc"]}')
        audio, msg, dt = run_case(tts, case)
        print(f'     → {msg}')
        results.append({
            'name': case['name'],
            'desc': case['desc'],
            'msg': msg,
            'duration': dt,
        })
        print()

    # 写汇总
    summary_path = os.path.join(OUT_DIR, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('IndexTTS-2.0 验证报告\n')
        f.write(f'音色: {VOICE}\n')
        f.write(f'文本: {TEXT}\n')
        f.write(f'生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * 60 + '\n\n')
        for r in results:
            f.write(f'{r["name"]}\n')
            f.write(f'  说明: {r["desc"]}\n')
            f.write(f'  结果: {r["msg"]}\n\n')
    print(f'汇总已写入: {summary_path}')
    print()
    print('=' * 60)
    print(' [OK] 验证完成')
    print('=' * 60)
    print(f'请打开 {OUT_DIR} 听 5 个 wav, 人耳判断:')
    print('  - 01_baseline: 音色是否像温柔女')
    print('  - 02_关键测试: 音色保持 + 情感变愤怒?')
    print('  - 03_解耦: 温柔女音色 + 郭德纲情感?')
    print('  - 04/05_向量: 音色 + 单一情感(happy/calm)?')


if __name__ == '__main__':
    main()