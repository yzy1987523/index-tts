# -*- coding: utf-8 -*-
"""
app.py - IndexTTS-2.0 MVP UI
============================
2 个 Tab:
  1️⃣ 音色库管理 - 上传/删除音色（复用 CosyVoice_V2 的 voice_library/）
  2️⃣ 按描述词生成 - IndexTTS-2 的核心卖点：音色保真 + 4 种情感控制模式

端口: 9880 （避开 CosyVoice_V2 的 9875 / IndexTTS 默认 7860）

启动: uv run python app.py --port 9880
"""
import argparse
import os
import sys
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from voice_lib_bridge import (
    library, settings,
    list_voice_names, get_voice_path, get_prompt_text, clean_text,
)

# ============================================================
# 模型懒加载 (第一次合成时才加载,避免启动时占用显存)
# ============================================================
_tts = None


def get_tts():
    """懒加载 IndexTTS2 实例"""
    global _tts
    if _tts is None:
        from indextts.infer_v2 import IndexTTS2
        cfg_path = os.path.join(ROOT_DIR, 'checkpoints', 'config.yaml')
        model_dir = os.path.join(ROOT_DIR, 'checkpoints')
        print(f'[app] 加载 IndexTTS-2 模型: {model_dir}')
        _tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=True,           # 8GB 显存必需
            use_cuda_kernel=False,  # Windows 默认关闭
            use_deepspeed=False,
        )
        print('[app] 模型加载完成')
    return _tts


# ============================================================
# Tab 1: 音色库管理
# ============================================================
def do_add_voice(name, audio, prompt_text):
    """添加音色到 voice_library/"""
    if audio is None:
        return list_voice_names(), None, '请上传音频文件'
    if not name or not name.strip():
        return list_voice_names(), None, '请输入音色名'
    if not prompt_text or not prompt_text.strip():
        return list_voice_names(), None, '请填写 Prompt 文本'
    # Gradio 的 audio 组件返回 (sample_rate, np.ndarray) tuple
    audio_path = audio[0] if isinstance(audio, tuple) else audio
    if isinstance(audio_path, str) and os.path.exists(audio_path):
        # 文件路径模式 - 直接传给 library.add
        ok, msg = library.add(name.strip(), audio_path, prompt_text.strip())
    else:
        # ndarray 模式 - 写入临时文件
        sr, data = audio
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=ROOT_DIR)
        sf.write(tmp.name, data, sr)
        tmp.close()
        ok, msg = library.add(name.strip(), tmp.name, prompt_text.strip())
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return list_voice_names(), gr.Audio(value=None), msg


def do_delete_voice(name):
    """删除音色"""
    if not name or not name.strip():
        return list_voice_names(), '请输入要删除的音色名'
    ok, msg = library.remove(name.strip())
    return list_voice_names(), msg


def refresh_table():
    """刷新音色表格"""
    rows = []
    for n in library.list():
        v = library.get(n)
        rows.append([n, v['prompt_text'], v.get('created_at', '')])
    return rows


# ============================================================
# Tab 2: 按描述词生成 (核心)
# ============================================================

# 8 维情感向量 (与 Qwen 模型输出一致)
EMO_LABELS = ['happy', 'angry', 'sad', 'afraid', 'disgusted', 'melancholic', 'surprised', 'calm']


def do_synthesize(voice_name, text,
                  emo_mode,
                  emo_ref_audio, emo_alpha,
                  emo_text,
                  emo_happy, emo_angry, emo_sad, emo_afraid,
                  emo_disgusted, emo_melancholic, emo_surprised, emo_calm):
    """根据所选情感模式调用 IndexTTS2.infer()"""
    if not voice_name:
        return None, '请先选择音色', ''
    if not text or not text.strip():
        return None, '请输入要合成的文本', ''

    spk_audio = get_voice_path(voice_name)
    if not spk_audio or not os.path.exists(spk_audio):
        return None, f'音色 "{voice_name}" 的音频不存在', ''

    # 文本预处理 (跳过小括号默认开)
    clean_text_val, log = clean_text(text, skip_parentheses=True,
                                     skip_brackets=False, skip_angle=False)
    if not clean_text_val.strip():
        return None, '预处理后文本为空', log

    log += f'\n[合成] 音色: {voice_name}'
    log += f'\n[合成] 情感模式: {emo_mode}'
    log += f'\n[合成] 文本: {clean_text_val[:50]}...' if len(clean_text_val) > 50 else f'\n[合成] 文本: {clean_text_val}'

    # 输出到临时文件
    out_fd, out_path = tempfile.mkstemp(suffix='.wav', dir=ROOT_DIR)
    os.close(out_fd)

    try:
        tts = get_tts()
        kwargs = dict(
            spk_audio_prompt=spk_audio,
            text=clean_text_val,
            output_path=out_path,
        )

        if emo_mode == 'ref_audio':
            # 模式 A: 情感参考音频
            if emo_ref_audio is None:
                return None, '请上传情感参考音频', log
            ref_path = emo_ref_audio[0] if isinstance(emo_ref_audio, tuple) and isinstance(emo_ref_audio[0], str) else None
            if ref_path is None and isinstance(emo_ref_audio, tuple):
                # ndarray 模式 - 写临时文件
                sr, data = emo_ref_audio
                tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=ROOT_DIR)
                sf.write(tmp.name, data, sr)
                tmp.close()
                ref_path = tmp.name
            kwargs['emo_audio_prompt'] = ref_path
            kwargs['emo_alpha'] = emo_alpha
            log += f'\n[合成] emo_audio_prompt={ref_path}, emo_alpha={emo_alpha}'

        elif emo_mode == 'vector':
            # 模式 B: 8 维情感向量
            emo_vector = [emo_happy, emo_angry, emo_sad, emo_afraid,
                          emo_disgusted, emo_melancholic, emo_surprised, emo_calm]
            kwargs['emo_vector'] = emo_vector
            log += f'\n[合成] emo_vector={emo_vector}'

        elif emo_mode == 'text':
            # 模式 C: 文本自动推断情感
            if not emo_text or not emo_text.strip():
                return None, '请填写情感描述文本', log
            kwargs['use_emo_text'] = True
            kwargs['emo_text'] = emo_text.strip()
            log += f'\n[合成] use_emo_text=True, emo_text={emo_text.strip()!r}'

        elif emo_mode == 'none':
            # 模式 D: 纯音色克隆 (不传情感参数)
            log += '\n[合成] 纯音色克隆, 无情感参数'
        else:
            return None, f'未知情感模式: {emo_mode}', log

        # 调用推理
        tts.infer(**kwargs)

        # 读取输出
        audio, sr = sf.read(out_path)
        # Gradio 需要 (sample_rate, np.ndarray) tuple
        log += f'\n[合成] ✓ 完成, sr={sr}, {len(audio)/sr:.2f}s'
        return (sr, audio), '✅ 完成', log

    except Exception as e:
        import traceback
        log += f'\n[合成 ERR] {e}'
        log += f'\n{traceback.format_exc()}'
        return None, f'❌ {e}', log
    finally:
        try:
            os.unlink(out_path)
        except Exception:
            pass


# ============================================================
# 构建 UI
# ============================================================
def build_ui():
    with gr.Blocks(title='IndexTTS-2.0 MVP', theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🎙️ IndexTTS-2.0 MVP
**音色保真 + 描述词控制** —— 通过「音色-情感解耦」架构同时兼得。
端口 9880，模型加载于首次合成时。

当前模型: `pretrained_models/IndexTTS-2`
""")

        with gr.Tabs():
            # ===== Tab 1: 音色库管理 =====
            with gr.Tab('1️⃣ 音色库管理'):
                gr.Markdown('### 复用 CosyVoice_V2 的 voice_library/（软链）')

                with gr.Row():
                    with gr.Column(scale=1):
                        v_name = gr.Textbox(label='音色名称', placeholder='如:我的新音色')
                        v_audio = gr.Audio(
                            label='音频文件 (WAV/MP3,≥16kHz)',
                            type='numpy',
                            sources=['upload'],
                        )
                        v_prompt = gr.Textbox(
                            label='对应的 Prompt 文本',
                            placeholder='音频内容必须与这里一字不差',
                            lines=3,
                        )
                        v_add_btn = gr.Button('➕ 添加音色', variant='primary')

                    with gr.Column(scale=2):
                        v_table = gr.Dataframe(
                            headers=['名称', 'Prompt 文本', '创建时间'],
                            datatype=['str', 'str', 'str'],
                            value=refresh_table(),
                            interactive=False,
                            wrap=True,
                            label='已有音色',
                        )
                        with gr.Row():
                            v_del_name = gr.Textbox(label='要删除的音色名', scale=2)
                            v_del_btn = gr.Button('🗑️ 删除', variant='stop', scale=1)
                        v_status = gr.Textbox(label='状态', interactive=False)

                v_add_btn.click(
                    do_add_voice,
                    inputs=[v_name, v_audio, v_prompt],
                    outputs=[gr.State(), v_audio, v_status],
                ).then(
                    refresh_table,
                    outputs=[v_table],
                )

                v_del_btn.click(
                    do_delete_voice,
                    inputs=[v_del_name],
                    outputs=[gr.State(), v_status],
                ).then(
                    refresh_table,
                    outputs=[v_table],
                )

            # ===== Tab 2: 按描述词生成 (核心) =====
            with gr.Tab('2️⃣ 按描述词生成'):
                gr.Markdown("""
### IndexTTS-2 核心能力：音色与情感独立控制

**4 种情感控制模式**（互斥，单选）：
- **A. 情感参考音频**：上传音频文件作为情感参考，音色保持不变
- **B. 8 维情感向量**：精确控制 8 种情感强度
- **C. 文本自动推断**：用自然语言描述情感（Qwen 推断）
- **D. 纯音色克隆**：不传情感参数，只克隆音色
""")

                with gr.Row():
                    with gr.Column(scale=1):
                        t_voice = gr.Dropdown(
                            label='🎭 音色（来自音色库）',
                            choices=list_voice_names(),
                            value=list_voice_names()[0] if list_voice_names() else None,
                        )
                        t_text = gr.Textbox(
                            label='📝 要合成的文本',
                            placeholder='如:今天天气真好,适合出去走走。',
                            lines=4,
                        )

                    with gr.Column(scale=2):
                        t_emo_mode = gr.Radio(
                            label='🎨 情感控制模式',
                            choices=[
                                ('A. 情感参考音频', 'ref_audio'),
                                ('B. 8 维情感向量', 'vector'),
                                ('C. 文本自动推断', 'text'),
                                ('D. 纯音色克隆', 'none'),
                            ],
                            value='none',
                        )

                        # 模式 A 的 UI
                        with gr.Group(visible=False) as g_ref:
                            t_emo_ref = gr.Audio(
                                label='情感参考音频 (与音色解耦)',
                                type='numpy',
                                sources=['upload'],
                            )
                            t_emo_alpha = gr.Slider(
                                label='情感强度 (0-1)',
                                minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                            )

                        # 模式 B 的 UI (8 个 slider)
                        with gr.Group(visible=False) as g_vec:
                            t_emo_happy = gr.Slider(label='happy (开心)', minimum=0, maximum=1, value=0, step=0.05)
                            t_emo_angry = gr.Slider(label='angry (愤怒)', minimum=0, maximum=1, value=0, step=0.05)
                            t_emo_sad = gr.Slider(label='sad (悲伤)', minimum=0, maximum=1, value=0, step=0.05)
                            t_emo_afraid = gr.Slider(label='afraid (害怕)', minimum=0, maximum=1, value=0, step=0.05)
                            t_emo_disgusted = gr.Slider(label='disgusted (厌恶)', minimum=0, maximum=1, value=0, step=0.05)
                            t_emo_melancholic = gr.Slider(label='melancholic (忧郁)', minimum=0, maximum=1, value=0, step=0.05)
                            t_emo_surprised = gr.Slider(label='surprised (惊讶)', minimum=0, maximum=1, value=0, step=0.05)
                            t_emo_calm = gr.Slider(label='calm (平静)', minimum=0, maximum=1, value=0, step=0.05)

                        # 模式 C 的 UI
                        with gr.Group(visible=False) as g_txt:
                            t_emo_text = gr.Textbox(
                                label='情感描述文本（Qwen 自动推断 8 维向量）',
                                placeholder='如:用非常愤怒的语气说,带点咬牙切齿的感觉',
                                lines=2,
                            )

                t_btn = gr.Button('🎨 按描述词生成', variant='primary', size='lg')

                with gr.Row():
                    t_audio = gr.Audio(label='🔊 生成的音频', type='numpy')
                    t_status = gr.Textbox(label='📊 状态', interactive=False)
                t_log = gr.Textbox(label='📋 执行日志', lines=8, interactive=False, max_lines=20)

                # 情感模式切换 - 显示/隐藏对应 UI
                def toggle_emo_ui(mode):
                    return {
                        g_ref: gr.update(visible=(mode == 'ref_audio')),
                        g_vec: gr.update(visible=(mode == 'vector')),
                        g_txt: gr.update(visible=(mode == 'text')),
                    }

                t_emo_mode.change(
                    toggle_emo_ui,
                    inputs=[t_emo_mode],
                    outputs=[g_ref, g_vec, g_txt],
                )

                t_btn.click(
                    do_synthesize,
                    inputs=[
                        t_voice, t_text, t_emo_mode,
                        t_emo_ref, t_emo_alpha,
                        t_emo_text,
                        t_emo_happy, t_emo_angry, t_emo_sad, t_emo_afraid,
                        t_emo_disgusted, t_emo_melancholic, t_emo_surprised, t_emo_calm,
                    ],
                    outputs=[t_audio, t_status, t_log],
                )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9880, help='监听端口')
    parser.add_argument('--no_browser', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--share', action='store_true', help='生成公网链接 (调试用)')
    args = parser.parse_args()

    print(f'[启动] 端口: {args.port}')
    print(f'[音色库] {len(list_voice_names())} 个音色')

    demo = build_ui()
    demo.queue(max_size=8, default_concurrency_limit=1)  # 推理慢,限制并发
    demo.launch(
        server_port=args.port,
        inbrowser=not args.no_browser,
        server_name='0.0.0.0',
        share=args.share,
        max_file_size='50 MB',
    )


if __name__ == '__main__':
    main()