# -*- coding: utf-8 -*-
"""
voice_lib_bridge.py
===================
IndexTTS 与 CosyVoice_V2 音色库的适配层。

字段映射 (完全兼容):
    library.voices[name] = {
        'audio_path': '...\\voice.wav',  # → IndexTTS 的 spk_audio_prompt
        'prompt_text': '...',            # → 文本对齐参考
        'created_at': '2026-...'
    }

注意: 这里把 VoiceLibrary / Settings / preprocess_text 内联,
避免 import batch_tts_ui.py 拖入整个 CosyVoice 依赖链。
原代码见 c:\\Custom\\MyTream\\CosyVoice_V2\\batch_tts_ui.py:49-294
"""
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

# ===== 路径配置 =====
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_LIB_DIR = os.path.join(ROOT_DIR, 'voice_library')
VOICES_JSON = os.path.join(VOICE_LIB_DIR, 'voices.json')
VOICE_AUDIO_DIR = os.path.join(VOICE_LIB_DIR, 'audio')
USER_CONFIG = os.path.join(ROOT_DIR, 'user_config.json')


# ============================================================
# Settings - 用户配置持久化
# ============================================================
class Settings:
    """读/写 user_config.json,保存用户上次的路径等"""

    DEFAULT = {
        'text_folder': '',
        'output_dir': os.path.join(ROOT_DIR, 'outfile'),
    }

    def __init__(self, config_path: str = USER_CONFIG):
        self.path = config_path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                for k, v in self.DEFAULT.items():
                    user_data.setdefault(k, v)
                return user_data
            except Exception:
                pass
        return dict(self.DEFAULT)

    def save(self):
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f'[Settings] 保存失败: {e}')

    def get(self, key, default=None):
        return self.data.get(key, default if default is not None else self.DEFAULT.get(key, ''))

    def set(self, key, value):
        self.data[key] = value
        self.save()


# ============================================================
# VoiceLibrary - 音色库 CRUD
# ============================================================
class VoiceLibrary:
    """音色库: 名称 -> {audio_path, prompt_text, created_at}"""

    def __init__(self, voice_lib_dir: str = VOICE_LIB_DIR):
        self.dir = voice_lib_dir
        self.voices_json = os.path.join(voice_lib_dir, 'voices.json')
        self.audio_dir = os.path.join(voice_lib_dir, 'audio')
        os.makedirs(self.audio_dir, exist_ok=True)
        self.voices = self._load()

    def _load(self):
        if os.path.exists(self.voices_json):
            try:
                with open(self.voices_json, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f'[警告] 读取音色库失败: {e}, 使用空库')
        return {}

    def _save(self):
        with open(self.voices_json, 'w', encoding='utf-8') as f:
            json.dump(self.voices, f, ensure_ascii=False, indent=2)

    def add(self, name, audio_path, prompt_text):
        """添加音色"""
        if not name or not name.strip():
            return False, '音色名不能为空'
        name = name.strip()
        if name in self.voices:
            return False, f'音色名 "{name}" 已存在'

        if not audio_path or not os.path.exists(audio_path):
            return False, '音频文件不存在'
        if not prompt_text or not prompt_text.strip():
            return False, 'prompt 文本不能为空'

        # 复制音频到 voice_library/audio/
        ext = Path(audio_path).suffix
        dest = os.path.join(self.audio_dir, f'{name}{ext}')
        try:
            shutil.copy2(audio_path, dest)
        except Exception as e:
            return False, f'复制音频失败: {e}'

        self.voices[name] = {
            'audio_path': dest,
            'prompt_text': prompt_text.strip(),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        self._save()
        return True, f'音色 "{name}" 添加成功'

    def remove(self, name):
        """删除音色"""
        if name not in self.voices:
            return False, f'音色 "{name}" 不存在'
        audio = self.voices[name].get('audio_path')
        if audio and os.path.exists(audio):
            try:
                os.remove(audio)
            except Exception:
                pass
        del self.voices[name]
        self._save()
        return True, f'音色 "{name}" 已删除'

    def list(self):
        """返回所有音色名（只返回音频文件存在的）"""
        return [n for n in self.voices.keys()
                if self.voices[n].get('audio_path')
                and os.path.exists(self.voices[n]['audio_path'])]

    def get(self, name):
        """获取音色配置"""
        return self.voices.get(name)


# ============================================================
# preprocess_text - 文本标记预处理
# ============================================================
def preprocess_text(text: str, skip_parentheses: bool = False, skip_brackets: bool = False,
                    skip_angle: bool = False, skip_custom: Optional[str] = None,
                    log_text: str = '') -> Tuple[str, str]:
    """
    预处理文本: 删除标记内容
    支持:
      - 小括号 () 和全角 ()
      - 中括号 [] 和全角 【】(警告:会和 IndexTTS/CosyVoice 的特殊 token 冲突)
      - 尖括号 <>(警告:会和 <strong>、<|tag|> 冲突)
      - 自定义正则
    """
    original = text
    skipped_parts = []

    if skip_parentheses:
        new_text = re.sub(r'[\(（][^\)）]*[\)）]', '', text)
        skipped = re.findall(r'[\(（][^\)）]*[\)）]', text)
        skipped_parts.extend(skipped)
        text = new_text

    if skip_brackets:
        new_text = re.sub(r'[\[【][^\]】]*[\]】]', '', text)
        skipped = re.findall(r'[\[【][^\]】]*[\]】]', text)
        skipped_parts.extend(skipped)
        text = new_text

    if skip_angle:
        new_text = re.sub(r'[\<《][^\>》]*[\>》]', '', text)
        skipped = re.findall(r'[\<《][^\>》]*[\>》]', text)
        skipped_parts.extend(skipped)
        text = new_text

    if skip_custom and skip_custom.strip():
        try:
            new_text = re.sub(skip_custom.strip(), '', text)
            skipped = re.findall(skip_custom.strip(), text)
            skipped_parts.extend(skipped)
            text = new_text
        except re.error as e:
            log_text += f'\n[警告] 自定义正则错误: {e}'

    text = re.sub(r'\s+', ' ', text).strip()

    if skipped_parts:
        log_text += f'\n[预处理] 跳过 {len(skipped_parts)} 段标记: {skipped_parts[:3]}' + (
            '...' if len(skipped_parts) > 3 else '')
        log_text += f'\n[预处理] 处理后: {text}'

    return text, log_text


# ============================================================
# 默认实例 (供 app.py 直接 import)
# ============================================================
library = VoiceLibrary(VOICE_LIB_DIR)
settings = Settings(USER_CONFIG)


# ============================================================
# 便捷函数
# ============================================================
def list_voice_names() -> list:
    """列出所有可用音色名"""
    return library.list()


def get_voice_path(name: str) -> Optional[str]:
    """获取音色的音频绝对路径"""
    voice = library.get(name)
    return voice['audio_path'] if voice else None


def get_prompt_text(name: str) -> str:
    """获取音色的参考文本"""
    voice = library.get(name)
    return voice['prompt_text'] if voice else ''


def clean_text(text: str,
               skip_parentheses: bool = True,
               skip_brackets: bool = False,
               skip_angle: bool = False) -> Tuple[str, str]:
    """文本预处理(默认跳过小括号 — 适合旁白)"""
    return preprocess_text(
        text,
        skip_parentheses=skip_parentheses,
        skip_brackets=skip_brackets,
        skip_angle=skip_angle,
        skip_custom=None,
    )


if __name__ == '__main__':
    # 自检
    print(f'[voice_lib_bridge] voice_library dir: {VOICE_LIB_DIR}')
    names = list_voice_names()
    print(f'[voice_lib_bridge] {len(names)} voices: {names}')
    if names:
        n = names[0]
        print(f'[voice_lib_bridge] sample "{n}":')
        print(f'  audio:   {get_voice_path(n)}')
        print(f'  prompt:  {get_prompt_text(n)}')