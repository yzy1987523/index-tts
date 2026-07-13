# IndexTTS-2.0 MVP 部署

> B 站 Index 团队开源的零样本 TTS 模型,通过「音色-情感解耦」架构同时解决:
> - ✅ **音色保真** —— 5 秒参考音频即可克隆音色
> - ✅ **描述词控制** —— 4 种情感控制模式独立工作

本目录是 `c:\Custom\MyTream\IndexTTS\`,**与 CosyVoice_V2 平级**,专门用于对比验证
IndexTTS-2.0 vs CosyVoice2 的「音色保真 + 描述词控制」兼得问题。

## 与 CosyVoice_V2 的对比

| 维度 | CosyVoice2-0.5B (Tab 3) | IndexTTS-2.0 |
|------|---------------------|--------------|
| 音色保真 | ✅ 高 | ✅ 高 |
| 描述词控制 | ⚠️ 会失真 | ✅ 不失真(架构解耦) |
| 显存需求 | 4-8 GB | 4-6 GB (FP16) |
| 速度 | ~120 chars/s | ~85 chars/s |
| 中文 | ✅ | ✅ |
| 时长控制 | ❌ | ✅ (本版本暂未开放) |

## 快速开始

```bash
# 一键启动 (会自动检查模型)
start.bat

# 或手动启动
uv run python app.py --port 9880
```

浏览器访问 `http://127.0.0.1:9880`

## 工作区结构

```
c:\Custom\MyTream\IndexTTS\
├── .venv/                    # uv 管理的 Python 3.11 环境
├── checkpoints/              # IndexTTS-2.0 模型权重(≈5.9GB,从 ModelScope 下载)
├── voice_library/            # → 软链接到 ..\CosyVoice_V2\voice_library
│   ├── audio/                # 11 个音色 wav
│   └── voices.json           # 音色元数据
├── outfile/                  # 输出目录
├── app.py                    # Gradio UI 入口(2 Tab)
├── voice_lib_bridge.py       # 音色库适配层(复用 CosyVoice_V2 的 VoiceLibrary 类)
├── start.bat                 # Windows 一键启动
├── pyproject.toml            # uv 项目配置
└── README.md
```

## UI 用法

### Tab 1: 音色库管理

- 上传音频 → 填写 prompt 文本 → 添加
- 已有音色自动显示(从 voice_library/voices.json 读取)
- 删除:输入音色名 → 删除

### Tab 2: 按描述词生成(核心)

4 种情感控制模式互斥,选择其一:

| 模式 | 输入 | 用途 |
|------|------|------|
| **A. 情感参考音频** | 上传音频文件 | 把音频的情感迁移到目标音色 |
| **B. 8 维情感向量** | 8 个 slider | 精确控制 8 种情感强度 |
| **C. 文本自动推断** | 文本框填描述 | "用愤怒的语气说" → Qwen 推断 |
| **D. 纯音色克隆** | (无) | 不传情感参数,只克隆音色 |

**示例文本**:`今天天气真好,适合出去走走。`

## 与 CosyVoice_V2 共存

| 服务 | 端口 | 启动 |
|------|------|------|
| CosyVoice_V2 (batch_tts_ui.py) | 9875 | 各自独立启动 |
| **IndexTTS-2 (app.py)** | **9880** | 本项目 |

两个服务**共用同一个 voice_library/**(软链),添加的音色对两边都可见。

## 验证场景(Phase 5)

1. **基线**: Tab 2 + 模式 D,音色=温柔女 → 音色高度相似
2. **关键测试**: Tab 2 + 模式 C,音色=温柔女,描述="用愤怒的语气说" → **音色保持 + 情感变愤怒**
3. **解耦验证**: Tab 2 + 模式 A,音色=温柔女,情感音频=郭德纲 → 温柔女音色 + 郭德纲的情感
4. **向量精确控制**: Tab 2 + 模式 B,`[happy=0.9, others=0]` → 音色 + happy 情感
5. **横向对比**: 同样输入跑 CosyVoice_V2 Tab 3 → 验证 IndexTTS 失真更小

## 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `CUDA out of memory` | 8GB 显存不够 | 关闭 CosyVoice_V2 服务 |
| 模型加载慢(>30s) | 首次加载到 GPU | 正常,等等即可 |
| `ModuleNotFoundError: indextts` | 没在项目目录 | `cd c:\Custom\MyTream\IndexTTS` |
| 模型下载失败 | ModelScope 网络问题 | 重试或换 HF 镜像 |

## 已知限制

- **时长控制**: 论文支持但本版本未开放(`max_text_tokens_per_segment` 参数暂时不用)
- **流式输出**: 不支持
- **flash-attn**: Windows 跳过,改用 SDPA(性能损失 <5%)
- **LICENSE**: 自定义(`LicenseRef-Bilibili-IndexTTS`),商用前读 [LICENSE.txt](checkpoints/LICENSE.txt)

## 参考资料

- [官方仓库](https://github.com/index-tts/index-tts)
- [论文 arXiv:2506.21619](https://arxiv.org/abs/2506.21619)
- [ModelScope 模型](https://modelscope.cn/models/IndexTeam/IndexTTS-2)
- [官方 Demo](https://index-tts.github.io/index-tts2.github.io/)