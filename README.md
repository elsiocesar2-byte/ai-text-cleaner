# 小说乱码清理工具

一个基于 Streamlit 的交互式网页应用，用于清理小说中的防盗版乱码。

## 功能特性

- 📁 文件上传：支持 `.txt` 格式文本文件
- 🔄 多模型支持：支持 Qwen（通义千问）和 DeepSeek 两种 API
- ⚡ 并行处理：支持 1-10 并发线程
- 🌟 自动切换：模型额度耗尽时自动切换至备选模型
- 📊 实时日志：显示处理进度和详细日志
- 🌓 日夜模式：支持亮色/暗色主题切换

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行应用

```bash
streamlit run app.py
```

### 3. 使用说明

1. 在侧边栏选择 API 提供商（Qwen 或 DeepSeek）
2. 选择要使用的模型（可多选，按顺序优先级）
3. 设置并发线程数（推荐 1-5）
4. 输入 API Key
5. 上传 `.txt` 文件
6. 点击「开始处理」按钮
7. 处理完成后点击下载按钮获取清理后的文本

## 配置说明

### API Key 获取

- **Qwen（通义千问）**：前往 [阿里云 DashScope](https://dashscope.console.aliyun.com/) 获取
- **DeepSeek**：前往 [DeepSeek 控制台](https://platform.deepseek.com/) 获取

### 模型列表

#### Qwen 模型
- `qwen-plus` - 标准模型
- `qwen-turbo` - 快速模型
- `qwen-max` - 大模型
- 更多模型请查看代码中的 MODEL_CONFIGS

#### DeepSeek 模型
- `deepseek-v4-flash` - 快速模型
- `deepseek-v4` - 标准模型
- `deepseek-v4-pro` - 专业模型
- `deepseek-r1` - R1 模型

## 项目结构

```
.
├── app.py              # 主应用文件
├── requirements.txt    # 依赖清单
├── README.md          # 说明文档
└── .streamlit/        # Streamlit 配置目录
    └── config.toml    # 配置文件
```

## 技术栈

- Python 3.10+
- Streamlit 1.50+
- OpenAI SDK 2.0+

## 许可证

MIT License
