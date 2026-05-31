import os
import time
import sys
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from openai import OpenAI
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, messagebox

# 全局变量：本次运行可用的模型列表和线程锁
available_models = []
models_lock = threading.Lock()

# 模型配置（按优先级排序）
MODEL_CONFIGS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": [
            "qwen-plus",
            "qwen-plus-latest",
            "qwen-turbo",
            "qwen-max",
            "qwen3-max",
            "qwen3.6-plus",
            "qwen3.6-plus-2026-04-02",
            "qwen3.6-flash",
            "qwen3.6-flash-2026-04-16",
            "qwen3.7-max",
            "qwen3.7-max-preview",
            "qwen3.7-max-2026-05-20",
            "qwen3.7-max-2026-05-17",
            "qwen3.6-max-preview",
            "qwen3.6-35b-a3b",
            "qwen3-max-preview",
            "qwen3-max-2026-01-23",
            "qwen3-max-2025-09-23",
            "qwen-flash",
            "qwen-flash-2025-07-28",
            "qwen3.5-plus",
            "qwen-plus-2025-07-28",
            "qwen-plus-2025-07-14",
            "qwen-plus-2025-04-28",
            "qwen-plus-2025-12-01",
            "qwen-plus-2025-09-11",
            "qwen-plus-2025-01-25",
            "qwen-plus-0112",
            "qwen-plus-1220",
            "qwen-plus-character",
            "qwen-flash-character",
            "qwen-flash-character-2026-01",
            "qwen-plus-25d1-0728",
            "qwen-coder-plus",
            "qwen-coder-turbo",
            "qwen3-coder-plus",
            "qwen3-coder-flash",
            "qwen3-coder-flash-2025-07-28",
            "qwen3-coder-next",
            "qwen3-coder-plus-2025-07-22",
            "qwen3-coder-plus-2025-09-23",
            "qwen3-coder-480b-a35b-instruct",
            "qwen3-coder-30b-a3b-instruct",
            "qwen-math-plus",
            "qwen-math-plus-latest",
            "qwen-math-turbo",
            "qwen-math-plus-plus-0919",
            "qwen-math-plus-plus-0816",
            "qwen2.5-coder-32b-instruct",
            "qwen2.5-72b-instruct",
            "qwen2.5-32b-instruct",
            "qwen2.5-14b-instruct",
            "qwen2.5-7b-instruct",
            "qwen3-32b-instruct",
            "qwen3-14b-instruct",
            "qwen3-8b-instruct",
            "qwen3-4b-instruct",
            "qwen3-32b",
            "qwen3-14b",
            "qwen3-8b",
            "qwen3-30b-a3b",
            "qwen3.6-27b",
            "qwen3.5-35b-a3b",
            "qwen3.5-27b",
            "qwen3.5-122b-a10b",
            "qwen3.5-397b-a17b",
            "qwen3-235b-a22b",
            "qwen3-235b-a22b-instruct",
            "qwen3-235b-a22b-instruct-2507",
            "qwen3-235b-a22b-thinking-2507",
            "qwen3.5-flash",
            "qwen3.5-flash-2026-02-23",
            "qwen3.5-plus-2026-04-20",
            "qwen3.5-plus-2026-02-15",
            "qwen3-30b-a3b-instruct-2507",
            "qwen3-30b-a3b-thinking-2507",
            "qwen3-next-80b-a3b-thinking",
            "qwen3-next-80b-a3b-instruct",
            "qwen3-vl-plus",
            "qwen3-vl-plus-2025-09-23",
            "qwen3-vl-plus-2025-12-19",
            "qwen3-vl-32b-instruct",
            "qwen3-vl-30b-a3b-instruct",
            "qwen3-vl-235b-a22b-instruct",
            "qwen3-vl-235b-a22b-thinking",
            "qwen3-vl-32b-thinking",
            "qwen3-vl-30b-a3b-thinking",
            "qwen3-vl-8b-thinking",
            "qwen3-vl-8b-instruct",
            "qwen3-vl-flash",
            "qwen3-vl-flash-2025-10-15",
            "qwen3-vl-flash-2026-01-22",
            "qwen-vl-plus",
            "qwen-vl-plus-latest",
            "qwen-vl-max",
            "qwen-vl-ocr",
            "qwen-vl-ocr-latest",
            "qwen-vl-ocr-1028",
            "qwen-vl-ocr-2025-04-13",
            "qwen-vl-ocr-2025-08-28",
            "qwen-vl-ocr-2025-11-20",
            "qwen-vl-vl-ocr-2025-04-13",
            "qwen-long",
            "qwen-long-long-latest",
            "qwen-long-2025-01-25",
            "qwen-mt-plus",
            "qwen-mt-turbo",
            "qwen-mt-flash",
            "qwen-mt-lite",
            "qwq-32b-preview",
            "qwq-plus",
            "qvq-max",
            "qvq-plus",
            "gui-plus",
            "gui-plus-2026-02-26",
            "glm-5",
            "glm-5.1",
            "glm-4.7",
            "glm-4.6",
            "glm-4.5",
            "glm-4.5-air",
            "llama-4-maverick-17b-128e-instruct",
            "llama-4-scout-17b-16e-instruct",
            "MiniMax-M2.5",
            "MiniMax-M2.1",
            "kimi-k2.5",
            "kimi-k2.6",
            "kimi-k2-thinking",
            "Moonshot-Kimi-K2-Instruct",
            "tongyi-xiaomi-analysis-flash",
            "tongyi-xiaomi-analysis-pro",
            "tongyi-intent-detect-v3",
            "deepseek-v4-flash",
            "deepseek-v4",
            "deepseek-v4-pro",
            "deepseek-v3.2-exp",
            "deepseek-v3.2",
            "deepseek-v3.1",
            "deepseek-v3",
            "deepseek-r1",
            "deepseek-r1-0528",
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-r1-distill-qwen-7b",
            "deepseek-r1-distill-qwen-14b",
            "deepseek-r1-distill-qwen-32b",
            "deepseek-r1-distill-llama-70b",
        ]
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "models": [
            "deepseek-v4-flash",
        ]
    }
}

def select_file(title="选择文件", filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path

def select_save_file(title="保存文件", initialfile="", defaultextension=".txt", filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.asksaveasfilename(
        title=title,
        initialfile=initialfile,
        defaultextension=defaultextension,
        filetypes=filetypes
    )
    root.destroy()
    return file_path

def ask_test_mode():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    result = tk.messagebox.askyesno("测试模式", "是否启用测试模式？\n\n测试模式仅处理前1万字左右，用于预览清理效果，避免浪费token。")
    root.destroy()
    return result

def ask_parallel_mode():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # 创建自定义对话框
    dialog = tk.Toplevel(root)
    dialog.title("并行处理")
    dialog.resizable(False, False)
    
    tk.Label(dialog, text="是否启用并行处理？").pack(pady=(15, 10))
    
    # 并行数选择
    tk.Label(dialog, text="并行数（越高越快，但更容易触发限流）：").pack()
    
    parallel_var = tk.IntVar(value=2)  # 默认2
    
    frame = tk.Frame(dialog)
    frame.pack(pady=10)
    
    for val, label in [(2, "2（稳定）"), (5, "5（较快）"), (10, "10（最快）")]:
        tk.Radiobutton(frame, text=label, variable=parallel_var, value=val).pack(side="left", padx=10)
    
    tk.Label(dialog, text="推荐：2 或 5，并行10可能触发限流", fg="gray").pack(pady=(0, 10))
    
    result = [False]
    workers = [2]
    
    def on_ok():
        result[0] = True
        workers[0] = parallel_var.get()
        dialog.destroy()
        root.destroy()
    
    def on_cancel():
        dialog.destroy()
        root.destroy()
    
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=15)
    tk.Button(btn_frame, text="是", command=on_ok, width=10).pack(side="left", padx=10)
    tk.Button(btn_frame, text="否（串行）", command=on_cancel, width=10).pack(side="left", padx=10)
    
    dialog.update_idletasks()
    dialog.geometry("350x200")
    
    # 居中
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    dialog.update()
    w = dialog.winfo_width()
    h = dialog.winfo_height()
    dialog.geometry(f"{w}x{h}+{(screen_w-w)//2}+{(screen_h-h)//2}")
    
    dialog.grab_set()
    root.wait_window(dialog)
    
    return result[0], workers[0]

CONFIG_FILE = ".api_config.json"

def load_api_config():
    """加载已保存的API配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                import json
                return json.load(f)
        except:
            return {'provider': '', 'keys': {}}
    return {'provider': '', 'keys': {}}

def save_api_config(provider, api_key):
    """保存API配置到文件"""
    config = load_api_config()
    config['provider'] = provider
    config['keys'][provider] = api_key
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        import json
        json.dump(config, f, indent=2)

def get_saved_api_key(provider):
    """获取指定提供商的已保存API Key"""
    config = load_api_config()
    return config['keys'].get(provider, '')

def ask_api_provider():
    """询问使用哪个API提供商"""
    config = load_api_config()
    last_provider = config.get('provider', '')
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    dialog = tk.Toplevel(root)
    dialog.title("选择API提供商")
    dialog.resizable(False, False)
    
    if last_provider:
        tk.Label(dialog, text=f"上次使用的是 {last_provider.upper()}").pack(pady=(15, 5))
    
    tk.Label(dialog, text="请选择要使用的API提供商：").pack(pady=(5, 10))
    
    result = [None]
    
    def select_qwen():
        result[0] = True
        dialog.destroy()
        root.destroy()
    
    def select_deepseek():
        result[0] = False
        dialog.destroy()
        root.destroy()
    
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=10)
    
    tk.Button(btn_frame, text="Qwen Plus（通义千问）", command=select_qwen, width=20, bg="#f0f0f0").pack(side="left", padx=10)
    tk.Button(btn_frame, text="DeepSeek", command=select_deepseek, width=20, bg="#f0f0f0").pack(side="left", padx=10)
    
    dialog.update_idletasks()
    dialog.geometry("420x150")
    
    # 居中
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    dialog.update()
    w = dialog.winfo_width()
    h = dialog.winfo_height()
    dialog.geometry(f"{w}x{h}+{(screen_w-w)//2}+{(screen_h-h)//2}")
    
    dialog.grab_set()
    root.wait_window(dialog)
    
    return result[0]

def ask_api_key(is_qwen):
    """询问API Key，支持自动加载上次保存的"""
    provider = "qwen" if is_qwen else "deepseek"
    saved_key = get_saved_api_key(provider)
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    api_name = "Qwen Plus（通义千问）" if is_qwen else "DeepSeek"
    
    # 如果有保存的Key，先询问是否使用
    if saved_key:
        use_saved = tk.messagebox.askyesno(
            "使用已保存的API Key", 
            f"检测到上次使用的{api_name} API Key，是否继续使用？\n\n（上次保存的Key：{saved_key[:10]}...）"
        )
        if use_saved:
            root.destroy()
            return saved_key
    
    # 显示输入框
    dialog = tk.Toplevel(root)
    dialog.title("输入API Key")
    dialog.resizable(False, False)
    
    tk.Label(dialog, text=f"请输入 {api_name} 的 API Key：").pack(pady=(15, 5))
    
    entry = tk.Entry(dialog, width=50)
    entry.pack(pady=5, padx=20)
    entry.focus()
    
    result = [None]
    
    def on_ok():
        result[0] = entry.get().strip()
        if result[0]:
            # 保存到配置文件（按提供商分开保存）
            save_api_config(provider, result[0])
        dialog.destroy()
        root.destroy()
    
    def on_cancel():
        dialog.destroy()
        root.destroy()
    
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=15)
    tk.Button(btn_frame, text="确定", command=on_ok, width=10).pack(side="left", padx=10)
    tk.Button(btn_frame, text="取消", command=on_cancel, width=10).pack(side="left", padx=10)
    
    dialog.bind("<Return>", lambda e: on_ok())
    dialog.bind("<Escape>", lambda e: on_cancel())
    
    dialog.update_idletasks()
    dialog.geometry("400x100")
    
    # 居中
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    dialog.update()
    w = dialog.winfo_width()
    h = dialog.winfo_height()
    dialog.geometry(f"{w}x{h}+{(screen_w-w)//2}+{(screen_h-h)//2}")
    
    dialog.grab_set()
    root.wait_window(dialog)
    
    return result[0]

def split_by_chapters(file_path):
    """按章节分割文本，每个章节一个片段"""
    import re
    chapter_patterns = [
        r'第[一二三四五六七八九十百千零\d]+章',
        r'第\s*[一二三四五六七八九十百千零\d]+\s*章',
        r'Chapter\s+\d+',
        r'CHAPTER\s+\d+',
        r'Vol\.?\s*\d+',
        r'第[一二三四五六七八九十百千零\d]+节',
    ]
    
    chunks = []
    current_lines = []
    in_chapter = False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 检测是否是章节标题行
            is_chapter = False
            for p in chapter_patterns:
                if re.search(p, line):
                    is_chapter = True
                    break
            
            if is_chapter:
                # 如果当前有内容，先保存
                if current_lines:
                    chunks.append(''.join(current_lines))
                    current_lines = []
                # 确保章节标题后有换行
                current_lines.append(line)
                if not line.endswith('\n'):
                    current_lines.append('\n')
                in_chapter = True
            elif in_chapter:
                current_lines.append(line)
            else:
                # 开头的非章节内容（如广告），合并到第一章
                current_lines.append(line)
    
    # 保存最后一个章节
    if current_lines:
        chunks.append(''.join(current_lines))
    
    return chunks

def split_text_into_chunks(file_path, min_chars=60000, max_chars=80000):
    """按字符数分割（保留），用于大文件或短上下文API"""
    chunks = []
    current_chunk = []
    current_chars = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_chars = len(line)
            if current_chars + line_chars > max_chars and current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = [line]
                current_chars = line_chars
            else:
                current_chunk.append(line)
                current_chars += line_chars
    
    if current_chunk:
        if current_chars < min_chars and len(chunks) > 0:
            chunks[-1] += ''.join(current_chunk)
        else:
            chunks.append(''.join(current_chunk))
    
    return chunks

def count_chapters(text):
    import re
    patterns = [
        r'第[一二三四五六七八九十百千零\d]+章',
        r'第\s*[一二三四五六七八九十百千零\d]+\s*章',
        r'Chapter\s+\d+',
        r'CHAPTER\s+\d+',
        r'Vol\.?\s*\d+',
        r'第[一二三四五六七八九十百千零\d]+节',
    ]
    count = 0
    for line in text.splitlines():
        for pattern in patterns:
            if re.search(pattern, line):
                count += 1
                break
    return count

def is_rate_limit_error(error_msg):
    """检测是否是限流或配额错误"""
    rate_limit_keywords = [
        "rate limit", "quota", "exceeded", "429", "insufficient",
        "限流", "配额", "超限", "额度不足"
    ]
    error_lower = str(error_msg).lower()
    return any(kw in error_lower for kw in rate_limit_keywords)

def process_chunk_with_model_switch(client, chunk, base_url, api_key, max_retries=3, chunk_index=0):
    """处理单个章节，支持自动切换模型
    返回: (清理后文本, 报告信息字典)
    使用全局 available_models 列表，自动剔除额度耗尽的模型
    """
    global available_models
    
    strict_prompt = """清理网文小说中的防盗版乱码。只删除乱码字符，不修改任何正常文字。章节标题必须原样保留。不确定就保留。只输出清理后的纯文本。"""
    conservative_prompt = """清理网文小说中的防盗版乱码。只删除100%确定的乱码字符，不确定就保留原文。章节标题必须原样保留。只输出清理后的纯文本。"""
    
    input_chapter_count = count_chapters(chunk)
    input_len = len(chunk)
    
    ad_keywords = ['本书由', '小说版权归', '仅供个人学习', '免费', '群号', '提取']
    ad_total = 0
    for line in chunk.splitlines(keepends=True):
        if any(kw in line for kw in ad_keywords):
            ad_total += len(line)
    
    # 报告信息
    report = {
        "章节序号": chunk_index + 1,
        "原文字数": input_len,
        "广告字数": ad_total,
        "使用的模型": "",
        "最终模式": "失败",
        "删除字数": 0,
        "删除率": 0,
        "清理后字数": input_len,
        "状态": "处理中"
    }
    
    # 获取当前可用模型列表的副本（带锁）
    with models_lock:
        total_models = len(available_models)
        models_to_try = list(available_models)
    
    # 如果没有可用模型
    if not models_to_try:
        print(f"  [章节{chunk_index+1}] 所有模型额度已耗尽，跳过")
        report["状态"] = "所有模型额度耗尽"
        return chunk, report
    
    # 遍历当前可用的模型
    for model_idx, model in enumerate(models_to_try):
        report["使用的模型"] = model
        print(f"  [章节{chunk_index+1}] 使用模型: {model} ({model_idx+1}/{total_models})")
        
        # 严格模式（最多3次）
        for attempt in range(max_retries):
            try:
                print(f"    [章节{chunk_index+1}] 严格模式 尝试 {attempt+1}/{max_retries}")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": strict_prompt}, {"role": "user", "content": chunk}],
                    timeout=120
                )
                output = response.choices[0].message.content.strip()
                output_len = len(output)
                
                output_chapter_count = count_chapters(output)
                if output_chapter_count < input_chapter_count:
                    print(f"    [章节{chunk_index+1}] 章节减少({input_chapter_count}→{output_chapter_count}),重试")
                    continue
                
                total_removed = input_len - output_len
                ad_removed = min(ad_total, total_removed)
                content_removed = total_removed - ad_removed
                content_len = input_len - ad_total
                content_removal_ratio = content_removed / content_len if content_len > 0 else 0
                
                print(f"    [章节{chunk_index+1}] 严格模式 删除率: {content_removal_ratio*100:.2f}% (阈值5%)")
                
                if content_removal_ratio > 0.05:
                    print(f"    [章节{chunk_index+1}] 删正文{content_removal_ratio*100:.1f}%,重试")
                    continue
                
                print(f"    [章节{chunk_index+1}] 严格模式成功 (模型: {model})")
                report["最终模式"] = "严格模式"
                report["删除字数"] = content_removed
                report["删除率"] = content_removal_ratio
                report["清理后字数"] = output_len
                report["状态"] = "成功"
                return output, report
            except Exception as e:
                print(f"    [章节{chunk_index+1}] 错误: {e}")
                if is_rate_limit_error(e):
                    # 从全局列表中移除该模型（带锁）
                    with models_lock:
                        if model in available_models:
                            available_models.remove(model)
                            print(f"\n>>> 警告：模型 {model} 额度已耗尽，已从本次运行中剔除！剩余 {len(available_models)} 个模型 <<<\n")
                    break  # 限流错误直接切换模型
        
        # 保守模式（最多1次）
        print(f"  [章节{chunk_index+1}] 切换保守模式...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": conservative_prompt}, {"role": "user", "content": chunk}],
                timeout=120
            )
            output = response.choices[0].message.content.strip()
            output_len = len(output)
            
            output_chapter_count = count_chapters(output)
            if output_chapter_count < input_chapter_count:
                print(f"  [章节{chunk_index+1}] 保守章节减少,保留原文")
                report["状态"] = "章节丢失-保留原文"
                return chunk, report
            
            total_removed = input_len - output_len
            ad_removed = min(ad_total, total_removed)
            content_removed = total_removed - ad_removed
            content_len = input_len - ad_total
            content_removal_ratio = content_removed / content_len if content_len > 0 else 0
            print(f"  [章节{chunk_index+1}] 保守模式 删除率: {content_removal_ratio*100:.2f}% (阈值1.5%)")
            if content_removal_ratio > 0.015:
                print(f"  [章节{chunk_index+1}] 保守删{content_removal_ratio*100:.1f}%,保留原文")
                report["最终模式"] = "保守模式-删除过多"
                report["状态"] = "删除过多-保留原文"
                return chunk, report
            
            print(f"  [章节{chunk_index+1}] 保守模式成功 (模型: {model})")
            report["最终模式"] = "保守模式"
            report["删除字数"] = content_removed
            report["删除率"] = content_removal_ratio
            report["清理后字数"] = output_len
            report["状态"] = "成功"
            return output, report
        except Exception as e:
            print(f"  [章节{chunk_index+1}] 保守失败: {e}")
            if is_rate_limit_error(e):
                # 从全局列表中移除该模型（带锁）
                with models_lock:
                    if model in available_models:
                        available_models.remove(model)
                        print(f"\n>>> 警告：模型 {model} 额度已耗尽，已从本次运行中剔除！剩余 {len(available_models)} 个模型 <<<\n")
                continue
    
    # 所有模型都失败，保留原文
    print(f"  [章节{chunk_index+1}] 所有模型均失败，保留原文")
    report["状态"] = "全部失败"
    return chunk, report

def process_chunk_wrapper(args):
    """并行处理的包装函数"""
    client, chunk, base_url, api_key, chunk_index = args
    cleaned_text, report = process_chunk_with_model_switch(
        client, chunk, base_url, api_key, chunk_index=chunk_index
    )
    return chunk_index, cleaned_text, report

def main():
    # 选择 API
    is_qwen = ask_api_provider()
    
    # 确定使用的配置
    provider = "qwen" if is_qwen else "deepseek"
    config = MODEL_CONFIGS[provider]
    base_url = config["base_url"]
    models = config["models"]
    
    # 输入 API Key（只输入一次）
    api_key = ask_api_key(is_qwen)
    if not api_key:
        print("未输入 API Key，程序退出")
        return
    
    print("请选择要处理的输入文件...")
    input_file = select_file("选择要处理的文本文件")
    
    if not input_file:
        print("未选择文件，程序退出")
        return
    
    if not os.path.exists(input_file):
        print(f"错误：未找到文件 {input_file}")
        return
    
    is_test = ask_test_mode()
    
    is_parallel, max_workers = False, 2
    if not is_test:
        parallel_result = ask_parallel_mode()
        is_parallel = parallel_result[0]
        max_workers = parallel_result[1]
    
    print("请选择输出文件保存位置...")
    input_filename = os.path.basename(input_file)
    if is_test:
        default_output = "test_output.txt"
    else:
        name_without_ext = os.path.splitext(input_filename)[0]
        default_output = f"{name_without_ext}_cleaned.txt"
    
    output_file = select_save_file("选择输出文件保存位置", initialfile=default_output)
    if not output_file:
        output_file = default_output
    
    # 创建 client
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 初始化全局可用模型列表
    global available_models
    available_models = list(models)
    print(f"\n配置: {provider.upper()}")
    print(f"备选模型 ({len(models)}个): {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
    print(f"模型额度耗尽后会自动剔除，后续章节直接跳过该模型\n")
    
    if is_test:
        print("正在读取前5万字进行测试...")
        with open(input_file, 'r', encoding='utf-8') as f:
            test_content = ''
            chars_count = 0
            target_chars = 50000
            for line in f:
                if chars_count + len(line) > target_chars:
                    break
                test_content += line
                chars_count += len(line)
        
        print(f"测试文本长度: {len(test_content)} 字符")
        chunks = [test_content]
        output_file = "test_output.txt"
        print(f"测试模式：结果将保存到 {output_file}")
    else:
        print("正在按章节分割文本...")
        chunks = split_by_chapters(input_file)
        print(f"共 {len(chunks)} 个章节")
    
    # 初始化结果列表和报告列表，保持章节顺序
    results = [None] * len(chunks)
    reports = []
    
    if is_parallel and len(chunks) > 1:
        print(f"\n并行处理模式：同时处理 {max_workers} 个章节\n")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 准备任务参数（不需要传 models，使用全局列表）
            tasks = [(client, chunk, base_url, api_key, i) for i, chunk in enumerate(chunks)]
            # 提交任务
            future_to_index = {executor.submit(process_chunk_wrapper, task): i for i, task in enumerate(tasks)}
            completed_count = 0
            
            # 处理完成的任务
            for future in tqdm(as_completed(future_to_index), total=len(chunks), desc="处理进度"):
                try:
                    i, cleaned_text, report = future.result()
                    results[i] = cleaned_text
                    reports.append(report)
                    completed_count += 1
                    # 实时保存
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for ct in results:
                            if ct:
                                if not ct.endswith('\n'):
                                    ct += '\n'
                                f.write(ct)
                except Exception as e:
                    print(f"\n处理第 {future_to_index[future] + 1} 个片段时失败: {e}")
                    print("已处理内容已保存，可以检查输出文件")
                    sys.exit(1)
    else:
        print("\n串行处理模式\n")
        # 串行处理，实时保存
        for i, chunk in enumerate(tqdm(chunks, desc="处理进度")):
            try:
                print(f"\n正在处理第 {i + 1}/{len(chunks)} 章...")
                cleaned_text, report = process_chunk_with_model_switch(
                    client, chunk, base_url, api_key, chunk_index=i
                )
                results[i] = cleaned_text
                reports.append(report)
                # 实时保存已处理的内容
                with open(output_file, 'w', encoding='utf-8') as f:
                    for ct in results:
                        if ct:
                            if not ct.endswith('\n'):
                                ct += '\n'
                            f.write(ct)
                print(f"  已保存到 {output_file}")
            except Exception as e:
                print(f"\n处理第 {i + 1} 个片段时失败: {e}")
                print("已处理内容已保存，可以检查输出文件")
                sys.exit(1)
    
    # 按顺序写入文件
    print(f"\n正在写入输出文件...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for cleaned_text in results:
            if cleaned_text:
                # 确保每章输出后有换行
                if not cleaned_text.endswith('\n'):
                    cleaned_text += '\n'
                f.write(cleaned_text)
    
    # 生成CSV报告
    report_file = output_file.rsplit('.', 1)[0] + '_report.csv'
    import csv
    with open(report_file, 'w', newline='', encoding='utf-8-sig') as f:
        if reports:
            writer = csv.DictWriter(f, fieldnames=reports[0].keys())
            writer.writeheader()
            writer.writerows(reports)
    print(f"报告已保存到 {report_file}")
    
    print(f"\n处理完成！结果已保存到 {output_file}")

if __name__ == "__main__":
    main()