import os
import re
import time
import threading
import streamlit as st
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# 模型配置
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
            "qwen3.6-flash",
            "qwen3.7-max",
            "qwen-flash",
            "deepseek-v4-flash",
            "deepseek-v4",
            "deepseek-r1",
        ]
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "models": [
            "deepseek-v4-flash",
            "deepseek-v4",
            "deepseek-v4-pro",
            "deepseek-r1",
        ]
    }
}

# 全局变量
available_models = []
models_lock = threading.Lock()

def count_chapters(text):
    """统计章节数量"""
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
        "限流", "配额", "超限", "额度不足", "FreeTierOnly"
    ]
    error_lower = str(error_msg).lower()
    return any(kw in error_lower for kw in rate_limit_keywords)

def split_by_chapters(content):
    """按章节分割文本"""
    chapter_patterns = [
        r'第[一二三四五六七八九十百千零\d]+卷.*第[一二三四五六七八九十百千零\d]+章',
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
    
    for line in content.splitlines(keepends=True):
        is_chapter = False
        for p in chapter_patterns:
            if re.search(p, line):
                is_chapter = True
                break
        
        if is_chapter:
            if current_lines:
                chunks.append(''.join(current_lines))
                current_lines = []
            current_lines.append(line)
            if not line.endswith('\n'):
                current_lines.append('\n')
            in_chapter = True
        elif in_chapter:
            current_lines.append(line)
        else:
            current_lines.append(line)
    
    if current_lines:
        chunks.append(''.join(current_lines))
    
    return chunks

def process_chunk(client, chunk, chunk_index, base_url, api_key, progress_callback=None, log_callback=None, 
                  strict_threshold=0.05, conservative_threshold=0.015, max_retries=3):
    """处理单个章节（线程安全版本，不直接调用Streamlit组件）"""
    global available_models
    
    # 使用线程安全的日志收集
    logs = []
    def safe_log(msg):
        logs.append(msg)
        if log_callback:
            try:
                log_callback(msg)
            except:
                pass  # 忽略Streamlit调用失败
    
    strict_prompt = """清理网文小说中的防盗版乱码。只删除乱码字符，不修改任何正常文字。章节标题必须原样保留。不确定就保留。只输出清理后的纯文本。"""
    conservative_prompt = """清理网文小说中的防盗版乱码。只删除100%确定的乱码字符，不确定就保留原文。章节标题必须原样保留。只输出清理后的纯文本。"""
    
    input_chapter_count = count_chapters(chunk)
    input_len = len(chunk)
    
    ad_keywords = ['本书由', '小说版权归', '仅供个人学习', '免费', '群号', '提取']
    ad_total = 0
    for line in chunk.splitlines(keepends=True):
        if any(kw in line for kw in ad_keywords):
            ad_total += len(line)
    
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
    
    with models_lock:
        total_models = len(available_models)
        models_to_try = list(available_models)
    
    if not models_to_try:
        safe_log(f"[章节{chunk_index+1}] 所有模型额度已耗尽，跳过")
        report["状态"] = "所有模型额度耗尽"
        return chunk, report, logs
    
    for model_idx, model in enumerate(models_to_try):
        report["使用的模型"] = model
        safe_log(f"[章节{chunk_index+1}] 使用模型: {model} ({model_idx+1}/{total_models})")
        
        # 严格模式
        for attempt in range(max_retries):
            try:
                safe_log(f"  [章节{chunk_index+1}] 严格模式 尝试 {attempt+1}/{max_retries}")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": strict_prompt}, {"role": "user", "content": chunk}],
                    timeout=120
                )
                output = response.choices[0].message.content.strip()
                output_len = len(output)
                
                output_chapter_count = count_chapters(output)
                if output_chapter_count < input_chapter_count:
                    safe_log(f"  [章节{chunk_index+1}] 章节减少({input_chapter_count}→{output_chapter_count}),重试")
                    continue
                
                total_removed = input_len - output_len
                ad_removed = min(ad_total, total_removed)
                content_removed = total_removed - ad_removed
                content_len = input_len - ad_total
                content_removal_ratio = content_removed / content_len if content_len > 0 else 0
                
                safe_log(f"  [章节{chunk_index+1}] 严格模式 删除率: {content_removal_ratio*100:.2f}% (阈值{strict_threshold*100:.1f}%)")
                
                if content_removal_ratio > strict_threshold:
                    safe_log(f"  [章节{chunk_index+1}] 删正文{content_removal_ratio*100:.1f}%,超过阈值{strict_threshold*100:.1f}%,进入保守模式")
                    break
                
                safe_log(f"  [章节{chunk_index+1}] 严格模式成功 (模型: {model})")
                report["最终模式"] = "严格模式"
                report["删除字数"] = content_removed
                report["删除率"] = content_removal_ratio
                report["清理后字数"] = output_len
                report["状态"] = "成功"
                return output, report, logs
            except Exception as e:
                error_msg = str(e)
                safe_log(f"  [章节{chunk_index+1}] 错误: {error_msg}")
                if is_rate_limit_error(e):
                    with models_lock:
                        if model in available_models:
                            available_models.remove(model)
                            safe_log(f"\n>>> 警告：模型 {model} 额度已耗尽，已从本次运行中剔除！剩余 {len(available_models)} 个模型 <<<\n")
                    break
        
        # 保守模式
        safe_log(f"  [章节{chunk_index+1}] 切换保守模式...")
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
                safe_log(f"  [章节{chunk_index+1}] 保守章节减少,保留原文")
                report["状态"] = "章节丢失-保留原文"
                return chunk, report, logs
            
            total_removed = input_len - output_len
            ad_removed = min(ad_total, total_removed)
            content_removed = total_removed - ad_removed
            content_len = input_len - ad_total
            content_removal_ratio = content_removed / content_len if content_len > 0 else 0
            
            safe_log(f"  [章节{chunk_index+1}] 保守模式 删除率: {content_removal_ratio*100:.2f}% (阈值{conservative_threshold*100:.1f}%)")
            if content_removal_ratio > conservative_threshold:
                safe_log(f"  [章节{chunk_index+1}] 保守删{content_removal_ratio*100:.1f}%,保留原文")
                report["最终模式"] = "保守模式-删除过多"
                report["状态"] = "删除过多-保留原文"
                return chunk, report, logs
            
            safe_log(f"  [章节{chunk_index+1}] 保守模式成功 (模型: {model})")
            report["最终模式"] = "保守模式"
            report["删除字数"] = content_removed
            report["删除率"] = content_removal_ratio
            report["清理后字数"] = output_len
            report["状态"] = "成功"
            return output, report, logs
        except Exception as e:
            error_msg = str(e)
            safe_log(f"  [章节{chunk_index+1}] 保守失败: {error_msg}")
            if is_rate_limit_error(e):
                with models_lock:
                    if model in available_models:
                        available_models.remove(model)
                        safe_log(f"\n>>> 警告：模型 {model} 额度已耗尽，已从本次运行中剔除！剩余 {len(available_models)} 个模型 <<<\n")
                continue
    
    safe_log(f"  [章节{chunk_index+1}] 所有模型均失败，保留原文")
    report["状态"] = "全部失败"
    return chunk, report, logs

def main():
    st.set_page_config(page_title="小说乱码清理工具", layout="wide")
    st.title("📖 小说乱码清理工具")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置选项")
        
        # 提供商选择
        provider = st.selectbox("选择API提供商", ["qwen", "deepseek"], key="provider")
        
        # 模型列表
        st.subheader("可用模型")
        selected_models = st.multiselect(
            "选择要使用的模型（按顺序优先级）",
            MODEL_CONFIGS[provider]["models"],
            default=MODEL_CONFIGS[provider]["models"][:5],
            key="selected_models"
        )
        
        # 并发线程数
        max_workers = st.slider("并发线程数", min_value=1, max_value=10, value=2, 
                                help="推荐1-5，太高容易触发限流", key="max_workers")
        
        # 清理阈值配置
        st.subheader("🧹 清理阈值")
        strict_threshold = st.slider("严格模式阈值(%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, 
                                     help="超过此百分比删除率则进入保守模式", key="strict_threshold") / 100
        conservative_threshold = st.slider("保守模式阈值(%)", min_value=0.0, max_value=20.0, value=1.5, step=0.5, 
                                           help="超过此百分比删除率则保留原文", key="conservative_threshold") / 100
        max_retries = st.slider("严格模式重试次数", min_value=1, max_value=5, value=3, 
                                help="严格模式失败后的重试次数", key="max_retries")
        
        # API Key
        api_key = st.text_input("输入API Key", type="password", key="api_key")
    
    # 主界面
    st.subheader("📁 文件上传")
    uploaded_file = st.file_uploader("选择要处理的文本文件", type="txt", key="uploaded_file")
    
    # 状态变量
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    if "output_text" not in st.session_state:
        st.session_state.output_text = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # 实时日志显示区域（在进度条上方）
    log_area = st.empty()
    
    # 进度条
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # 检查是否可以处理
    can_process = uploaded_file is not None and len(api_key.strip()) > 0 and len(selected_models) > 0
    
    # 开始处理按钮
    if st.button("🚀 开始处理", disabled=not can_process):
        if not uploaded_file:
            st.error("请先上传文件")
        elif not api_key:
            st.error("请输入API Key")
        elif not selected_models:
            st.error("请选择至少一个模型")
        else:
            # 重置状态
            st.session_state.logs = []
            st.session_state.progress = 0
            st.session_state.output_text = ""
            st.session_state.processing = True
            
            # 读取文件内容
            content = uploaded_file.read().decode('utf-8')
            local_logs = [f"✅ 文件读取完成，共 {len(content)} 字符"]
            
            # 按章节分割
            chunks = split_by_chapters(content)
            local_logs.append(f"✅ 章节分割完成，共 {len(chunks)} 个章节")
            
            # 更新进度条
            progress_bar.progress(0)
            progress_text.text(f"初始化完成，准备处理 0/{len(chunks)}")
            log_area.info("\n".join(local_logs))
            
            # 初始化全局模型列表
            global available_models
            available_models = list(selected_models)
            
            # 创建客户端
            local_logs.append(f"🔄 正在连接 {provider} API...")
            log_area.info("\n".join(local_logs))
            base_url = MODEL_CONFIGS[provider]["base_url"]
            client = OpenAI(api_key=api_key, base_url=base_url)
            local_logs.append(f"✅ API 连接成功，使用模型: {selected_models[0]}")
            log_area.info("\n".join(local_logs))
            
            # 结果列表
            results = [None] * len(chunks)
            local_progress = [0]
            
            def log(msg):
                local_logs.append(msg)
                # 实时更新日志
                log_area.info("\n".join(local_logs))
            
            progress_lock = threading.Lock()
            
            def update_progress():
                with progress_lock:
                    local_progress[0] += 1
                    current_progress = min(local_progress[0], len(chunks))
                    st.session_state.progress = current_progress
                    progress_bar.progress(current_progress / len(chunks))
                    progress_text.text(f"处理进度: {current_progress}/{len(chunks)}")
            
            # 处理逻辑
            if max_workers > 1 and len(chunks) > 1:
                local_logs.append(f"🔄 开始并行处理（{max_workers}个线程）...")
                log_area.info("\n".join(local_logs))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for i, chunk in enumerate(chunks):
                        local_logs.append(f"📤 提交章节 {i+1} 到处理队列...")
                        log_area.info("\n".join(local_logs))
                        future = executor.submit(process_chunk, client, chunk, i, base_url, api_key, update_progress, log, 
                                                strict_threshold, conservative_threshold, max_retries)
                        futures[future] = i
                    
                    local_logs.append(f"✅ 已提交所有章节到处理队列")
                    log_area.info("\n".join(local_logs))
                    
                    for future in as_completed(futures):
                        i = futures[future]
                        exc = future.exception()
                        if exc:
                            import traceback
                            error_details = traceback.format_exception(type(exc), exc, exc.__traceback__)
                            error_msg = f"章节 {i+1} 处理失败: {str(exc)}\n{''.join(error_details[-3:])}"
                            local_logs.append(error_msg)
                            results[i] = chunks[i]
                        else:
                            try:
                                cleaned_text, _, chunk_logs = future.result()
                                results[i] = cleaned_text
                                for log_msg in chunk_logs:
                                    local_logs.append(log_msg)
                                local_logs.append(f"✅ 章节 {i+1} 处理完成")
                            except Exception as e:
                                import traceback
                                error_details = traceback.format_exception(type(e), e, e.__traceback__)
                                error_msg = f"章节 {i+1} 获取结果失败: {str(e)}\n{''.join(error_details[-3:])}"
                                local_logs.append(error_msg)
                                results[i] = chunks[i]
                        
                        update_progress()
                        log_area.info("\n".join(local_logs))
                
                    # 并行处理完成后更新状态
                    output_text = ''.join([r for r in results if r])
                    local_logs.append("🎉 所有章节处理完成！")
                    local_logs.append(f"📊 处理统计：共 {len(chunks)} 个章节")
                    
                    # 更新状态
                    st.session_state.output_text = output_text
                    st.session_state.logs = local_logs
                    st.session_state.processing = False
                    st.session_state.original_filename = uploaded_file.name
                    
                    # 最终更新
                    progress_bar.progress(1.0)
                    progress_text.text("✅ 处理完成！")
                    log_area.info("\n".join(local_logs))
            else:
                local_logs.append("🔄 开始串行处理...")
                log_area.info("\n".join(local_logs))
                for i, chunk in enumerate(chunks):
                    local_logs.append(f"📝 正在处理章节 {i+1}/{len(chunks)}...")
                    log_area.info("\n".join(local_logs))
                    cleaned_text, _, chunk_logs = process_chunk(client, chunk, i, base_url, api_key, None, log, 
                                                                strict_threshold, conservative_threshold, max_retries)
                    results[i] = cleaned_text
                    for log_msg in chunk_logs:
                        local_logs.append(log_msg)
                    local_logs.append(f"✅ 章节 {i+1} 处理完成")
                    update_progress()
                    log_area.info("\n".join(local_logs))
                
                # 合并结果
                output_text = ''.join([r for r in results if r])
                local_logs.append("🎉 所有章节处理完成！")
                local_logs.append(f"📊 处理统计：共 {len(chunks)} 个章节")
                
                # 更新状态
                st.session_state.output_text = output_text
                st.session_state.logs = local_logs
                st.session_state.processing = False
                st.session_state.original_filename = uploaded_file.name
                
                # 最终更新
                progress_bar.progress(1.0)
                progress_text.text("✅ 处理完成！")
                log_area.info("\n".join(local_logs))
                
                # 直接显示下载按钮（不依赖 rerun）
                st.subheader("📥 下载结果")
                st.download_button(
                    label="下载清理后的文本",
                    data=st.session_state.output_text,
                    file_name="清理后的文本.txt",
                    mime="text/plain"
                )
    
    # 显示最终日志（处理完成后）
    if st.session_state.get("logs"):
        st.subheader("📝 处理日志")
        st.text_area("", value="\n".join(st.session_state.logs), height=300, disabled=True)
    
    # 调试信息（帮助诊断问题）
    st.subheader("🔍 调试信息")
    st.write(f"output_text 是否存在: {st.session_state.get('output_text') is not None}")
    st.write(f"output_text 长度: {len(st.session_state.get('output_text', ''))}")
    st.write(f"processing 状态: {st.session_state.get('processing')}")
    
    # 下载按钮（只要有输出就显示）
    output_text = st.session_state.get("output_text")
    if output_text and len(output_text) > 0:
        st.subheader("📥 下载结果")
        original_name = st.session_state.get("original_filename", "cleaned")
        file_name = f"清理完成_{original_name}"
        st.download_button(
            label="⬇️ 下载清理后的文本",
            data=output_text,
            file_name=file_name,
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
