import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 设置网页标题和图标
st.set_page_config(page_title="癌症样本类型预测系统", page_icon="🩺", layout="wide")

# 1. 加载模型和预处理器（利用缓存避免重复加载）
@st.cache_resource
def load_models():
    preprocessor = joblib.load('preprocessor.pkl')
    model = joblib.load('cancer_model.pkl')
    le = joblib.load('label_encoder.pkl')
    return preprocessor, model, le

preprocessor, model, le = load_models()

# 2. 页面布局设计
st.title("🩺 基于集成学习的癌症样本类型预测系统")
st.markdown("""
### 项目简介
本项目基于 10,194 例泛癌种临床与基因组数据，通过**随机森林**与 **XGBoost（SMOTE+权重强化）**算法，
预测样本是 **原发 (Primary)**、**转移 (Metastasis)** 还是 **复发 (Recurrent)**。

### 核心亮点
*   **数据中文化映射**：消除医学专业术语阅读壁垒。
*   **攻克数据泄露**：发现“肿瘤分期_IV期”风险。
*   **突破极度不平衡**：SMOTE+权重双重强化，大幅提升复发灶识别能力。
""")

st.divider()
st.subheader("🔮 交互式模型预测演示")
st.write("请在下方输入患者临床与基因组特征，点击“开始预测”获取 AI 诊断建议。")

# 3. 构建交互式输入表单
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("诊断年龄 (岁)", 1, 100, 60)
    tumor_purity = st.slider("肿瘤纯度 (%)", 10, 100, 50)
    dna_input = st.number_input("DNA输入量 (ng)", 50, 50000, 3000, step=500)
with col2:
    tmb = st.number_input("TMB（非同义）", 0.0, 20.0, 0.3, step=0.1)
    mutation_count = st.number_input("突变数量", 1, 500, 6)
    sample_coverage = st.number_input("样本覆盖度", 300, 6000, 1200, step=100)
with col3:
    stage = st.selectbox("肿瘤分期", ["I期", "II期", "III期", "IV期", "未知"])
    cancer_type = st.selectbox("癌症类型", ["结直肠癌", "肝细胞癌", "乳腺癌", "胃癌", "胰腺癌"])
    sex = st.selectbox("性别", ["女性", "男性"])
    smoke = st.selectbox("吸烟状况", ["未知", "不吸烟", "吸烟"])

# 默认值（其他影响较小的特征）
specimen_preservation = "福尔马林固定石蜡包埋"
specimen_type = "手术"
treatment = "未治疗"
samples_per_patient = 1

# 根据用户输入自动计算年龄分组和TMB等级
age_group = '<30' if age < 30 else '30-50' if age <= 50 else '50-70' if age <= 70 else '>70'
tmb_level = '高TMB' if tmb > 0.31 else '低TMB'

# 4. 点击按钮进行预测
if st.button("🔮 开始预测"):
    with st.spinner('AI 正在分析临床数据...'):
        # 构造输入 DataFrame（列名必须与训练时完全一致）
        input_data = pd.DataFrame([{
            '诊断年龄': age,
            '肿瘤纯度': tumor_purity,
            'DNA输入量': dna_input,
            '突变数量': mutation_count,
            'TMB（非同义）': tmb,
            '样本覆盖度': sample_coverage,
            '肿瘤分期': stage,
            '癌症类型': cancer_type,
            '性别': sex,
            '吸烟状况': smoke,
            '标本保存类型': specimen_preservation,
            '标本类型': specimen_type,
            '治疗': treatment,
            '每位患者样本数': samples_per_patient,
            '年龄分组': age_group,
            'TMB_等级': tmb_level
        }])
        
        # 使用预处理器转换数据，再进行预测
        try:
            input_prep = preprocessor.transform(input_data)
            pred_encoded = model.predict(input_prep)
            prediction = le.inverse_transform(pred_encoded)[0]
            
            # 显示结果
            st.success(f"✅ AI 预测结果为：**{prediction}**")
            st.warning("⚠️ 免责声明：本结果仅作为课程实训演示，不作为临床诊断依据。")
        except Exception as e:
            st.error(f"预测出错：{e}\n请检查输入数据格式是否正确。")

st.divider()

import os
import streamlit as st

# ==========================================
# 5. 展示已生成的图表及深度业务解读（防报错版）
# ==========================================
st.divider()
st.header("📊 模型训练与评估成果")

# 核心防御：自动获取 app.py 所在的绝对路径，确保无论从哪里启动都能找到图片
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 创建三个标签页，让界面更清爽
tab1, tab2, tab3 = st.tabs(["混淆矩阵", "特征重要性", "肿瘤纯度分析"])

# ---- 图1：混淆矩阵 ----
with tab1:
    col_img1, col_text1 = st.columns([1.5, 1])
    with col_img1:
        img1 = os.path.join(BASE_DIR, "confusion_matrix.png")
        if os.path.exists(img1):
            st.image(img1, caption="强化模型 - 混淆矩阵")
        else:
            st.warning(f"⚠️ 图片未找到，请确认文件确实存在于：{img1}")
    with col_text1:
        st.subheader("📌 图解分析")
        st.markdown("""
        *   **整体表现**：模型准确率稳定在 **0.87**，综合判别能力较强。
        *   **类别挑战**：由于“复发灶”真实样本仅有35例，数据极度不平衡，混淆矩阵显示模型在识别“原发灶”时表现极佳，但“复发灶”与“转移灶”的边界存在一定重叠。
        *   **改进成效**：经过 SMOTE 过采样和样本权重双重强化后，成功打破了基线模型“盲目猜原发灶”的多数类陷阱。
        """)

# ---- 图2：特征重要性 ----
with tab2:
    col_img2, col_text2 = st.columns([1.5, 1])
    with col_img2:
        img2 = os.path.join(BASE_DIR, "importance.png")
        if os.path.exists(img2):
            st.image(img2, caption="特征重要性 Top 15")
        else:
            st.warning(f"⚠️ 图片未找到，请确认文件确实存在于：{img2}")
    with col_text2:
        st.subheader("📌 关键业务洞察")
        st.markdown("""
        *   **核心特征**：排名第一的是 `肿瘤分期_IV期`，其次是 `肿瘤纯度` 和 `TMB`。
        *   **数据泄露警示**：在临床医学上，IV期本身通常意味着已发生转移，用它预测转移属于“逻辑作弊”（Data Leakage）。
        *   **改进方向**：未来的实际应用中应剔除这种滞后性特征，更多依赖 `TMB` 和 `肿瘤纯度` 等独立的分子生物标志物，才是模型具备泛化能力的关键。
        """)

# ---- 图3：肿瘤纯度分析 ----
with tab3:
    col_img3, col_text3 = st.columns([1.5, 1])
    with col_img3:
        img3 = os.path.join(BASE_DIR, "tumor_purity.png")
        if os.path.exists(img3):
            st.image(img3, caption="不同样本类型的肿瘤纯度分布")
        else:
            st.warning(f"⚠️ 图片未找到，请确认文件确实存在于：{img3}")
    with col_text3:
        st.subheader("📌 肿瘤纯度深度解读")
        st.markdown("""
        *   **现象**：复发灶（Recurrent）的肿瘤纯度中位数显著高于原发和转移灶，且箱体较短（分布集中）。
        *   **临床归因**：复发灶肿瘤纯度偏高，可能与肿瘤微环境中基质细胞较少、癌细胞占比极高有关。
        *   **模型价值**：这是一个区分度极高的特征，在特征重要性中排在靠前位置，对预测复发具有极高的临床参考价值。
        """)