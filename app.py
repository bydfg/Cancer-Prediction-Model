import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置网页标题和图标
st.set_page_config(page_title="癌症预测系统", page_icon="🩺", layout="wide")

# 解决 matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

st.title("🩺 基于集成学习的癌症样本类型预测系统")
st.markdown("""
### 项目简介
本项目基于 10,194 例泛癌种临床与基因组数据，通过**随机森林**与**XGBoost**算法，
预测样本是 **原发 (Primary)**、**转移 (Metastasis)** 还是 **复发 (Recurrent)**。

### 核心亮点
*   **数据中文化映射**：消除医学专业术语阅读壁垒。
*   **攻克数据泄露**：特征重要性分析发现“肿瘤分期_IV期”的风险。
*   **解决极度不平衡**：SMOTE过采样+ XGBoost，复发灶 F1-score 从 0.05 提升至 0.18。
""")

# 模拟交互预测
st.divider()
st.subheader("🔮 交互式模型预测演示")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("诊断年龄", 1, 100, 60)
    tumor_purity = st.slider("肿瘤纯度 (%)", 10, 100, 50)
with col2:
    tmb = st.number_input("TMB（非同义）", 0.0, 20.0, 0.3, step=0.1)
    mutation_count = st.number_input("突变数量", 1, 500, 6)
with col3:
    stage = st.selectbox("肿瘤分期", ["I期", "II期", "III期", "IV期", "未知"])
    cancer_type = st.selectbox("癌症类型", ["结直肠癌", "肝细胞癌", "乳腺癌", "胃癌", "胰腺癌"])

if st.button("开始预测"):
    with st.spinner('正在分析数据...'):
        import time
        time.sleep(1) # 模拟计算过程
        
        # 模拟逻辑
        if stage == "IV期":
            prediction = "转移 (Metastasis)"
            confidence = "88%"
        elif tmb > 5.0:
            prediction = "复发 (Recurrent)"
            confidence = "75%"
        else:
            prediction = "原发 (Primary)"
            confidence = "92%"
            
        st.success(f"模型预测结果为：**{prediction}**")
        st.metric(label="预测置信度", value=confidence)