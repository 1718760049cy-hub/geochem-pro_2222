import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram, linkage
import base64
import io

st.set_page_config(page_title="GeoChem Pro", layout="wide")

try:
    from openai import OpenAI
except:
    st.error("请运行: pip install openai")
    st.stop()

COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E', '#16A085', '#C0392B', '#8E44AD', '#2980B9', '#27AE60', '#D35400']


if 'dfs' not in st.session_state:
    st.session_state.dfs = {'major':None,'trace':None,'ree':None}
if 'processed' not in st.session_state:
    st.session_state.processed = {'major':None,'trace':None,'ree':None}
if 'figs' not in st.session_state:
    st.session_state.figs = {}
if 'fig_desc' not in st.session_state:
    st.session_state.fig_desc = {}
if 'generated_figures' not in st.session_state:
    st.session_state.generated_figures = {}
if 'figure_descriptions' not in st.session_state:
    st.session_state.figure_descriptions = {}

CHONDRITE = {'La': 0.237, 'Ce': 0.613, 'Pr': 0.0928, 'Nd': 0.457, 'Sm': 0.148, 'Eu': 0.0563, 'Gd': 0.199, 'Tb': 0.0361, 'Dy': 0.246, 'Ho': 0.0546, 'Er': 0.160, 'Tm': 0.0247, 'Yb': 0.161, 'Lu': 0.0246}
MAJOR_ELEMENTS = ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']
LOI_NAMES = ['LOI', 'loi', 'Loss', 'loss', 'H2O', 'H2O+', 'H2O-']
MW = {'Al2O3': 101.96, 'CaO': 56.08, 'Na2O': 61.98, 'K2O': 94.2, 'MgO': 40.30, 'P2O5': 141.94}

def to_mol(df, ox):
    if ox in df.columns:
        return df[ox] / MW.get(ox, 1)
    return pd.Series(0, index=df.index)

def get_combined_data():
    valid = [d for d in st.session_state.processed.values() if d is not None]
    if not valid:
        return None
    merged = valid[0].copy()
    for d in valid[1:]:
        cols = d.columns.difference(merged.columns)
        merged = merged.join(d[cols], how='outer')
    return merged

def save_fig(fig, name, description=""):
    st.session_state.generated_figures[name] = fig
    st.session_state.figure_descriptions[name] = description

def export_fig(fig, name, description=""):
    save_fig(fig, name, description)
    c1, c2, c3 = st.columns(3)
    c1.download_button("📥 SVG", fig.to_image(format="svg"), f"{name}.svg", "image/svg+xml")
    c2.download_button("📥 PDF", fig.to_image(format="pdf"), f"{name}.pdf", "application/pdf")
    c3.download_button("📥 PNG", fig.to_image(format="png", scale=3), f"{name}.png", "image/png")

def call_ai(prompt, api_key, temperature=0.3, max_tokens=2000):
    if not api_key:
        return "请填写API Key"
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是世界顶级地球化学专家，拥有30年沉积地球化学、物源分析、古环境重建经验。请提供专业、准确、详细的分析。"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"API调用失败: {str(e)}"

def run_classification(X, y, features, api_key):
    X_scaled = StandardScaler().fit_transform(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    st.success(f"✅ {len(X)} 样本, {len(features)} 特征, {n_classes} 类别")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        model_type = st.selectbox("模型", ["随机森林", "梯度提升", "SVM", "KNN"])
    with c2:
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.25, 0.05)
    with c3:
        cv_folds = st.slider("交叉验证折数", 3, 10, 5)
    
    c1, c2 = st.columns(2)
    with c1:
        use_blind = st.checkbox("保留盲测集", value=True)
    with c2:
        blind_size = st.slider("盲测集比例", 0.1, 0.25, 0.15) if use_blind else 0
    
    if st.button("🚀 开始训练", type="primary"):
        with st.spinner("训练中..."):
            if use_blind:
                X_main, X_blind, y_main, y_blind = train_test_split(X_scaled, y_enc, test_size=blind_size, random_state=42, stratify=y_enc)
                X_train, X_test, y_train, y_test = train_test_split(X_main, y_main, test_size=test_size, random_state=42, stratify=y_main)
            else:
                X_main, y_main = X_scaled, y_enc
                X_blind, y_blind = None, None
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=test_size, random_state=42, stratify=y_enc)
            
            models_dict = {
                "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
                "梯度提升": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "SVM": SVC(random_state=42, probability=True),
                "KNN": KNeighborsClassifier(n_neighbors=5)
            }
            model = models_dict[model_type]
            model.fit(X_train, y_train)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            cv_scores = cross_val_score(model, X_main, y_main, cv=cv_folds)
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            
            st.markdown("## 📊 验证报告")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("训练准确率", f"{train_acc:.1%}")
            c2.metric("测试准确率", f"{test_acc:.1%}")
            c3.metric("CV均值", f"{cv_scores.mean():.1%}")
            c4.metric("CV标准差", f"±{cv_scores.std():.1%}")
            
            overfit = train_acc - test_acc
            if overfit > 0.15:
                st.warning(f"⚠️ 可能过拟合（差异: {overfit:.1%}）")
            else:
                st.success(f"✅ 泛化良好（差异: {overfit:.1%}）")
            
            st.markdown("### 交叉验证")
            fig_cv = go.Figure(go.Bar(x=[f"Fold {i+1}" for i in range(cv_folds)], y=cv_scores, marker_color=COLORS[:cv_folds], marker_line_color='black', marker_line_width=1, text=[f"{x:.1%}" for x in cv_scores], textposition='outside'))
            fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="red", annotation_text=f"均值: {cv_scores.mean():.1%}")
            fig_cv.update_layout(width=600, height=400, yaxis=dict(title="准确率", range=[0, 1.15], showline=True, linecolor='black'), xaxis=dict(showline=True, linecolor='black'), plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig_cv)
            export_fig(fig_cv, "cv_scores", f"交叉验证结果,均值{cv_scores.mean():.1%}")
            
            st.markdown("### 混淆矩阵")
            cm = confusion_matrix(y_test, y_pred_test)
            fig_cm = go.Figure(go.Heatmap(z=cm, x=[str(c) for c in le.classes_], y=[str(c) for c in le.classes_], colorscale='Blues', text=cm, texttemplate="%{text}", textfont=dict(size=14)))
            fig_cm.update_layout(width=500, height=500, xaxis=dict(title="预测类别", showline=True, linecolor='black'), yaxis=dict(title="真实类别", autorange="reversed", showline=True, linecolor='black'), plot_bgcolor='white')
            st.plotly_chart(fig_cm)
            export_fig(fig_cm, "confusion_matrix", f"混淆矩阵,测试准确率{test_acc:.1%}")
            
            st.markdown("### 分类报告")
            report = classification_report(y_test, y_pred_test, target_names=[str(c) for c in le.classes_], output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(3))
            
            if hasattr(model, 'feature_importances_'):
                st.markdown("### 特征重要性")
                imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
                fig_imp = go.Figure(go.Bar(x=imp_df['Importance'], y=imp_df['Feature'], orientation='h', marker_color=COLORS[0], marker_line_color='black', marker_line_width=1))
                fig_imp.update_layout(width=600, height=max(300, len(features)*25), xaxis=dict(title="重要性", showline=True, linecolor='black'), yaxis=dict(showline=True, linecolor='black'), plot_bgcolor='white')
                st.plotly_chart(fig_imp)
                export_fig(fig_imp, "feature_importance", f"特征重要性,Top3: {imp_df.tail(3)['Feature'].tolist()}")
                
                st.markdown("**关键判别指标 Top 5：**")
                for _, row in imp_df.tail(5).iloc[::-1].iterrows():
                    st.write(f"- **{row['Feature']}**: {row['Importance']:.3f}")
            
            if use_blind and X_blind is not None:
                st.markdown("### 🔒 盲测验证")
                y_pred_blind = model.predict(X_blind)
                blind_acc = accuracy_score(y_blind, y_pred_blind)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("盲测准确率", f"{blind_acc:.1%}")
                c2.metric("与测试集差异", f"{abs(test_acc - blind_acc):.1%}")
                c3.metric("盲测样本数", len(X_blind))
                
                cm_blind = confusion_matrix(y_blind, y_pred_blind)
                fig_cmb = go.Figure(go.Heatmap(z=cm_blind, x=[str(c) for c in le.classes_], y=[str(c) for c in le.classes_], colorscale='Oranges', text=cm_blind, texttemplate="%{text}", textfont=dict(size=14)))
                fig_cmb.update_layout(width=500, height=500, xaxis=dict(title="预测类别"), yaxis=dict(title="真实类别", autorange="reversed"), plot_bgcolor='white')
                st.plotly_chart(fig_cmb)
                export_fig(fig_cmb, "blind_confusion_matrix", f"盲测混淆矩阵,准确率{blind_acc:.1%}")
                
                if blind_acc >= test_acc * 0.95:
                    st.success("✅ 模型泛化能力优秀，可用于物源预测")
                elif blind_acc >= test_acc * 0.85:
                    st.info("ℹ️ 模型泛化能力良好")
                else:
                    st.warning("⚠️ 模型可能过拟合，建议增加样本或简化特征")

with st.sidebar:
    st.title("🔬 GeoChem Pro")
    st.markdown("---")
    api_key = st.text_input("🔑 DeepSeek API Key", type="password")
    st.markdown("---")
    nav = st.radio("📌 功能导航", [
        "1. 数据导入与预处理",
        "2. 风化指标计算",
        "3. 风化指标图",
        "4. 二元图",
        "5. 三角图",
        "6. PCA双标图",
        "7. 聚类分析",
        "8. 物源分类",
        "9. AI智能分析"
    ])
    st.markdown("---")
    if st.session_state.generated_figures:
        st.markdown(f" 已生成 **{len(st.session_state.generated_figures)}** 个图件")

if nav == "1. 数据导入与预处理":
    st.header(" 数据导入与预处理")
    tab1, tab2, tab3 = st.tabs(["🔴 主量元素", "🔵 微量元素", "🟢 稀土元素"])
    
    with tab1:
        st.subheader("主量元素处理")
        f = st.file_uploader("上传主量元素数据", type=['xlsx', 'csv'], key='major_upload')
        if f:
            raw = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
            raw = raw.set_index(raw.columns[0])
            st.session_state.dfs['major'] = raw
            st.write(f"**原始数据**: {raw.shape[0]} 样本, {raw.shape[1]} 列")
            st.dataframe(raw.head())
            
            st.markdown("---")
            st.markdown("###  预处理设置")
            
            available_major = [c for c in raw.columns if c in MAJOR_ELEMENTS or any(m in str(c) for m in MAJOR_ELEMENTS)]
            other_cols = [c for c in raw.columns if c not in available_major]
            
            st.markdown("**1️ 选择保留的元素**")
            loi_detected = [c for c in raw.columns if any(loi in str(c).upper() for loi in ['LOI', 'LOSS', 'H2O'])]
            if loi_detected:
                st.info(f" 检测到烧失量列: {loi_detected}，建议排除")
            
            default_selected = [c for c in available_major if c not in loi_detected]
            selected_major = st.multiselect("保留以下元素", raw.columns.tolist(), default=default_selected if default_selected else available_major)
            
            st.markdown("**2️ 特殊值处理**")
            for col in selected_major[:5]:
                if raw[col].dtype == object:
                    special = raw[col][raw[col].astype(str).str.contains('<|>|nd|ND|bdl|BDL', na=False)]
                    if len(special) > 0:
                        st.write(f"- {col}: {len(special)} 个特殊值 (如 {special.iloc[0]})")
            
            special_method = st.selectbox("特殊值处理方式", ["替换为检出限的一半", "替换为0", "替换为NaN", "自定义值"])
            custom_val = st.number_input("自定义替换值", value=0.005) if special_method == "自定义值" else None
            
            st.markdown("**3️ 缺失值处理**")
            missing_method = st.selectbox("缺失值处理方式", ["删除含缺失值的行", "均值填充", "中位数填充", "线性插值", "保留不处理"])
            
            st.markdown("**4️ 归一化**")
            do_normalize = st.checkbox("归一化到100%（排除烧失量）", value=True)
            
            if st.button(" 执行主量元素预处理", type="primary"):
                df = raw[selected_major].copy()
                
                for col in df.columns:
                    if df[col].dtype == object:
                        mask = df[col].astype(str).str.contains('<|>|nd|ND|bdl|BDL', na=False)
                        if mask.any():
                            nums = df.loc[mask, col].astype(str).str.extract(r'([0-9.]+)')[0].astype(float)
                            if special_method == "替换为检出限的一半":
                                df.loc[mask, col] = nums / 2
                            elif special_method == "替换为0":
                                df.loc[mask, col] = 0
                            elif special_method == "替换为NaN":
                                df.loc[mask, col] = np.nan
                            elif special_method == "自定义值":
                                df.loc[mask, col] = custom_val
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.astype(float)
                
                n_before = len(df)
                if missing_method == "删除含缺失值的行":
                    df = df.dropna()
                elif missing_method == "均值填充":
                    df = df.fillna(df.mean())
                elif missing_method == "中位数填充":
                    df = df.fillna(df.median())
                elif missing_method == "线性插值":
                    df = df.interpolate(method='linear', axis=0).bfill().ffill()
                n_after = len(df)
                
                if do_normalize:
                    row_sum = df.sum(axis=1)
                    df = df.div(row_sum, axis=0) * 100
                
                st.session_state.processed['major'] = df
                st.success(f" 预处理完成: {n_after} 样本 (删除了 {n_before - n_after} 个), {len(df.columns)} 元素")
                st.dataframe(df.head().round(2))
                st.markdown("**统计摘要：**")
                st.dataframe(df.describe().round(2))
    
    with tab2:
        st.subheader("微量元素处理")
        f = st.file_uploader("上传微量元素数据", type=['xlsx', 'csv'], key='trace_upload')
        if f:
            raw = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
            raw = raw.set_index(raw.columns[0])
            st.session_state.dfs['trace'] = raw
            st.write(f"**原始数据**: {raw.shape[0]} 样本, {raw.shape[1]} 列")
            st.dataframe(raw.head())
            
            selected_trace = st.multiselect("选择保留的元素", raw.columns.tolist(), default=raw.columns.tolist())
            special_method_t = st.selectbox("特殊值处理", ["替换为检出限的一半", "替换为0", "替换为NaN"], key='trace_sp')
            missing_method_t = st.selectbox("缺失值处理", ["删除含缺失值的行", "均值填充", "中位数填充", "线性插值"], key='trace_mi')
            
            st.markdown("**数据转换**")
            do_log = st.checkbox("对数转换 (log10)", value=False)
            do_std = st.checkbox("Z-score标准化", value=False)
            
            # 新增：负数处理选项
            neg_method_t = st.selectbox("负数处理", ["替换为0", "替换为NaN", "取绝对值", "不处理"], key='trace_neg')
            
            if st.button(" 执行微量元素预处理", type="primary"):
                df = raw[selected_trace].copy()
                
                for col in df.columns:
                    if df[col].dtype == object:
                        mask = df[col].astype(str).str.contains('<|>|nd|ND|bdl|BDL', na=False)
                        if mask.any():
                            nums = df.loc[mask, col].astype(str).str.extract(r'([0-9.]+)')[0].astype(float)
                            if special_method_t == "替换为检出限的一半":
                                df.loc[mask, col] = nums / 2
                            elif special_method_t == "替换为0":
                                df.loc[mask, col] = 0
                            else:
                                df.loc[mask, col] = np.nan
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.astype(float)
                
                if missing_method_t == "删除含缺失值的行":
                    df = df.dropna()
                elif missing_method_t == "均值填充":
                    df = df.fillna(df.mean())
                elif missing_method_t == "中位数填充":
                    df = df.fillna(df.median())
                elif missing_method_t == "线性插值":
                    df = df.interpolate(method='linear').bfill().ffill()
                
                if do_log:
                    df = np.log10(df.replace(0, np.nan)).fillna(0)
                
                if do_std:
                    scaler = StandardScaler()
                    df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
                
                # 新增：处理负数
                neg_count = (df < 0).sum().sum()
                if neg_count > 0:
                    st.warning(f"检测到 {neg_count} 个负数")
                    if neg_method_t == "替换为0":
                        df = df.clip(lower=0)
                    elif neg_method_t == "替换为NaN":
                        df = df.where(df >= 0, np.nan)
                    elif neg_method_t == "取绝对值":
                        df = df.abs()
                    st.success(" 负数已处理")
                
                st.session_state.processed['trace'] = df
                st.success(f" 预处理完成: {df.shape[0]} 样本, {df.shape[1]} 元素")
                st.dataframe(df.head().round(3))
                st.dataframe(df.describe().round(3))
    
    with tab3:
        st.subheader("稀土元素处理")
        f = st.file_uploader("上传稀土元素数据", type=['xlsx', 'csv'], key='ree_upload')
        if f:
            raw = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
            raw = raw.set_index(raw.columns[0])
            st.session_state.dfs['ree'] = raw
            st.write(f"**原始数据**: {raw.shape[0]} 样本, {raw.shape[1]} 列")
            st.dataframe(raw.head())
            
            ree_elements = list(CHONDRITE.keys())
            detected_ree = [c for c in raw.columns if c in ree_elements]
            if detected_ree:
                st.success(f"检测到稀土元素: {detected_ree}")
            
            selected_ree = st.multiselect("选择稀土元素", raw.columns.tolist(), default=detected_ree if detected_ree else raw.columns.tolist())
            special_method_r = st.selectbox("特殊值处理", ["替换为检出限的一半", "替换为0", "替换为NaN"], key='ree_sp')
            missing_method_r = st.selectbox("缺失值处理", ["删除含缺失值的行", "均值填充", "线性插值"], key='ree_mi')
            
            st.markdown("**球粒陨石标准化**")
            do_chondrite = st.checkbox("球粒陨石标准化 (Sun & McDonough 1989)", value=True)
            if do_chondrite:
                st.dataframe(pd.DataFrame([CHONDRITE]).T.rename(columns={0: "标准值"}))
            
            if st.button(" 执行稀土元素预处理", type="primary"):
                df = raw[selected_ree].copy()
                
                for col in df.columns:
                    if df[col].dtype == object:
                        mask = df[col].astype(str).str.contains('<|>|nd|ND|bdl|BDL', na=False)
                        if mask.any():
                            nums = df.loc[mask, col].astype(str).str.extract(r'([0-9.]+)')[0].astype(float)
                            if special_method_r == "替换为检出限的一半":
                                df.loc[mask, col] = nums / 2
                            elif special_method_r == "替换为0":
                                df.loc[mask, col] = 0
                            else:
                                df.loc[mask, col] = np.nan
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.astype(float)
                
                if missing_method_r == "删除含缺失值的行":
                    df = df.dropna()
                elif missing_method_r == "均值填充":
                    df = df.fillna(df.mean())
                elif missing_method_r == "线性插值":
                    df = df.interpolate(method='linear').bfill().ffill()
                
                if do_chondrite:
                    for col in df.columns:
                        if col in CHONDRITE:
                            df[col] = df[col] / CHONDRITE[col]
                    df.columns = [f"{c}_N" if c in CHONDRITE else c for c in df.columns]
                
                st.session_state.processed['ree'] = df
                st.success(f" 预处理完成: {df.shape[0]} 样本, {df.shape[1]} 元素")
                st.dataframe(df.head().round(2))

elif nav == "2. 风化指标计算":
    st.header(" 风化指标与地球化学指标计算")
    df = st.session_state.processed['major']
    
    if df is None:
        st.warning("请先在「数据导入与预处理」中处理主量元素数据")
    else:
        st.dataframe(df.head())
        
        # 负数处理
        st.markdown("###  负数检测与处理")
        neg_count = (df < 0).sum().sum()
        if neg_count > 0:
            st.warning(f"检测到 {neg_count} 个负数值")
            neg_method = st.selectbox("负数处理方式", ["替换为0", "替换为NaN", "取绝对值", "不处理"])
            if st.button("处理负数"):
                if neg_method == "替换为0":
                    df = df.clip(lower=0)
                elif neg_method == "替换为NaN":
                    df = df.where(df >= 0, np.nan)
                elif neg_method == "取绝对值":
                    df = df.abs()
                st.session_state.processed['major'] = df
                st.success(f" 负数处理完成")
        else:
            st.success("未检测到负数")
        
        st.markdown("---")
        st.markdown("###  选择计算的指标")
        
        calc_options = st.multiselect("选择要计算的指标", [
            "风化指标 (CIA, CIW, PIA, WIP, CIX)",
            "古气候指标 (MAT, αAlNa, τNa)",
            "物源指标 (Zr/Ti, K2O/Al2O3)",
            "氧化还原指标 (V/Cr, U/Th, Uef, Moef)",
            "A-CN-K三角图数据",
            "矿物相关指标 (C/Q等)"
        ], default=["风化指标 (CIA, CIW, PIA, WIP, CIX)", "A-CN-K三角图数据"])
        
        if st.button(" 计算选中指标", type="primary"):
            calc = df.copy()
            
            # 摩尔转换函数
            def mol(ox):
                mw = {'Al2O3':101.96, 'CaO':56.08, 'Na2O':61.98, 'K2O':94.2, 'MgO':40.30, 'P2O5':141.94, 'SiO2':60.08, 'TiO2':79.87, 'Fe2O3':159.69, 'FeO':71.85, 'MnO':70.94}
                return calc[ox] / mw.get(ox, 1) if ox in calc.columns else pd.Series(0, index=calc.index)
            
            m_Al = mol('Al2O3')
            m_Ca = mol('CaO')
            m_Na = mol('Na2O')
            m_K = mol('K2O')
            m_Mg = mol('MgO')
            m_P = mol('P2O5')
            m_Si = mol('SiO2')
            
            # CaO* 校正（去除�iteite/磷�ite石中的CaO）
            m_Ca_star = np.minimum(np.maximum(m_Ca - 10/3 * m_P, 0), m_Na)
            calc['CaO*'] = m_Ca_star * 56.08  # 转回wt%
            
            # ========== 风化指标 ==========
            if "风化指标 (CIA, CIW, PIA, WIP, CIX)" in calc_options:
                # CIA: Chemical Index of Alteration (Nesbitt & Young, 1982)
                calc['CIA'] = m_Al / (m_Al + m_Ca_star + m_Na + m_K) * 100
                
                # CIW: Chemical Index of Weathering (Harnois, 1988)
                calc['CIW'] = m_Al / (m_Al + m_Ca_star + m_Na) * 100
                
                # PIA: Plagioclase Index of Alteration (Fedo et al., 1995)
                calc['PIA'] = (m_Al - m_K) / (m_Al + m_Ca_star + m_Na - m_K) * 100
                
                # WIP: Weathering Index of Parker (Parker, 1970)
                # WIP = 100 × [(2Na2O/0.35) + (MgO/0.9) + (2K2O/0.25) + (CaO/0.7)]
                if all(c in calc.columns for c in ['Na2O', 'MgO', 'K2O', 'CaO']):
                    calc['WIP'] = 100 * (2*calc['Na2O']/0.35 + calc['MgO']/0.9 + 2*calc['K2O']/0.25 + calc['CaO']/0.7)
                
                # CIX: Chemical Index of Weathering modified (Garzanti et al., 2014)
                # CIX = Al2O3/(Al2O3 + Na2O + K2O) × 100
                if all(c in calc.columns for c in ['Al2O3', 'Na2O', 'K2O']):
                    calc['CIX'] = calc['Al2O3'] / (calc['Al2O3'] + calc['Na2O'] + calc['K2O']) * 100
                
                st.success(" 风化指标计算完成: CIA, CIW, PIA, WIP, CIX")
            
            # ========== 古气候指标 ==========
            if "古气候指标 (MAT, αAlNa, τNa)" in calc_options:
                # MAT: Mean Annual Temperature (Sheldon et al., 2002)
                # MAT = -18.5 × (S/100) + 17.3, where S = CIA-K = Al2O3/(Al2O3+CaO*+Na2O)×100
                if 'CIA' not in calc.columns:
                    calc['CIA'] = m_Al / (m_Al + m_Ca_star + m_Na + m_K) * 100
                
                S = m_Al / (m_Al + m_Ca_star + m_Na) * 100  # CIA without K
                calc['MAT'] = -18.5 * (S / 100) + 17.3
                
                # αAlNa: Al-Na transfer coefficient
                # αAlNa = (Al/Na)sample / (Al/Na)UCC
                # UCC: Al2O3=15.4%, Na2O=3.27%
                if all(c in calc.columns for c in ['Al2O3', 'Na2O']):
                    al_na_ucc = 15.4 / 3.27
                    calc['αAlNa'] = (calc['Al2O3'] / calc['Na2O'].replace(0, np.nan)) / al_na_ucc
                
                # τNa: Mass transfer coefficient for Na
                # τNa = [(Na/Ti)sample / (Na/Ti)parent] - 1
                if all(c in calc.columns for c in ['Na2O', 'TiO2']):
                    na_ti_ucc = 3.27 / 0.64  # UCC as parent
                    calc['τNa'] = (calc['Na2O'] / calc['TiO2'].replace(0, np.nan)) / na_ti_ucc - 1
                
                st.success(" 古气候指标计算完成: MAT, αAlNa, τNa")
            
            # ========== 物源指标 ==========
            if "物源指标 (Zr/Ti, K2O/Al2O3)" in calc_options:
                if all(c in calc.columns for c in ['K2O', 'Al2O3']):
                    calc['K2O/Al2O3'] = calc['K2O'] / calc['Al2O3'].replace(0, np.nan)
                st.success(" 物源指标计算完成: K2O/Al2O3")
            
            # ========== 氧化还原指标 ==========
            if "氧化还原指标 (V/Cr, U/Th, Uef, Moef)" in calc_options:
                # 这些通常需要微量元素数据
                trace_df = st.session_state.processed.get('trace')
                if trace_df is not None:
                    # V/Cr
                    if 'V' in trace_df.columns and 'Cr' in trace_df.columns:
                        calc['V/Cr'] = trace_df['V'] / trace_df['Cr'].replace(0, np.nan)
                    
                    # U/Th
                    if 'U' in trace_df.columns and 'Th' in trace_df.columns:
                        calc['U/Th'] = trace_df['U'] / trace_df['Th'].replace(0, np.nan)
                    
                    # Uef (Authigenic U): Uef = Utotal - Th/3 (Wignall & Myers, 1988)
                    if 'U' in trace_df.columns and 'Th' in trace_df.columns:
                        calc['Uef'] = trace_df['U'] - trace_df['Th'] / 3
                    
                    # Moef (Authigenic Mo): Mo_ef = Mo_sample - (Mo/Al)PAAS × Al_sample
                    # PAAS Mo/Al = 0.13/9.97 (ppm/%)
                    if 'Mo' in trace_df.columns and 'Al' in trace_df.columns:
                        calc['Moef'] = trace_df['Mo'] - (0.13/9.97) * trace_df['Al']
                    elif 'Mo' in trace_df.columns and 'Al2O3' in calc.columns:
                        # Al2O3 wt% to Al ppm: Al = Al2O3 × 0.5293 × 10000
                        Al_ppm = calc['Al2O3'] * 0.5293 * 10000
                        calc['Moef'] = trace_df['Mo'] - (0.13/9.97) * Al_ppm / 10000
                    
                    st.success(" 氧化还原指标计算完成")
                else:
                    st.warning(" 氧化还原指标需要微量元素数据")
            
            # ========== A-CN-K 三角图数据 ==========
            if "A-CN-K三角图数据" in calc_options:
                total = m_Al + m_Ca_star + m_Na + m_K
                calc['A_norm'] = m_Al / total * 100
                calc['CN_norm'] = (m_Ca_star + m_Na) / total * 100
                calc['K_norm'] = m_K / total * 100
                st.success(" A-CN-K数据计算完成")
            
            # ========== 矿物相关指标 ==========
            if "矿物相关指标 (C/Q等)" in calc_options:
                # 这些需要矿物数据（XRD结果）
                # C/Q: Clay/Quartz ratio
                # 检查是否有矿物数据
                mineral_cols = ['Calcite', 'Dolomite', 'Quartz', 'Illite', 'Kaolinite', 'Chlorite', 'K-Feldspar', 'Plagioclase', 'Albite', 'Muscovite', 'Pyrite', 'Siderite']
                found_minerals = [c for c in mineral_cols if c in df.columns]
                
                if found_minerals:
                    st.info(f"检测到矿物数据: {found_minerals}")
                    
                    # C/Q: 粘土矿物/石英
                    clay_minerals = ['Illite', 'Kaolinite', 'Chlorite', 'Smectite', 'Montmorillonite']
                    clay_cols = [c for c in clay_minerals if c in df.columns]
                    if clay_cols and 'Quartz' in df.columns:
                        calc['C/Q'] = df[clay_cols].sum(axis=1) / df['Quartz'].replace(0, np.nan)
                    
                    # Calytol: 方解石+白云石 (碳酸盐总量)
                    carb_cols = [c for c in ['Calcite', 'Dolomite'] if c in df.columns]
                    if carb_cols:
                        calc['Calytol'] = df[carb_cols].sum(axis=1)
                    
                    st.success(" 矿物指标计算完成")
                else:
                    st.warning(" 未检测到矿物数据列")
            
            # 处理计算结果中的无穷值和负数
            calc = calc.replace([np.inf, -np.inf], np.nan)
            
            # 保存结果
            st.session_state.processed['major'] = calc
            
            # 显示结果
            st.markdown("###  计算结果统计")
            new_cols = [c for c in calc.columns if c not in df.columns]
            if new_cols:
                st.dataframe(calc[new_cols].describe().round(3))
            
            st.markdown("###  完整数据预览")
            st.dataframe(calc.head(10).round(3))
            
            # 指标解释
            with st.expander(" 指标说明"):
                st.markdown("""
**风化指标：**
- **CIA** (Chemical Index of Alteration): 化学蚀变指数，50-65弱风化，65-85中等风化，>85强风化
- **CIW** (Chemical Index of Weathering): 化学风化指数，不含K2O
- **PIA** (Plagioclase Index of Alteration): 斜长石蚀变指数
- **WIP** (Weathering Index of Parker): Parker风化指数，值越小风化越强
- **CIX**: 改进的化学风化指数

**古气候指标：**
- **MAT**: 年均温度估算 (°C)
- **αAlNa**: Al-Na迁移系数，>1表示Na淋失
- **τNa**: Na质量迁移系数，<0表示Na亏损

**氧化还原指标：**
- **V/Cr**: <2氧化环境，2-4.25次氧化-次还原，>4.25还原环境
- **U/Th**: >1.25还原环境
- **Uef**: 自生铀，>5ppm还原环境
- **Moef**: 自生钼，>25ppm强还原

**物源指标：**
- **K2O/Al2O3**: <0.2表示粘土矿物为主，>0.3表示长石为主
- **Zr/Ti**: 物源稳定性指标

**矿物指标：**
- **C/Q**: 粘土/石英比，反映风化程度
- **Calytol**: 碳酸盐总量
                """)
elif nav == "3. 风化指标图":
    st.header(" 风化指标图")
    df = st.session_state.processed['major']
    
    if df is None or 'CIA' not in df.columns:
        st.warning("请先计算风化指标")
    else:
        indices = [c for c in ['CIA', 'CIW', 'PIA'] if c in df.columns]
        selected = st.multiselect("选择指标", indices, default=indices)
        
        if selected:
            with st.expander(" 图表样式设置", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    fig_w = st.number_input("图表宽度", 400, 1600, 1000, 50)
                    single_h = st.number_input("单图高度", 100, 500, 200, 20)
                with c2:
                    line_w = st.slider("线条粗细", 1, 5, 2)
                    marker_s = st.slider("点大小", 4, 20, 8)
                with c3:
                    marker_symbol = st.selectbox("点形状", ["circle", "square", "diamond", "cross", "x", "triangle-up", "star"])
                    show_grid = st.checkbox("显示网格", value=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    y_min = st.number_input("Y轴最小值", 0, 100, 30)
                with c2:
                    y_max = st.number_input("Y轴最大值", 0, 100, 100)
                
                st.markdown("**自定义颜色**")
                custom_colors = []
                cols = st.columns(len(selected))
                for i, (col, idx) in enumerate(zip(cols, selected)):
                    with col:
                        custom_colors.append(st.color_picker(f"{idx}", COLORS[i]))
            
            samples = df.index.tolist()
            n = len(selected)
            
            fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.08)
            
            for i, idx in enumerate(selected):
                fig.add_trace(go.Scatter(
                    x=samples, y=df[idx], mode='lines+markers', name=idx,
                    line=dict(color=custom_colors[i], width=line_w),
                    marker=dict(size=marker_s, color=custom_colors[i], symbol=marker_symbol, line=dict(width=1, color='black'))
                ), row=i+1, col=1)
                
                fig.update_yaxes(title_text=idx, row=i+1, col=1, showline=True, linecolor='black', mirror=True, range=[y_min, y_max], showgrid=show_grid, gridcolor='#EEE')
            
            fig.update_xaxes(tickangle=45, row=n, col=1, showline=True, linecolor='black', mirror=True)
            fig.update_layout(width=fig_w, height=single_h*n+100, showlegend=False, plot_bgcolor='white', paper_bgcolor='white')
            
            st.plotly_chart(fig, use_container_width=False)
            desc = f"风化指标变化图,{','.join(selected)},样本数{len(samples)}"
            export_fig(fig, "weathering_indices", desc)

elif nav == "4. 二元图":
    st.header(" 二元图")
    df = st.session_state.processed['major']
    
    if df is None:
        st.warning("请先在「数据导入与预处理」中处理主量元素数据")
    else:
        exclude_cols = ['CIA', 'CIW', 'PIA', 'A_norm', 'CN_norm', 'K_norm']
        major_cols = [c for c in df.columns if c not in exclude_cols]
        
        if len(major_cols) < 2:
            st.warning("主量元素不足2个")
        else:
            c1, c2 = st.columns(2)
            with c1:
                x_elem = st.selectbox("X轴元素", major_cols, index=0)
            with c2:
                y_idx = min(1, len(major_cols)-1)
                y_elem = st.selectbox("Y轴元素", major_cols, index=y_idx)
            
            with st.expander(" 图表样式设置", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    fig_size = st.number_input("图大小", 400, 1200, 700, 50)
                    marker_s = st.slider("点大小", 5, 30, 12)
                with c2:
                    pt_color = st.color_picker("点颜色", "#E74C3C")
                    marker_symbol = st.selectbox("点形状", ["circle", "square", "diamond", "cross", "x", "triangle-up", "star"], key="binary_marker")
                with c3:
                    marker_opacity = st.slider("透明度", 0.1, 1.0, 0.8, 0.1)
                    border_width = st.slider("边框粗细", 0, 3, 1)
                
                st.markdown("**坐标轴设置**")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    x_min = st.number_input("X轴最小", value=0.0, step=1.0)
                with c2:
                    x_max_default = float(df[x_elem].max() * 1.1)
                    x_max = st.number_input("X轴最大", value=x_max_default, step=1.0)
                with c3:
                    y_min = st.number_input("Y轴最小", value=0.0, step=1.0)
                with c4:
                    y_max_default = float(df[y_elem].max() * 1.1)
                    y_max = st.number_input("Y轴最大", value=y_max_default, step=1.0)
                
                c1, c2 = st.columns(2)
                with c1:
                    x_unit = st.text_input("X轴单位", "wt%")
                    x_title = st.text_input("X轴标题", f"{x_elem} ({x_unit})")
                with c2:
                    y_unit = st.text_input("Y轴单位", "wt%")
                    y_title = st.text_input("Y轴标题", f"{y_elem} ({y_unit})")
                
                show_regression = st.checkbox("显示回归线", value=False)
                equal_axis = st.checkbox("等比例坐标轴", value=False)
            
            x_data = df[x_elem].values
            y_data = df[y_elem].values
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data, mode='markers', showlegend=False,
                marker=dict(size=marker_s, color=pt_color, opacity=marker_opacity, symbol=marker_symbol, line=dict(width=border_width, color='black')),
                text=df.index,
                hovertemplate=f"<b>%{{text}}</b><br>{x_elem}: %{{x:.2f}}<br>{y_elem}: %{{y:.2f}}<extra></extra>"
            ))
            
            if show_regression:
                mask = ~(np.isnan(x_data) | np.isnan(y_data))
                if mask.sum() > 2:
                    z = np.polyfit(x_data[mask], y_data[mask], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x_min, x_max, 100)
                    fig.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', line=dict(color='gray', dash='dash', width=2), showlegend=False, name='回归线'))
                    corr = np.corrcoef(x_data[mask], y_data[mask])[0, 1]
                    st.info(f"相关系数 r = {corr:.3f}, 回归方程: y = {z[0]:.3f}x + {z[1]:.3f}")
            
            layout_opts = dict(
                width=fig_size, height=fig_size,
                xaxis=dict(title=x_title, range=[x_min, x_max], showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=True, gridcolor='#EEE', zeroline=False, dtick=(x_max-x_min)/5),
                yaxis=dict(title=y_title, range=[y_min, y_max], showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=True, gridcolor='#EEE', zeroline=False, dtick=(y_max-y_min)/5),
                plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=80, r=40, t=40, b=80)
            )
            
            if equal_axis:
                layout_opts['yaxis']['scaleanchor'] = 'x'
                layout_opts['yaxis']['scaleratio'] = 1
            
            fig.update_layout(**layout_opts)
            
            st.plotly_chart(fig, use_container_width=False)
            desc = f"二元图,{x_elem} vs {y_elem},样本数{len(df)}"
            if show_regression:
                desc += f",r={corr:.3f}"
            export_fig(fig, f"binary_{x_elem}_{y_elem}", desc)

elif nav == "5. 三角图":
    st.header(" A-CN-K 三角图")
    df = st.session_state.processed['major']
    
    if df is None or 'A_norm' not in df.columns:
        st.warning("请先计算风化指标")
    else:
        with st.expander(" 图表样式设置", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                fig_size = st.number_input("图大小", 400, 1200, 700, 50)
                marker_s = st.slider("点大小", 5, 25, 12)
            with c2:
                pt_color = st.color_picker("点颜色", "#E74C3C")
                marker_symbol = st.selectbox("点形状", ["circle", "square", "diamond", "triangle-up", "star"], key="ternary_marker")
            with c3:
                marker_opacity = st.slider("透明度", 0.1, 1.0, 0.8, 0.1, key="ternary_opacity")
                border_width = st.slider("边框粗细", 0, 3, 1, key="ternary_border")
            
            c1, c2 = st.columns(2)
            with c1:
                show_ref = st.checkbox("显示参考矿物", value=True)
                ref_color = st.color_picker("参考矿物颜色", "#FF0000") if show_ref else "#FF0000"
            with c2:
                show_cia = st.checkbox("显示CIA等值线", value=True)
                cia_values = st.multiselect("CIA等值线值", [50, 60, 70, 80, 90], default=[50, 60, 70, 80, 90]) if show_cia else []
            
            st.markdown("**坐标轴标题**")
            c1, c2, c3 = st.columns(3)
            with c1:
                a_title = st.text_input("A顶点", "A (Al₂O₃)")
            with c2:
                cn_title = st.text_input("CN顶点", "CN (CaO*+Na₂O)")
            with c3:
                k_title = st.text_input("K顶点", "K (K₂O)")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterternary(
            a=df['A_norm'], b=df['CN_norm'], c=df['K_norm'],
            mode='markers', showlegend=False,
            marker=dict(size=marker_s, color=pt_color, opacity=marker_opacity, symbol=marker_symbol, line=dict(width=border_width, color='black')),
            text=df.index,
            hovertemplate="<b>%{text}</b><br>A: %{a:.1f}<br>CN: %{b:.1f}<br>K: %{c:.1f}<extra></extra>"
        ))
        
        if show_ref:
            refs = {'高岭石/绿泥石': (100, 0, 0), '伊利石': (75, 0, 25), '钾长石': (35, 0, 65), '斜长石': (50, 50, 0), '蒙脱石': (90, 5, 5)}
            for name, (a, b, c) in refs.items():
                fig.add_trace(go.Scatterternary(
                    a=[a], b=[b], c=[c], mode='markers+text', text=[name], textposition='top center',
                    marker=dict(size=14, symbol='diamond', color=ref_color, line=dict(width=2, color='black')),
                    showlegend=False, textfont=dict(size=10, color=ref_color)
                ))
        
        if show_cia and cia_values:
            for cia in cia_values:
                fig.add_trace(go.Scatterternary(
                    a=[cia, cia], b=[100-cia, 0], c=[0, 100-cia],
                    mode='lines', line=dict(color='gray', width=1, dash='dash'), showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatterternary(
                    a=[cia], b=[(100-cia)/2], c=[(100-cia)/2], mode='text', text=[f"CIA={cia}"],
                    textposition='middle center', textfont=dict(size=8, color='gray'), showlegend=False
                ))
        
        fig.update_layout(
            width=fig_size, height=fig_size,
            ternary=dict(
                sum=100,
                aaxis=dict(title=a_title, linewidth=2, linecolor='black', gridcolor='lightgray', ticksuffix='%'),
                baxis=dict(title=cn_title, linewidth=2, linecolor='black', gridcolor='lightgray', ticksuffix='%'),
                caxis=dict(title=k_title, linewidth=2, linecolor='black', gridcolor='lightgray', ticksuffix='%'),
                bgcolor='white'
            ),
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=False)
        desc = f"A-CN-K三角图,样本数{len(df)},CIA范围{df['CIA'].min():.1f}-{df['CIA'].max():.1f}"
        export_fig(fig, "ternary_ACNK", desc)

elif nav == "6. PCA双标图":
    st.header(" PCA 双标图")
    df = get_combined_data()
    
    if df is None:
        st.warning("请先处理数据")
    else:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        exclude = ['CIA', 'CIW', 'PIA', 'A_norm', 'CN_norm', 'K_norm']
        default = [c for c in num_cols if c not in exclude][:10]
        
        features = st.multiselect("选择变量", num_cols, default=default)
        
        if len(features) >= 3:
            X = df[features].dropna()
            
            if len(X) < 3:
                st.warning("有效样本不足3个")
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                pca = PCA(n_components=2)
                scores = pca.fit_transform(X_scaled)
                loadings = pca.components_.T
                var_exp = pca.explained_variance_ratio_
                
                with st.expander("图表样式设置", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        fig_size = st.number_input("图大小", 400, 1200, 800, 50)
                        marker_s = st.slider("样本点大小", 5, 25, 10)
                    with c2:
                        sample_color = st.color_picker("样本点颜色", "#3498DB")
                        marker_symbol = st.selectbox("样本点形状", ["circle", "square", "diamond", "triangle-up"], key="pca_marker")
                    with c3:
                        arrow_scale = st.slider("箭头长度系数", 1.0, 15.0, 5.0, 0.5)
                        arrow_width = st.slider("箭头粗细", 1, 5, 2)
                    
                    st.markdown("**坐标轴范围设置**")
                    auto_range = st.checkbox("自动范围", value=True)
                    if not auto_range:
                        c1, c2 = st.columns(2)
                        with c1:
                            axis_min = st.number_input("坐标轴最小值", value=-5.0, step=0.5)
                        with c2:
                            axis_max = st.number_input("坐标轴最大值", value=5.0, step=0.5)
                    
                    st.markdown("**箭头颜色设置（为每个变量设置颜色）**")
                    arrow_colors = {}
                    n_cols = min(5, len(features))
                    rows = (len(features) + n_cols - 1) // n_cols
                    for row_i in range(rows):
                        cols = st.columns(n_cols)
                        for col_i, col in enumerate(cols):
                            feat_idx = row_i * n_cols + col_i
                            if feat_idx < len(features):
                                feat = features[feat_idx]
                                with col:
                                    arrow_colors[feat] = st.color_picker(feat, COLORS[feat_idx % len(COLORS)], key=f"arrow_{feat}")
                
                fig = go.Figure()
                
                # 样本点
                fig.add_trace(go.Scatter(
                    x=scores[:, 0], y=scores[:, 1], mode='markers', name='样本',
                    marker=dict(size=marker_s, color=sample_color, symbol=marker_symbol, line=dict(width=1, color='black')),
                    text=X.index, hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
                ))
                
                # 箭头
                for i, feat in enumerate(features):
                    x_end = loadings[i, 0] * arrow_scale
                    y_end = loadings[i, 1] * arrow_scale
                    color = arrow_colors.get(feat, COLORS[i % len(COLORS)])
                    
                    fig.add_annotation(
                        x=x_end, y=y_end, ax=0, ay=0, xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=arrow_width, arrowcolor=color
                    )
                    fig.add_annotation(
                        x=x_end * 1.15, y=y_end * 1.15, text=f"<b>{feat}</b>", showarrow=False,
                        font=dict(size=11, color=color)
                    )
                
                if auto_range:
                    max_score = np.abs(scores).max()
                    max_loading = np.abs(loadings).max() * arrow_scale
                    axis_lim = max(max_score, max_loading) * 1.3
                    axis_min, axis_max = -axis_lim, axis_lim
                
                fig.update_layout(
                    width=fig_size, height=fig_size,
                    xaxis=dict(title=f"PC1 ({var_exp[0]:.1%})", range=[axis_min, axis_max], showline=True, linewidth=2, linecolor='black', mirror=True, zeroline=True, zerolinecolor='lightgray', zerolinewidth=1, showgrid=True, gridcolor='#EEE'),
                    yaxis=dict(title=f"PC2 ({var_exp[1]:.1%})", range=[axis_min, axis_max], showline=True, linewidth=2, linecolor='black', mirror=True, zeroline=True, zerolinecolor='lightgray', zerolinewidth=1, showgrid=True, gridcolor='#EEE', scaleanchor='x', scaleratio=1),
                    plot_bgcolor='white', paper_bgcolor='white', showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=False)
                desc = f"PCA双标图,PC1解释{var_exp[0]:.1%},PC2解释{var_exp[1]:.1%},总计{sum(var_exp):.1%}"
                export_fig(fig, "pca_biplot", desc)
                
                st.markdown("### 载荷矩阵")
                loadings_df = pd.DataFrame(loadings, index=features, columns=['PC1', 'PC2'])
                st.dataframe(loadings_df.round(3))
                
                st.markdown("### 方差解释")
                st.write(f"- PC1: {var_exp[0]:.1%}")
                st.write(f"- PC2: {var_exp[1]:.1%}")
                st.write(f"- 累计: {sum(var_exp):.1%}")

elif nav == "7. 聚类分析":
    st.header(" 聚类分析")
    df = get_combined_data()
    
    if df is None:
        st.warning("请先处理数据")
    else:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        exclude = ['CIA', 'CIW', 'PIA', 'A_norm', 'CN_norm', 'K_norm']
        default = [c for c in num_cols if c not in exclude][:8]
        
        features = st.multiselect("选择特征变量", num_cols, default=default)
        
        if len(features) >= 2:
            X = df[features].dropna()
            X_scaled = StandardScaler().fit_transform(X)
            
            st.success(f" {len(X)} 样本, {len(features)} 特征")
            
            method = st.selectbox("聚类方法", ["K-Means", "层次聚类", "DBSCAN"])
            
            if method == "K-Means":
                st.markdown("###  最优K值搜索")
                c1, c2 = st.columns(2)
                with c1:
                    k_min = st.number_input("K最小值", 2, 10, 2)
                with c2:
                    k_max = st.number_input("K最大值", 3, 20, 10)
                
                if st.button(" 生成调参报告"):
                    ks = list(range(k_min, k_max + 1))
                    sil_scores, cal_scores, db_scores, inertias = [], [], [], []
                    
                    progress = st.progress(0)
                    for i, k in enumerate(ks):
                        km = KMeans(n_clusters=k, n_init=10, random_state=42)
                        labels = km.fit_predict(X_scaled)
                        sil_scores.append(silhouette_score(X_scaled, labels))
                        cal_scores.append(calinski_harabasz_score(X_scaled, labels))
                        db_scores.append(davies_bouldin_score(X_scaled, labels))
                        inertias.append(km.inertia_)
                        progress.progress((i + 1) / len(ks))
                    
                    fig = make_subplots(rows=2, cols=2, subplot_titles=['肘部法则 (Inertia)', '轮廓系数 (越大越好)', 'Calinski-Harabasz (越大越好)', 'Davies-Bouldin (越小越好)'])
                    
                    fig.add_trace(go.Scatter(x=ks, y=inertias, mode='lines+markers', marker=dict(color=COLORS[0], size=10), line=dict(width=2)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=ks, y=sil_scores, mode='lines+markers', marker=dict(color=COLORS[1], size=10), line=dict(width=2)), row=1, col=2)
                    fig.add_trace(go.Scatter(x=ks, y=cal_scores, mode='lines+markers', marker=dict(color=COLORS[2], size=10), line=dict(width=2)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=ks, y=db_scores, mode='lines+markers', marker=dict(color=COLORS[3], size=10), line=dict(width=2)), row=2, col=2)
                    
                    best_k_sil = ks[np.argmax(sil_scores)]
                    fig.add_vline(x=best_k_sil, line_dash="dash", line_color="red", row=1, col=2)
                    
                    fig.update_layout(width=900, height=700, showlegend=False, plot_bgcolor='white')
                    fig.update_xaxes(title_text="K", showline=True, linecolor='black')
                    fig.update_yaxes(showline=True, linecolor='black')
                    
                    st.plotly_chart(fig)
                    export_fig(fig, "kmeans_tuning", f"K-Means调参,推荐K={best_k_sil}")
                    
                    st.success(f" 推荐 K = {best_k_sil} (轮廓系数最优: {max(sil_scores):.3f})")
                    st.session_state['best_k'] = best_k_sil
                
                st.markdown("---")
                st.markdown("###  执行聚类")
                
                final_k = st.number_input("聚类数 K", 2, 20, st.session_state.get('best_k', 3))
                
                with st.expander(" 图表样式设置"):
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_size_cluster = st.number_input("图大小", 400, 1200, 700, 50, key="cluster_size")
                        marker_s_cluster = st.slider("点大小", 5, 25, 12, key="cluster_marker")
                    with c2:
                        show_centers = st.checkbox("显示聚类中心", value=True)
                        marker_symbol_cluster = st.selectbox("点形状", ["circle", "square", "diamond", "triangle-up"], key="cluster_symbol")
                    
                    st.markdown("**聚类颜色**")
                    cluster_colors = []
                    cols = st.columns(min(final_k, 7))
                    for i in range(final_k):
                        with cols[i % len(cols)]:
                            cluster_colors.append(st.color_picker(f"类{i+1}", COLORS[i % len(COLORS)], key=f"cluster_color_{i}"))
                
                if st.button(" 执行K-Means聚类", type="primary"):
                    km = KMeans(n_clusters=final_k, n_init=10, random_state=42)
                    labels = km.fit_predict(X_scaled)
                    
                    st.session_state['cluster_labels'] = labels
                    st.session_state['cluster_index'] = X.index
                    st.session_state['n_clusters'] = final_k
                    
                    pca = PCA(n_components=2)
                    scores = pca.fit_transform(X_scaled)
                    var_exp = pca.explained_variance_ratio_
                    
                    fig = go.Figure()
                    for c in range(final_k):
                        mask = labels == c
                        fig.add_trace(go.Scatter(
                            x=scores[mask, 0], y=scores[mask, 1], mode='markers', name=f'Cluster {c+1} (n={mask.sum()})',
                            marker=dict(size=marker_s_cluster, color=cluster_colors[c], symbol=marker_symbol_cluster, line=dict(width=1, color='black')),
                            text=X.index[mask], hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
                        ))
                    
                    if show_centers:
                        centers_pca = pca.transform(km.cluster_centers_)
                        fig.add_trace(go.Scatter(
                            x=centers_pca[:, 0], y=centers_pca[:, 1], mode='markers', name='聚类中心',
                            marker=dict(size=18, symbol='x', color='black', line=dict(width=2))))
                    
                    sil = silhouette_score(X_scaled, labels)
                    cal = calinski_harabasz_score(X_scaled, labels)
                    db = davies_bouldin_score(X_scaled, labels)
                    
                    axis_lim = np.abs(scores).max() * 1.2
                    fig.update_layout(width=fig_size_cluster, height=fig_size_cluster,
                        xaxis=dict(title=f"PC1 ({var_exp[0]:.1%})", range=[-axis_lim, axis_lim], showline=True, linecolor='black', mirror=True, showgrid=True, gridcolor='#EEE', zeroline=True, zerolinecolor='lightgray'),
                        yaxis=dict(title=f"PC2 ({var_exp[1]:.1%})", range=[-axis_lim, axis_lim], showline=True, linecolor='black', mirror=True, showgrid=True, gridcolor='#EEE', zeroline=True, zerolinecolor='lightgray', scaleanchor='x'),
                        plot_bgcolor='white', legend=dict(x=1.02, y=1))
                    
                    st.plotly_chart(fig)
                    st.session_state.figs['cluster_kmeans'] = fig
                    st.session_state.fig_desc['cluster_kmeans'] = f"K-Means聚类,K={final_k},轮廓系数={sil:.3f}"
                    export_fig(fig, "cluster_kmeans", f"K={final_k},Sil={sil:.3f}")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("轮廓系数", f"{sil:.3f}")
                    c2.metric("Calinski-Harabasz", f"{cal:.1f}")
                    c3.metric("Davies-Bouldin", f"{db:.3f}")
                    
                    st.markdown("### 各聚类特征均值")
                    df_result = X.copy()
                    df_result['Cluster'] = [f"C{x+1}" for x in labels]
                    st.dataframe(df_result.groupby('Cluster')[features].mean().round(3))
            
            elif method == "层次聚类":
                linkage_method = st.selectbox("连接方法", ['ward', 'complete', 'average', 'single'])
                
                if st.button("生成树状图"):
                    Z = linkage(X_scaled, method=linkage_method)
                    dendro = scipy_dendrogram(Z, labels=X.index.tolist(), no_plot=True)
                    
                    fig = go.Figure()
                    for i in range(len(dendro['icoord'])):
                        fig.add_trace(go.Scatter(x=dendro['icoord'][i], y=dendro['dcoord'][i], mode='lines', line=dict(color='#34495E', width=1.5), showlegend=False))
                    
                    fig.update_layout(width=max(900, len(X)*15), height=500,
                        xaxis=dict(ticktext=dendro['ivl'], tickvals=list(range(5, len(dendro['ivl'])*10, 10)), tickangle=45, showline=True, linecolor='black'),
                        yaxis=dict(title='距离', showline=True, linecolor='black'), plot_bgcolor='white')
                    st.plotly_chart(fig)
                    st.session_state.figs['dendrogram'] = fig
                    st.session_state.fig_desc['dendrogram'] = f"层次聚类树状图,方法={linkage_method}"
                    export_fig(fig, "dendrogram", f"层次聚类,{linkage_method}")
                
                n_clusters = st.number_input("切割聚类数", 2, 15, 3)
                if st.button("执行层次聚类", type="primary"):
                    hier = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
                    labels = hier.fit_predict(X_scaled)
                    st.session_state['cluster_labels'] = labels
                    st.session_state['cluster_index'] = X.index
                    st.session_state['n_clusters'] = n_clusters
                    
                    pca = PCA(n_components=2)
                    scores = pca.fit_transform(X_scaled)
                    fig = go.Figure()
                    for c in range(n_clusters):
                        mask = labels == c
                        fig.add_trace(go.Scatter(x=scores[mask, 0], y=scores[mask, 1], mode='markers', name=f'Cluster {c+1}',
                            marker=dict(size=12, color=COLORS[c % len(COLORS)], line=dict(width=1, color='black'))))
                    fig.update_layout(width=700, height=700, plot_bgcolor='white')
                    st.plotly_chart(fig)
                    st.session_state.figs['cluster_hier'] = fig
                    st.metric("轮廓系数", f"{silhouette_score(X_scaled, labels):.3f}")
            
            elif method == "DBSCAN":
                c1, c2 = st.columns(2)
                with c1:
                    eps = st.slider("eps (邻域半径)", 0.1, 5.0, 0.5, 0.1)
                with c2:
                    min_samples = st.slider("min_samples", 2, 20, 5)
                
                if st.button("执行DBSCAN", type="primary"):
                    db = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = db.fit_predict(X_scaled)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    c1, c2 = st.columns(2)
                    c1.metric("聚类数", n_clusters)
                    c2.metric("噪声点", n_noise)
                    
                    if n_clusters > 0:
                        st.session_state['cluster_labels'] = labels
                        st.session_state['cluster_index'] = X.index
                        st.session_state['n_clusters'] = n_clusters
                        
                        pca = PCA(n_components=2)
                        scores = pca.fit_transform(X_scaled)
                        fig = go.Figure()
                        for label in sorted(set(labels)):
                            mask = labels == label
                            name = '噪声' if label == -1 else f'Cluster {label+1}'
                            color = 'lightgray' if label == -1 else COLORS[label % len(COLORS)]
                            symbol = 'x' if label == -1 else 'circle'
                            fig.add_trace(go.Scatter(x=scores[mask, 0], y=scores[mask, 1], mode='markers', name=name,
                                marker=dict(size=10 if label != -1 else 6, color=color, symbol=symbol)))
                        fig.update_layout(width=700, height=700, plot_bgcolor='white')
                        st.plotly_chart(fig)
                        st.session_state.figs['cluster_dbscan'] = fig
                        if n_clusters > 1:
                            valid_mask = labels != -1
                            st.metric("轮廓系数", f"{silhouette_score(X_scaled[valid_mask], labels[valid_mask]):.3f}")

elif nav == "8. 物源分类":
    st.header(" 物源分类")
    df = get_combined_data()
    if df is None:
        st.warning("请先处理数据")
    else:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        obj_cols = df.select_dtypes(include='object').columns.tolist()
        exclude = ['CIA','CIW','PIA','A_norm','CN_norm','K_norm']
        feature_cols = [c for c in num_cols if c not in exclude]
        
        real_categories = [c for c in obj_cols if df[c].nunique() < 20 and df[c].nunique() > 1]
        
        st.markdown("###  分类标签来源")
        
        if 'cluster_labels' in st.session_state:
            st.success(f" 检测到聚类结果: {st.session_state['n_clusters']} 类")
        
        if real_categories:
            st.success(f" 检测到分类变量: {real_categories}")
        
        label_source = st.radio("选择标签来源", ["使用聚类结果", "使用已有分类变量"] if 'cluster_labels' in st.session_state and real_categories else (["使用聚类结果"] if 'cluster_labels' in st.session_state else ["使用已有分类变量"] if real_categories else []))
        
        if not label_source:
            st.warning("请先进行聚类分析或确保数据中有分类变量")
        else:
            if label_source == "使用聚类结果":
                labels = st.session_state['cluster_labels']
                idx = st.session_state['cluster_index']
                df_work = df.loc[idx].copy()
                df_work['Label'] = [f"C{x+1}" for x in labels]
                target = 'Label'
            else:
                target = st.selectbox("目标变量", real_categories)
                df_work = df.dropna(subset=[target]).copy()
            
            st.write(f"**类别分布:** {df_work[target].value_counts().to_dict()}")
            
            features = st.multiselect("特征变量", feature_cols, default=feature_cols[:8] if len(feature_cols) >= 8 else feature_cols)
            
            if len(features) >= 2:
                X = df_work[features].dropna()
                y = df_work.loc[X.index, target].astype(str)
                run_classification(X, y, features, api_key)

elif nav == "9. AI智能分析":
    st.header(" AI智能分析")
    df = get_combined_data()
    
    if df is None:
        st.warning("请先处理数据")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("样本数", len(df))
        c2.metric("变量数", len(df.columns))
        major_cnt = len([c for c in df.columns if any(m in str(c) for m in MAJOR_ELEMENTS)])
        c3.metric("主量元素", major_cnt)
        c4.metric("缺失值", df.isnull().sum().sum())
        
        if st.session_state.figs:
            st.markdown("###  已生成图件")
            st.info(f"可分析: {', '.join(st.session_state.figs.keys())}")
        
        analysis_type = st.selectbox("分析类型", ["数据质量评估", "地球化学特征分析", "风化程度解释", "聚类结果解释", "物源判别解释", "图件解读分析", "自定义分析"])
        
        if analysis_type == "图件解读分析":
            if not st.session_state.figs:
                st.warning("暂无图件，请先生成图表")
            else:
                sel_fig = st.selectbox("选择图件", list(st.session_state.figs.keys()))
                if sel_fig:
                    st.plotly_chart(st.session_state.figs[sel_fig])
                    fig_info = st.session_state.fig_desc.get(sel_fig, "")
                    
                    prompts_map = {
                        'binary': "分析二元图:元素相关性、数据分布、异常点、物源指示",
                        'ternary': "分析A-CN-K三角图:风化程度、矿物关系、古气候指示",
                        'pca': "分析PCA双标图:主成分意义、样品分组、载荷解释",
                        'cluster': "分析聚类结果:各聚类特征、地质意义、岩性推断",
                        'weathering': "分析风化指标:风化程度、变化趋势、古气候演化",
                        'cv': "分析交叉验证:模型稳定性、过拟合风险",
                        'confusion': "分析混淆矩阵:分类准确性、误分类原因",
                        'feature': "分析特征重要性:关键判别元素、地质意义",
                        'dendrogram': "分析树状图:聚类层次、切割建议"
                    }
                    auto_prompt = "请分析图件特征"
                    for k, v in prompts_map.items():
                        if k in sel_fig.lower():
                            auto_prompt = v
                            break
                    
                    custom_fig_prompt = st.text_area("分析要点", auto_prompt, height=80)
        
        elif analysis_type == "自定义分析":
            custom_prompt = st.text_area("输入分析需求", placeholder="请分析稀土元素配分模式的地质意义...", height=120)
            include_data = st.multiselect("包含数据", ["基本统计", "主量元素", "风化指标", "相关性矩阵"], default=["基本统计"])
        
        with st.expander(" 高级设置"):
            c1, c2 = st.columns(2)
            with c1:
                temperature = st.slider("创造性", 0.0, 1.0, 0.3, 0.1)
                max_tokens = st.slider("最大字数", 500, 3000, 1500, 100)
            with c2:
                language = st.selectbox("语言", ["中文", "English"])
                detail = st.selectbox("详细程度", ["简要", "标准", "详细"])
        
        if st.button(" 开始AI分析", type="primary"):
            if not api_key:
                st.error("请输入API Key")
            else:
                with st.spinner("分析中..."):
                    data_ctx = f"样本数:{len(df)}, 变量数:{len(df.columns)}\n"
                    
                    major_cols = [c for c in df.columns if any(m in str(c) for m in MAJOR_ELEMENTS)]
                    if major_cols:
                        data_ctx += f"\n主量元素统计:\n{df[major_cols].describe().round(2).to_string()}\n"
                    
                    weather_cols = [c for c in ['CIA','CIW','PIA'] if c in df.columns]
                    if weather_cols:
                        data_ctx += f"\n风化指标:\n{df[weather_cols].describe().round(2).to_string()}\n"
                    
                    if analysis_type == "图件解读分析" and sel_fig:
                        final_prompt = f"""作为地球化学专家，分析以下图件:

图件: {sel_fig}
描述: {st.session_state.fig_desc.get(sel_fig, '')}
分析要点: {custom_fig_prompt}

数据背景:
{data_ctx}

要求: {language}, {detail}程度分析
请给出专业的地球化学解释，包括地质意义、成因分析和科学建议。"""
                    
                    elif analysis_type == "自定义分析":
                        final_prompt = f"""作为地球化学专家:

数据:
{data_ctx}

分析需求: {custom_prompt}

要求: {language}, {detail}"""
                    
                    else:
                        templates = {
                            "数据质量评估": f"评估数据质量:\n{data_ctx}\n分析样本量、数据合理性、缺失值影响、预处理建议",
                            "地球化学特征分析": f"分析地球化学特征:\n{data_ctx}\n分析元素组成、岩性特征、构造环境",
                            "风化程度解释": f"解释风化程度:\n{data_ctx}\n分析CIA/CIW/PIA指标、古气候条件",
                            "聚类结果解释": f"解释聚类结果:\n{data_ctx}\n聚类数:{st.session_state.get('n_clusters','未知')}\n分析各聚类特征差异、地质意义",
                            "物源判别解释": f"解释物源判别:\n{data_ctx}\n分析物源区特征、判别元素意义"
                        }
                        final_prompt = templates.get(analysis_type, f"分析:\n{data_ctx}") + f"\n\n要求: {language}, {detail}"
                    
                    try:
                        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                        resp = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[
                                {"role": "system", "content": "你是顶级地球化学专家，擅长沉积地球化学、物源分析、古环境重建。"},
                                {"role": "user", "content": final_prompt}
                            ],
                            temperature=temperature, max_tokens=max_tokens
                        )
                        result = resp.choices[0].message.content
                        
                        st.markdown("---")
                        st.markdown("##  分析结果")
                        st.markdown(result)
                        
                        st.markdown("---")
                        full_report = f"# {analysis_type}\n\n## 数据\n{data_ctx}\n\n## 分析\n{result}\n\n---\n生成时间: {pd.Timestamp.now()}"
                        c1, c2 = st.columns(2)
                        c1.download_button("📄 下载MD", full_report, f"{analysis_type}.md")
                        c2.download_button("📄 下载TXT", full_report, f"{analysis_type}.txt")
                        
                        st.markdown("### 💬 追问")
                        follow = st.text_input("继续提问:")
                        if follow and st.button("发送"):
                            follow_resp = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[{"role": "user", "content": f"基于:{result}\n\n追问:{follow}"}],
                                temperature=temperature, max_tokens=max_tokens
                            )
                            st.markdown(follow_resp.choices[0].message.content)
                    
                    except Exception as e:
                        st.error(f"失败: {e}")
