import streamlit as st  
import pandas as pd  
import numpy as np  
import joblib  
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化 session_state 中的 data
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=['Gender','Age','CA15-3','CEA','CYFRA21-1','NSE','ALB','HDL','TG','Urea','Expectoration','Hemoptysis','Distress','Fever','Prediction', 'Label'])

# 设置页面为宽模式
st.set_page_config(layout="wide")

st.sidebar.image("hospital_logo2.png", caption="", width=300)
# Language setting  
lang = st.sidebar.selectbox('Choose language', ['中文', 'English'])  

# Footer  
if lang == '中文':  
    st.sidebar.subheader('程序说明')  
    st.sidebar.write("<p style='font-size: 12px;'>申明: 这款小程序旨在提供一般信息，不能替代专业医疗建议或诊断。如果您对自己的健康有任何担忧，请务必咨询合格的医疗保健专业人员。</p>", unsafe_allow_html=True)
else:  
    st.sidebar.subheader('Program Description')  
    st.sidebar.write("<p style='font-size: 12px;'>Disclaimer: This mini app is designed to provide general information and is not a substitute for professional medical advice or diagnosis. Always consult with a qualified healthcare professional if you have any concerns about your health.</p>", unsafe_allow_html=True)

st.image("hospital_logo.png", caption="")
if lang == '中文':  
    st.header("肺结节恶性概率")
else:  
    st.header("Malignant probability of lung nodules")

st.write("LightGBM Model")

# 将输入分成两列，每列7个
col1, col2 = st.columns(2)

gender_mapping = {
    '中文': {'男': 1, '女': 2},
    'English': {'Male': 1, 'Female': 2}
}

with col1:
    if lang == '中文':
        options = list(gender_mapping['中文'].keys())
        a = st.selectbox("性别", options, index=0)
        a_val = gender_mapping['中文'][a]
        
        b_col, label_col = st.columns([8,2])
        b = b_col.number_input("年龄", min_value=0, max_value=100, value=62)
        label_col.markdown("岁")
        
        c_col, c_label_col = st.columns([8,2])
        c = c_col.number_input("肿瘤相关抗原15-3", min_value=0.00, max_value=2000.00, value=12.50)
        c_label_col.markdown("U/mL")
        
        d_col, d_label_col = st.columns([8,2])
        d = d_col.number_input("癌胚抗原", min_value=0.00, max_value=10000.00, value=2.49)
        d_label_col.markdown("ng/mL")
        
        e_col, e_label_col = st.columns([8,2])
        e = e_col.number_input("细胞角蛋白19片段", min_value=0.00, max_value=200.00, value=2.74)
        e_label_col.markdown("ng/mL")
        
        f_col, f_label_col = st.columns([8,2])
        f = f_col.number_input("神经元特异性烯醇化酶", min_value=0.00, max_value=500.00, value=16.60)
        f_label_col.markdown("ng/mL")
        
        g_col, g_label_col = st.columns([8,2])
        g = g_col.number_input("白蛋白", min_value=0.0, max_value=100.0, value=39.7)
        g_label_col.markdown("g/L")
    else:
        options = list(gender_mapping['English'].keys())
        a = st.selectbox("Gender", options, index=0)
        a_val = gender_mapping['English'][a]
        
        b_col, label_col = st.columns([8,2])
        b = b_col.number_input("Age", min_value=0, max_value=100, value=62)
        label_col.markdown("years")
        
        c_col, c_label_col = st.columns([8,2])
        c = c_col.number_input("CA15-3", min_value=0.00, max_value=2000.00, value=12.50)
        c_label_col.markdown("U/mL")
        
        d_col, d_label_col = st.columns([8,2])
        d = d_col.number_input("CEA", min_value=0.00, max_value=10000.00, value=2.49)
        d_label_col.markdown("ng/mL")
        
        e_col, e_label_col = st.columns([8,2])
        e = e_col.number_input("CYFRA21-1", min_value=0.00, max_value=200.00, value=2.74)
        e_label_col.markdown("ng/mL")
        
        f_col, f_label_col = st.columns([8,2])
        f = f_col.number_input("NSE", min_value=0.00, max_value=500.00, value=16.60)
        f_label_col.markdown("ng/mL")
        
        g_col, g_label_col = st.columns([8,2])
        g = g_col.number_input("ALB", min_value=0.0, max_value=100.0, value=39.7)
        g_label_col.markdown("g/L")

with col2:
    if lang == '中文':
        h_col, h_label_col = st.columns([8,2])
        h = h_col.number_input("高密度脂蛋白", min_value=0.00, max_value=10.0, value=1.12)
        h_label_col.markdown("mmol/L")
        
        i_col, i_label_col = st.columns([8,2])
        i = i_col.number_input("甘油三酯", min_value=0.0, max_value=20.00, value=1.09)
        i_label_col.markdown("mmol/L")
        
        j_col, j_label_col = st.columns([8,2])
        j = j_col.number_input("血尿素", min_value=0.0, max_value=20.0, value=5.2)
        j_label_col.markdown("mmol/L")
        
        options = ["无", "有"]
        k = st.selectbox("咳痰", options, index=0)
        l = st.selectbox("咯血", options, index=0)
        m = st.selectbox("胸闷", options, index=0)
        n = st.selectbox("发热", options, index=0)
        
        k_val = 1 if k == "有" else 0
        l_val = 1 if l == "有" else 0
        m_val = 1 if m == "有" else 0
        n_val = 1 if n == "有" else 0
    else:
        h_col, h_label_col = st.columns([8,2])
        h = h_col.number_input("HDL", min_value=0.00, max_value=10.0, value=1.12)
        h_label_col.markdown("mmol/L")
        
        i_col, i_label_col = st.columns([8,2])
        i = i_col.number_input("TG", min_value=0.0, max_value=20.00, value=1.09)
        i_label_col.markdown("mmol/L")
        
        j_col, j_label_col = st.columns([8,2])
        j = j_col.number_input("Urea", min_value=0.0, max_value=20.0, value=5.2)
        j_label_col.markdown("mmol/L")
        
        options = ["No", "Yes"]
        k = st.selectbox("Expectoration", options, index=0)
        l = st.selectbox("Hemoptysis", options, index=0)
        m = st.selectbox("Distress", options, index=0)
        n = st.selectbox("Fever", options, index=0)
        
        k_val = 1 if k == "Yes" else 0
        l_val = 1 if l == "Yes" else 0
        m_val = 1 if m == "Yes" else 0
        n_val = 1 if n == "Yes" else 0

# 定义不同语言下的因子名称
feature_names = {
    '中文': ['性别', '年龄', '肿瘤相关抗原15-3', '癌胚抗原', '细胞角蛋白19片段', '神经元特异性烯醇化酶', '白蛋白', '高密度脂蛋白', '甘油三酯', '血尿素', '咳痰', '咯血', '胸闷', '发热'],
    'English': ['Gender', 'Age', 'CA15-3', 'CEA', 'CYFRA21-1', 'NSE', 'ALB', 'HDL', 'TG', 'Urea', 'Expectoration', 'Hemoptysis', 'Distress', 'Fever']
}

# Unpickle classifier    
mm = joblib.load('LightGBM.pkl')
    
# If button is pressed
if st.button("Submit"):
    # Store inputs into dataframe
    X = pd.DataFrame([[a_val, b, c, d, e, f, g, h, i, j, k_val, l_val, m_val, n_val]], 
                     columns=["Gender","Age","5A","8A","9A","10A","33A","42A","47A","50A","2BB","3BB","4BB","11BB"])
    
    # Get prediction
    result111 = mm.predict(X)
    result_prob_pos = mm.predict_proba(X)[0][1] * 100
    
    if lang == '中文':
        st.text(f"肺结节恶性概率是: {round(result_prob_pos, 2)}%")
    else:
        st.text(f"The probability of malignancy is: {round(result_prob_pos, 2)}%")
    
    # SHAP分析
    explainer = shap.TreeExplainer(mm) 
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    else:
        shap_values = shap_values
    
    shap_values = shap_values.reshape((1, -1))
    
    if isinstance(explainer.expected_value, list) and len(explainer.expected_value) == 2:
        expected_value = explainer.expected_value[1]
    else:
        expected_value = explainer.expected_value
    
    # 绘制 SHAP force plot
    shap_fig = shap.force_plot(expected_value, shap_values, X.iloc[0], feature_names=feature_names[lang], matplotlib=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(shap_fig)
    
    # 创建一个新的DataFrame来存储用户输入的数据
    new_data = pd.DataFrame([[a, b, c, d, e, f, g, h, i, j, k, l, m, n, result_prob_pos/100, None]], 
                            columns=st.session_state['data'].columns)
    
    # 将预测结果添加到新数据中
    st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)

# 上传文件按钮
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    column_mapping = {
        'Gender': 'Gender',
        'Age': 'Age',
        'CA15-3': '5A',
        'CEA': '8A',
        'CYFRA21-1': '9A',
        'NSE': '10A',
        'ALB': '33A',
        'HDL': '42A',
        'TG': '47A',
        'Urea': '50A',
        'Expectoration': '2BB',
        'Hemoptysis': '3BB',
        'Distress': '4BB',
        'Fever': '11BB'
    }
    
    df = df.rename(columns=column_mapping)
    
    required_cols = ["Gender","Age","5A","8A","9A","10A","33A","42A","47A","50A","2BB","3BB","4BB","11BB"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing columns in the uploaded file: {', '.join(missing_cols)}")
    else:
        for _, row in df.iterrows():
            X = pd.DataFrame([row[required_cols]], columns=required_cols)
            result = mm.predict(X)[0]
            result_prob = mm.predict_proba(X)[0][1]
            
            new_data = pd.DataFrame([[row["Gender"], row["Age"], row["5A"], row["8A"], row["9A"], row["10A"], row["33A"], row["42A"], row["47A"], row["50A"], row["2BB"], row["3BB"], row["4BB"], row["11BB"], result_prob, None]], 
                                    columns=st.session_state['data'].columns)
            st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)
    
st.write(st.session_state['data'])

st.markdown('<div style="font-size: 12px; text-align: right;">郑州大学第一附属医院</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 12px; text-align: right;">THE FIRST AFFILIATED HOSPITAL OF ZHENGZHOU UNIVERSITY</div>', unsafe_allow_html=True)