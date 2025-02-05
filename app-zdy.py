import streamlit as st  
import pandas as pd  
import numpy as np  
import joblib  
import shap
from streamlit_shap import st_shap

# 初始化 session_state 中的 data
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=['Gender', 'Age', 'CA15-3', 'CEA', 'CYFRA21-1', 'NSE', 'MCH', 'Expectoration', 'Hemoptysis', 'Fever', 'Duration','Prediction', 'Label'])

# 设置页面为宽模式
st.set_page_config(layout="wide")
st.sidebar.image("hospital_logo2.png", caption="", width=300)
# Language setting  
lang = st.sidebar.selectbox('Choose language', ['中文', 'English'], index=1)

# Footer  
if lang == '中文':  
    st.sidebar.subheader('程序说明')  
    st.sidebar.write("<p style='font-size: 12px;'>申明: 这款小程序旨在提供一般信息，不能替代专业医疗建议或诊断。如果您对自己的健康有任何担忧，请务必咨询合格的医疗保健专业人员。</p>", unsafe_allow_html=True)
else:  
    st.sidebar.subheader('Program Description')  
    st.sidebar.write("<p style='font-size: 12px;'>Disclaimer: This mini app is designed to provide general information and is not a substitute for professional medical advice or diagnosis. Always consult with a qualified healthcare professional if you have any concerns about your health.</p>", unsafe_allow_html=True)

st.image("hospital_logo.png", caption="")
if lang == '中文':  
    st.header("结节风险评估模型（NRA）V1.0 ")
else:  
    st.header("NoduleRisk Assessment（NRA) nodel V1.0")


# 将输入分成三列，每列3-4个
col1, col2, col3 = st.columns(3)

gender_mapping = {
    '中文': {'男': 1, '女': 2},
    'English': {'Male': 1, 'Female': 2}
}

with col1:
    if lang == '中文':
        options = list(gender_mapping['中文'].keys())
        a = st.selectbox("性别", options, index=0)
        a_val = gender_mapping['中文'][a]
        
        options_bl = ["无", "有"]
        k = st.selectbox("咳痰", options_bl, index=0)
        l = st.selectbox("咯血", options_bl, index=0)
        m = st.selectbox("发热", options_bl, index=0)
        
        k_val = 1 if k == "有" else 0
        l_val = 1 if l == "有" else 0
        m_val = 1 if m == "有" else 0
    else:
        options = list(gender_mapping['English'].keys())
        a = st.selectbox("Gender", options, index=0)
        a_val = gender_mapping['English'][a]
		
        options_bl = ["No", "Yes"]
        k = st.selectbox("Expectoration", options_bl, index=0)
        l = st.selectbox("Hemoptysis", options_bl, index=0)
        m = st.selectbox("Distress", options_bl, index=0)
        
        k_val = 1 if k == "Yes" else 0
        l_val = 1 if l == "Yes" else 0
        m_val = 1 if m == "Yes" else 0

with col2:
    if lang == '中文':
        b_col, label_col = st.columns([8,2])
        b = b_col.number_input("年龄", min_value=0, max_value=100, value=62)
        label_col.markdown("岁")	

        n_col, n_label_col = st.columns([8,2])
        n = n_col.number_input("吸烟年数", min_value=0, max_value=100, value=30)
        n_label_col.markdown("年")		

        c_col, c_label_col = st.columns([8,2])
        c = c_col.number_input("癌抗原15-3", min_value=0.00, max_value=2000.00, value=24.50)
        c_label_col.markdown("U/mL")
        
        d_col, d_label_col = st.columns([8,2])
        d = d_col.number_input("癌胚抗原", min_value=0.00, max_value=10000.00, value=2.49)
        d_label_col.markdown("ng/mL")
    else:
        b_col, label_col = st.columns([8,2])
        b = b_col.number_input("Age", min_value=0, max_value=100, value=62)
        label_col.markdown("years")
		
        n_col, n_label_col = st.columns([8,2])
        n = n_col.number_input("Duration", min_value=0, max_value=100, value=30)
        n_label_col.markdown("years")

        c_col, c_label_col = st.columns([8,2])
        c = c_col.number_input("CA15-3", min_value=0.00, max_value=2000.00, value=24.50)
        c_label_col.markdown("U/mL")
        
        d_col, d_label_col = st.columns([8,2])
        d = d_col.number_input("CEA", min_value=0.00, max_value=10000.00, value=2.49)
        d_label_col.markdown("ng/mL")
		
with col3:
    if lang == '中文':
        e_col, e_label_col = st.columns([8,2])
        e = e_col.number_input("细胞角蛋白19片段", min_value=0.00, max_value=200.00, value=32.74)
        e_label_col.markdown("ng/mL")
        
        f_col, f_label_col = st.columns([8,2])
        f = f_col.number_input("神经元特异性烯醇化酶", min_value=0.00, max_value=500.00, value=21.60)
        f_label_col.markdown("ng/mL")
        
        g_col, g_label_col = st.columns([8,2])
        g = g_col.number_input("平均红细胞血红蛋白含量", min_value=0.0, max_value=100.0, value=39.7)
        g_label_col.markdown("pg")
    else:
        e_col, e_label_col = st.columns([8,2])
        e = e_col.number_input("CYFRA21-1", min_value=0.00, max_value=200.00, value=32.74)
        e_label_col.markdown("ng/mL")
        
        f_col, f_label_col = st.columns([8,2])
        f = f_col.number_input("NSE", min_value=0.00, max_value=500.00, value=21.60)
        f_label_col.markdown("ng/mL")
        
        g_col, g_label_col = st.columns([8,2])
        g = g_col.number_input("MCH", min_value=0.0, max_value=100.0, value=39.7)
        g_label_col.markdown("pg")
		
# 定义不同语言下的因子名称
feature_names = {
    '中文': ['性别', '年龄', '癌抗原15-3', '癌胚抗原', '细胞角蛋白19片段', '神经元特异性烯醇化酶', '平均红细胞血红蛋白含量', '咳痰', '咯血', '发热', '吸烟年数'],
    'English': ['Gender', 'Age', 'CA15-3', 'CEA', 'CYFRA21-1', 'NSE', 'MCH', 'Expectoration', 'Hemoptysis', 'Fever', 'Duration']
}

# Unpickle classifier    
mm = joblib.load('random_forest.pkl')
    
# If button is pressed
if lang == '中文':
    submit_button = st.button("提交")
else:
    submit_button = st.button("Submit")

if submit_button:
    # Store inputs into dataframe
    X = pd.DataFrame([[a_val, b, c, d, e, f, g, k_val,l_val,m_val,n]], 
                     columns=["Gender", "Age", "CA15-3", "CEA", "CYFRA21-1", "NSE", "MCH", "Expectoration", "Hemoptysis", "Fever", "Duration"])
    
    # Get prediction
    result111 = mm.predict(X)
    result_prob_pos = mm.predict_proba(X)[0][1] * 100
    
    if lang == '中文':
        st.text(f"肺结节恶性概率是: {round(result_prob_pos, 2)}%")
    else:
        st.text(f"The probability of malignancy is: {round(result_prob_pos, 2)}%")
    
    # 生成SHAP值并绘图
    shap.initjs()
    explainer = shap.TreeExplainer(mm) 
    shap_values = explainer.shap_values(X,check_additivity=False)[0]
    shap_values = shap_values[:,1] 
    shap_values = shap_values.reshape((1, -1)) 
    # 绘制 SHAP force plot
    shap_plot = shap.force_plot(explainer.expected_value[1], shap_values[0], X.iloc[0], feature_names=feature_names[lang])
    # st_shap(shap_plot, height=150, width=800)
    st_shap(shap_plot, height=150)
	
    # 创建一个新的DataFrame来存储用户输入的数据
    new_data = pd.DataFrame([[a_val, b, c, d, e, f, g, k_val,l_val,m_val,n, result_prob_pos/100, None]], 
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
        'CA15-3': 'CA15-3',
        'CEA': 'CEA',
        'CYFRA21-1': 'CYFRA21-1',
        'NSE': 'NSE',
        'MCH': 'MCH',
        'Expectoration': 'Expectoration',
        'Hemoptysis': 'Hemoptysis',
        'Fever': 'Fever',
        'Duration': 'Duration'
    }
    
	# 假设 'Label' 列在 Excel 文件中存在并且不参与计算
    label_column = 'label'  # 这是 Excel 文件中未参与计算的列名
	
    df = df.rename(columns=column_mapping)
    
    required_cols = ["Gender","Age","CA15-3","CEA","CYFRA21-1","NSE","MCH","Expectoration","Hemoptysis","Fever","Duration"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing columns in the uploaded file: {', '.join(missing_cols)}")
    else:
        for _, row in df.iterrows():
            X = pd.DataFrame([row[required_cols]], columns=required_cols)
            result = mm.predict(X)[0]
            result_prob = mm.predict_proba(X)[0][1]
            
			# 获取标签列的值
            label = row[label_column] if label_column in row else None
			
            new_data = pd.DataFrame([[row["Gender"], row["Age"], row["CA15-3"], row["CEA"], row["CYFRA21-1"], row["NSE"], row["MCH"], row["Expectoration"], row["Hemoptysis"], row["Fever"], row["Duration"], result_prob, label]], 
                                    columns=st.session_state['data'].columns)
            st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)
    
st.write(st.session_state['data'])

st.markdown('<div style="font-size: 12px; text-align: right;">郑州大学第一附属医院</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 12px; text-align: right;">THE FIRST AFFILIATED HOSPITAL OF ZHENGZHOU UNIVERSITY</div>', unsafe_allow_html=True)
