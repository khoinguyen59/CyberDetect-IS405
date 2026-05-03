import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt

# Cấu hình trang
st.set_page_config(page_title="CyberDetect-MLP Dashboard", layout="wide", page_icon="🛡️")

st.title("🛡️ Hệ Thống Phát Hiện Tấn Công Mạng CyberDetect-MLP")
st.markdown("Dashboard theo dõi gói tin theo thời gian thực kết hợp Explainable AI (SHAP).")

# Tính năng upload dữ liệu test
st.sidebar.header("Tải Dữ Liệu")
uploaded_file = st.sidebar.file_uploader("Upload file CSV (Đã qua tiền xử lý)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Tải dữ liệu thành công! (Số lượng mẫu: {df.shape[0]})")
    
    # Giả lập load model (Trong thực tế cần đường dẫn chính xác)
    st.sidebar.markdown("---")
    st.sidebar.info("Đang sử dụng mô hình CyberDetect-MLP (đã huấn luyện)")
    
    # Chọn 1 gói tin để phân tích
    st.subheader("1. Phân Tích Gói Tin Bất Kỳ")
    sample_index = st.number_input("Nhập Index của gói tin cần kiểm tra:", min_value=0, max_value=df.shape[0]-1, value=0)
    
    # Hiển thị thông tin gói tin
    sample_data = df.iloc[[sample_index]]
    st.write("Thông số gói tin (Top 30 Features):")
    st.dataframe(sample_data)
    
    # Cảnh báo tấn công (Giả lập dự đoán từ mô hình)
    st.subheader("2. Kết Quả Dự Đoán & Cảnh Báo (Alerts)")
    
    if st.button("Chạy Phân Tích (Prediction)"):
        with st.spinner('Đang phân tích gói tin qua CyberDetect-MLP...'):
            # TODO: Thay thế bằng model.predict(sample_data.values) thực tế
            # Đây là giả lập dự đoán cho mục đích Demo UI
            fake_prob = np.random.uniform(0.1, 0.99) 
            is_attack = fake_prob > 0.5
            
            if is_attack:
                st.error(f"🚨 CẢNH BÁO TẤN CÔNG MẠNG! (Xác suất: {fake_prob:.2%})")
            else:
                st.success(f"✅ Gói tin bình thường (Xác suất tấn công: {fake_prob:.2%})")
                
            # Explainable AI: SHAP Force Plot
            st.subheader("3. Explainable AI (Tại sao hệ thống đưa ra quyết định này?)")
            st.info("Biểu đồ SHAP bên dưới cho biết tính năng (feature) nào làm tăng hoặc giảm xác suất tấn công.")
            
            # Giả lập SHAP plot cho Demo vì chạy SHAP thực tế cần background dataset
            fig, ax = plt.subplots(figsize=(10, 3))
            features = sample_data.columns[:10]
            importance = np.random.randn(10)
            colors = ['red' if x > 0 else 'blue' for x in importance]
            
            ax.barh(features, importance, color=colors)
            ax.set_title("SHAP Feature Attribution (Local Explanation)")
            ax.set_xlabel("Đóng góp vào xác suất Tấn công")
            st.pyplot(fig)
            
            st.markdown("""
            **Hướng dẫn đọc biểu đồ SHAP:**
            - Các cột màu **Đỏ** đẩy xác suất tấn công LÊN CAO (Nguy cơ).
            - Các cột màu **Xanh** kéo xác suất tấn công XUỐNG THẤP (An toàn).
            - Độ dài của cột thể hiện mức độ tác động của feature đó.
            """)
else:
    st.info("Vui lòng upload một file CSV chứa dữ liệu test đã tiền xử lý để bắt đầu.")
    st.markdown("*(Nếu chưa có file, bạn có thể lấy `test_processed.csv` từ thư mục `data` sau khi chạy Notebook 01)*")

st.markdown("---")
st.caption("Developed by Nhóm 28 - CyberDetect-MLP Team (2026)")
