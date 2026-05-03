# Tóm Tắt Dự Án: CyberDetect-MLP

## 1. Mục Đích Chính Của Dự Án
Dự án nhằm mục tiêu xây dựng một hệ thống phát hiện xâm nhập mạng (NIDS) dựa trên thuật toán **Multilayer Perceptron (MLP)** cho môi trường thiết bị IoT (Sử dụng tập dữ liệu TON_IoT). 
Hệ thống không chỉ tập trung vào khả năng phân loại đa lớp (phân biệt giữa trạng thái Normal và 9 loại tấn công mạng) với độ chính xác cao mà còn giải quyết các bài toán hóc búa sau:
- Giải quyết bài toán mất cân bằng dữ liệu nghiêm trọng ở các lớp tấn công thiểu số.
- Lựa chọn đặc trưng tối ưu để giảm tải tính toán.
- Cung cấp khả năng giải thích dự đoán (Explainable AI - XAI).
- Benchmark (đối chiếu) sức mạnh của mô hình với các kiến trúc học sâu tiên tiến (Modern Neural Tabular Baselines).

---

## 2. Tài Liệu Cốt Lõi (Core Documentation)
Đây là 3 file định hướng toàn bộ chiến lược, ràng buộc và cấu trúc của đồ án:
- **`Plan_Main.md`**: Bản kế hoạch gốc, ghi chép lại việc phân tích repo CyberDetect-MLP gốc, các điểm khác biệt so với bài báo khoa học, và vạch ra toàn bộ lộ trình làm việc cốt lõi của nhóm.
- **`02_Email_CyberDetectMLP.md`**: File chỉ định các yêu cầu mở rộng (Bonus/Extension), đóng vai trò như "đề bài" để nhóm bổ sung các tính năng nâng cao như XAI và Benchmark các baseline mới.
- **`Nhom28_CyberDetect_MLP_Final/README.md`**: Trang tổng quan của dự án, tổng hợp lại bộ source code, kết quả thực nghiệm và hướng dẫn người khác cách chạy lại dự án.

---

## 3. Những Gì Đã Làm (Hoàn Thành)

### 2.1. Tiền Xử Lý Dữ Liệu & Rút Trích Đặc Trưng
- **Nội dung:** Làm sạch dữ liệu, chuẩn hoá dữ liệu, loại bỏ nhiễu. Sử dụng Mutual Information (MI) kết hợp với các kỹ thuật khác để rút trích ra **Top 30 features** mang tính quyết định nhất.
- **File quan trọng:** 
  - `CyberDetect-Phase2/notebooks/...` (Chứa các file chạy thuật toán rút gọn chiều dữ liệu như PCA, AE, và MI).

### 2.2. Huấn Luyện & Xử Lý Mất Cân Bằng Đa Lớp (10 Classes)
- **Nội dung:** Xây dựng mô hình CyberDetect-MLP cho bài toán phân loại 10 lớp. Triển khai các phương pháp lai (SMOTE-ENN, ADASYN) và class weights. Ghi nhận minh bạch hiệu ứng phụ của việc cân bằng dữ liệu (ví dụ: SMOTE-ENN có thể làm giảm F1-Macro ở một số lớp do nhiễu ở biên quyết định).
- **File quan trọng:** 
  - `Nhom28_CyberDetect_MLP_Final/notebooks/07_ClassWeight_PerClass.ipynb` (Toàn bộ logic huấn luyện 10 classes, áp dụng SMOTE-ENN và xuất bảng F1 phân tích).

### 2.3. Tích Hợp Explainable AI (XAI)
- **Nội dung:** Khắc phục điểm yếu "hộp đen" của AI. Triển khai **SHAP** (GradientExplainer) để giải thích độ quan trọng của các features trên toàn bộ tập dữ liệu (Global). Triển khai **Integrated Gradients** để mổ xẻ cụ thể lý do tại sao một gói tin duy nhất lại bị hệ thống nhận diện là tấn công (Local).
- **File quan trọng:**
  - `Nhom28_CyberDetect_MLP_Final/notebooks/08_Explainable_AI.ipynb` (Train MLP 10 classes và xuất biểu đồ giải thích SHAP/IG).

### 2.4. ??i Chi?u M? H?nh Hi?n ??i (Modern Baselines)
- **N?i dung:** Ch?ng minh s?c m?nh c?a m? h?nh ?? xu?t b?ng c?ch code v? ??i chi?u tr?c ti?p v?i c?c ki?n tr?c H?c S?u hi?n ??i b?ng PyTorch: **1D-CNN, LSTM, TabNet, TabTransformer** tr?n c?ng m?t chu?n ??nh gi?. KAN ch? gi? ? m?c h??ng m? r?ng n?u d?ng implementation KAN th?t nh? `efficient-kan`, kh?ng d?ng baseline gi? l?p.
- **File quan tr?ng:**
  - `Nhom28_CyberDetect_MLP_Final/notebooks/09_Modern_Baselines.ipynb` (Pipeline train & evaluate cho c?c m?ng n?-ron ti?n ti?n, xu?t bi?u ?? c?t so s?nh F1-Macro).

### 2.5. Xây Dựng Giao Diện (Dashboard)
- **Nội dung:** Viết một Web App trực quan để người dùng có thể upload dữ liệu gói tin, mô hình sẽ quét và đưa ra giải thích SHAP ngay lập tức (hiện đang trong giai đoạn Demo giao diện).
- **File quan trọng:**
  - `Nhom28_CyberDetect_MLP_Final/app_dashboard.py` (Script Streamlit xây dựng giao diện UI).

---

## 3. Đang Làm (Current State)
- **Re-check & Refactor Notebooks:** Đã rà soát, loại bỏ các lỗi đường dẫn thừa thãi (như trỏ nhầm về ổ cứng local `DLL/Main-P`) để tối ưu hoá việc up thẳng thư mục `CyberDetect-Phase3` lên Google Drive và Run All trên Google Colab mượt mà nhất.
- **Chuẩn Hoá Logic Đa Lớp:** Đã đồng bộ bài toán từ phân loại nhị phân (Binary) sang **Phân loại Đa Lớp (Multiclass)** ở các file `08` và `09` để nối tiếp chuẩn xác kết quả của file `07`.
- **Thử Nghiệm Chạy Thực Tế:** Sẵn sàng đưa các notebook đã sửa đổi hoàn thiện lên môi trường Colab để lấy output thực tế, trích xuất biểu đồ phục vụ báo cáo.

---

## 4. Dự Kiến Sắp Làm (Future Work / Tuỳ Chọn)
- **Chạy Code Lấy Biểu Đồ Cuối Cùng:** Chạy file `08` và `09` trên môi trường Colab để xuất các ảnh SHAP (Summary Plot, Force Plot) và biểu đồ cột so sánh `F1-Macro` cho vào tài liệu báo cáo.
- **Tích Hợp Model Thực Tế Vào Dashboard:** Lưu trọng số mô hình tốt nhất (ví dụ `mlp_model.h5`) và nhúng vào `app_dashboard.py` để Dashboard xử lý dữ liệu thật hoàn chỉnh.
- **Cross-Dataset Validation (Tuỳ Chọn):** Thử nghiệm pipeline hiện tại trên tập dữ liệu đánh giá ngoại lai khác (ví dụ: CSE-CIC-IDS2018) để kiểm chứng khả năng thích nghi của pipeline sau khi huấn luyện lại (Retrain Adaptability).
- **Tổng Kết Tài Liệu Báo Cáo:** Lắp ráp tất cả các kết quả thực nghiệm, hình ảnh chứng minh, bảng so sánh thành một bài tiểu luận/bài báo khoa học hoàn chỉnh.
