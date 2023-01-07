# Supervised Learning - Naive Bayes

### Nội dung:
1. Giới thiệu
2. Các ứng dụng
3. Thuật toán
4. Ưu/khuyết điểm
5. Xây dựng Naive Bayes sử dụng sklearn

## 1. Giới thiệu
+ Phân loại Naive Bayesian dựa trên định lý Bayes với các giả định độc lập giữa các yếu tố dự báo.
+ Một mô hình Naive Bayesian dễ xây dựng, không có ước lượng tham số lặp phức tạp nào khiến cho nó đặc biệt hữu ích cho các tập dữ liệu rất lớn.
+ Mặc dù sự đơn giản của nó, trình phân loại Naive Bayesian thường rất đánh ngạc nhiên và được sử dụng rộng rãi bởi vì nó thường làm tốt hơn các phương pháp phân loại phức tạp hơn.
+ Naive Bayes là một thuật toán thuộc nhóm Supervised Learning sử dụng rất hiệu quả cho Classification.
+ Khá giống với Linear Regression, tuy nhiên, Naive Bayes có xu hướng đào tạo nhanh hơn. Cái giá phải trả cho hiệu quả nhanh này là mô hình Naive Bayes thường cung cấp hiệu suất tổng quát hơi kém hơn so với các phân loại tuyến tính như Logistic Regression và LinearSVC.
+ Lí do mà các mô hình Naive Bayes rất hiệu quả là chúng học các tham số bằng cách xem xét từng đặc điểm/tính năng riêng lẻ và thu thập số liệu thống kê mỗi lớp đơn giản từ từng tính năng.
+ Có ba loại phân phối sử dụng phổ biến của Naive Bayes là GaussianNB, BernoulliNB và MultinomialNB

### GaussianNB
+ Có thể được áp dụng cho bất kỳ dữ liệu liên tục nào
+ GaussianNB lưu trữ giá trị trung bình và độ lệch chuẩn (standard deviation) của từng tính năng cho mỗi lớp.
+ GaussianNB chủ yếu được sử dụng trên dữ liệu high-dimensional data

### BernoulliNB 
+ Áp dụng trên dữ liệu nhị phân (mỗi thành phần là một giá trị binary bằng 0 hoặc bằng 1)
+ Được sử dụng rộng rãi cho dữ liệu đếm thưa thớt (Sparse count data), ví dụ: văn bản

### MultinomialNB
+ Áp dụng cho dữ liệu đếm (nghĩa là mỗi đối tượng đại diện cho một số nguyên đếm của thứ gì đó, ví dụ như tần suất một từ xuất hiện trong một câu)
+ Tính đến giá trị trung bình của từng tính năng (feature) cho mỗi lớp (class)
+ Được sử dụng rộng rãi cho dữ liệu đếm thưa thớt (sparse count data)

### So sánh
+ BernoulliNB và MultinomialNB chủ yếu được sử dụng trong phân loại dữ liệu văn bản.
+ MultinomialNB thường hoạt động tốt hơn BernoulliNB, đặc biệt là trên các tập dữ liệu với một số lượng tương đối lớn các tính năng nonzero (ví dụ: các tài liệu lớn)

### Đặc điểm chung
+ Với tất cả, để dự đoán, một điểm dữ liệu được so sánh với các số liệu thống kê cho mỗi lớp và lớp phù hợp nhất được dự đoán
+ Các mô hình Naive Bayes chia sẻ nhiều điểm mạnh và điểm yếu của các mô hình tuyến tính, đào tạo và dự đoán rất nhanh, quy trình đào tạo rất dễ hiểu
+ Các mô hình hoạt động rất tốt với dữ liệu thưa thớt nhiều chiều (high-dimensional sparse data) và tương đối mạnh mẽ với các tham số

## 2. Các ứng dụng
+ Realtime Prediction (dự đoán thời gian thực)
+ Multiclass Prediction (dự đoán đa lớp)
+ Text classification/Spam Filtering/Sentiment Analysis (phân loại văn bản, lọc thưc rác, phân tích trạng thái - trong phân tích truyền thông xã hội, để xác định tình cảm khách hành - tích cực và tiêu cực)
+ Recommendation System (hệ thống đề xuất): Naive Bayes Classifier và Collaborative Filtering cùng nhau xây dựng hệ thống đề xuất sử dụng kỹ thuật Machine Learning và khai thác dữ liệu để lọc thông tin không xác định và dự đoán người dùng có muốn một tài nguyên cụ thể hay không

## 3. Thuật toán
Định lý Bayes cung cấp một cách tính xác suất như sau: P(c|x), từ P(c), P(x) và P(x|c)
+ Phân loại Baive Bayes giả định rằng tác động của giá trị của một yếu tố dự báo - predictor - (x) trên một lớp nhất định (c) là độc lập với các giá trị của các yếu tố dự báo khác. Giả định này được gọi là lớp độc lập có điều kiện.
+ **Công thức:**

$$\textrm{P(c|x)} = \frac{\textrm{P(x|c)*P(c)}}{\textrm{P(x}}$$

**Trong đó:**
+ P(c|x): Xác suất xảy ra (posterior probability) của một sự kiện ngẫu nhiên c khi biết sự kiện liên quan x đã xảy ra.
+ P(x|c): Xác suất xảy ra x khi biết c xảy ra (likelihood)
+ P(c): Xác suất xảy ra của riêng c mà không quan tâm tới x (class prior probability)
+ P(x): Xác suất xảy ra của riêng x mà không quan tâm tới c (predictor prior probabilitiy)

## 4. Ưu/khuyết điểm
### Ưu điểm
+ Tính toán nhanh chóng
+ Đơn giản, dễ triển khai
+ Hoạt động tốt với bộ dữ liệu nhỏ
+ Hoạt động tốt với high-dimensionals
+ Hoạt động tốt ngay cả khi Naive Assumption không được đáp ứng hoàn hảo. Trong nhiều trường hợp, giá trị xấp xỉ (approximation) là đủ để xây dựng một trình phân loại tốt

### Khuyết điểm & cách giải quyết
+ Có thể có các tính năng tương quan => để giải quyết: loại bỏ các tính năng tương quan, vì khi các tính năng tương quan cao được chọn hai lần trong mô hình có thể dẫn đến việc vượt quá tầm quan trọng.
+ Có thể các tính năng liên tục (continuous features) không có phân bố chuẩn (normal distribution) xuất hiện. Để giải quyết ta nên sử dụng phép biến đổi hoặc các phương thức khác nhau để chuyển đổi nó thành phân phối chuẩn.
+ Nếu một biến phân loại có một danh mục trong tập dữ liệu thử nghiệm mà không được quan sát thấy trong tập dữ liệu huấn luyện, thì mô hình sẽ gán một xác suất bằng không. Nó sẽ không thể dự đoán được. Điều này thường gọi là "Tần số không" => Để giải quyết, ta có thể sử dụng kỹ thuật làm mịn. Một trong những kỹ thuật làm mịn đơn giản nhất được gọi là ước tính Laplace (tăng số lượng của biến có 0 lên một giá trị nhỏ, thường là 1, trong tử số, để xác suất tổng thể không trở thành 0). Mặc định Naive Bayes sử dụng kỹ thuật này.

## 5. Xây dựng Naive Bayes sử dụng sklearn

Dùng `sklearn.naive_bayes`

### Các bước thực hiện
+ Chọn model sẽ sử dụng là **Naive_bayes**
+ Tạo một tập dữ liệu feature và một tập target chứa các nhãn cho các thực thể
+ Chia dữ liệu thành hai phần: train-test
+ Áp dụng mô hình phù hợp: **GaussianNB, BernoulliNB** hoặc **MultinomialNB**
+ Xây dựng model với training data
+ Sử dụng model (fitted model) cho dữ liệu chưa biết (test data)
+ Đánh giá độ chính xác
+ Áp dụng model cho dự đoán dữ liệu mới
