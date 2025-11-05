
# 📱 Spam SMS Filtering using Machine Learning Techniques  

### 📄 Project Overview  
This project aims to automatically detect and filter **spam SMS messages** using **Machine Learning (ML)** and **Natural Language Processing (NLP)**, deployed as a **Streamlit web app**.  

With the exponential growth of mobile communication, spam messages have become a major concern. Unlike emails, SMS messages are short, unstructured, and often use deceptive wording — making traditional spam filters less effective.  

This project uses a Kaggle dataset of 5,572 messages to train multiple ML models like Decision Trees, Naive Bayes and Multilayer Perceptron networks models and identify spam messages efficiently. The app allows users to input text and instantly receive a prediction on whether it’s **Spam** or **Ham (non-spam)**.  

👉 **Try it live here:** 🔗[Spam SMS Classification Web Application] 
[Render](https://spam-sms-classification-a5c1.onrender.com) |
[Stremlit](https://spamsmsclassification-vikash.streamlit.app/)

---

## 🚀 Features  
- 🧠 **Machine Learning-based classification** of SMS as Spam or Ham  
- 💬 **Streamlit web interface** for real-time predictions  
- 🧹 **Preprocessing** using stop word removal, lemmatization, and TF-IDF  
- ⚡ **Lightweight, fast, and highly accurate** (98.81% accuracy)  
- 🧩 Resistant to *“Good Word Attack”* adversarial techniques  

---

## 💻 Streamlit Web Interface  

An interactive **Streamlit** app is built for end users to easily test SMS messages.  
Simply type or paste a message and the model will classify it as **Spam** or **Ham** instantly.

👉 **Live Demo:** 
[Render](https://spam-sms-classification-a5c1.onrender.com) |
[Stremlit](https://spamsmsclassification-vikash.streamlit.app/)

### 🧭 Run Locally
```bash
# Clone this repository
git clone https://github.com/Vicky9890/Spam_SMS_Classification.git

# Navigate to project folder
cd Spam_SMS_Classification

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 🧠 Models Used  
The following machine learning algorithms were implemented and compared:  

| Model | Description | Accuracy |
|--------|--------------|-----------|
| **Naïve Bayes** | Probabilistic classifier ideal for text data | **98.81%** |
| **Decision Tree** | Tree-based model for interpretability | 96.12% |
| **MLP (Neural Network)** | Learns complex non-linear relationships | 97.24% |

✅ **Naïve Bayes** performed best with **98.81% accuracy** and only **1% false positives**, making it the optimal choice for SMS spam detection.

---

## 📊 Results Visualization  

| Metric | Best Model (Naïve Bayes) | Result |
|---------|--------------------------|---------|
| Accuracy (VRR) | ✅ 98.81% | Excellent |
| False Positive Rate (FPR) | 🔻 1% | Very low |
| Detection Speed | ⚡ Fast | Suitable for real-time apps |

---

## 🖼️ Output Screenshots  

### 🔹 Streamlit Web Interface  
<img width="1494" height="682" alt="image" src="https://github.com/user-attachments/assets/10a3d9a6-9edf-4717-89e0-05355045f924" />


### 🔹 Spam Detection Result  
<img width="1505" height="944" alt="image" src="https://github.com/user-attachments/assets/e64a353d-8587-41b8-87c9-06cd75cac25d" />


### 🔹 Ham Detection Result  
<img width="1443" height="795" alt="image" src="https://github.com/user-attachments/assets/1e7c704b-474c-4c05-ba0d-0c20a172f2d2" />


---

## 🛠️ Technologies Used  
- **Python**  
- **Streamlit** (for web interface)  
- **Scikit-learn**  
- **NLTK**  
- **Pandas & NumPy**  
- **Matplotlib** (for visualization)  

---

## 📈 Future Enhancements  
- Add **Deep Learning models** (e.g., LSTM, BERT) for improved accuracy  
- Deploy on **AWS / Hugging Face Spaces**  
- Extend detection to **multilingual SMS datasets**  

---

## 👨‍💻 Authors  
**Vikash Kumar Pradhan**,  

📧 Contact: [vpradhan2002@gmail.com](mailto:vpradhan2002@gmail.com)

---

## 🏁 Conclusion  
This project demonstrates how integrating **Machine Learning**, **NLP**, and **Streamlit** can provide a reliable and user-friendly solution for SMS spam detection.  
With an accuracy of **98.81%**, the system effectively filters unwanted SMS messages in real-time.  

👉 **Experience the live app:** [https://spam-sms-classification-a5c1.onrender.com](https://spam-sms-classification-a5c1.onrender.com)  or  [https://spamsmsclassification-vikash.streamlit.app](https://spamsmsclassification-vikash.streamlit.app)
