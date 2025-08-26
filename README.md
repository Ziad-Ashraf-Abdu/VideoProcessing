# 🎾 VideoProcessing

A Python-based **video processing server** that connects with our mobile app to extract tennis performance parameters.
The server automatically processes uploaded videos, extracts key motion and performance metrics, and forwards them to a **helper server**. The helper server then communicates with our **Hugging Face model API** to analyze and return insights back to the app.

---

## 📁 Project Structure

```
VideoProcessing/
├── .gitignore                 # Ignored files and folders
├── VideoProcessing.py         # Core video processing pipeline
├── server.py                  # Helper Server (forwards metrics to Hugging Face)
└── README.md                  # Project documentation
```

---

## 🧰 Features

* **Video Processing Server**: Extracts frames, motion data, and performance parameters from tennis videos.
* **Helper Server**: Sends extracted parameters to a Hugging Face endpoint for ML-based analysis.
* **App Integration**: Designed to communicate seamlessly with our mobile application.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Ziad-Ashraf-Abdu/VideoProcessing.git
cd VideoProcessing
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Start the Video Processing Server

```bash
python VideoProcessing.py
```

This server listens for video input from the app, processes it, and extracts parameters.

### 2. Start the Helper Server

```bash
python server.py
```

This forwards the extracted parameters to the Hugging Face endpoint and returns analyzed results.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/name`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to your branch (`git push origin feature/name`)
5. Open a Pull Request

Please follow the existing code style and include tests for new functionality.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Ziad Ashraf Abdu**

* GitHub: [@Ziad-Ashraf-Abdu](https://github.com/Ziad-Ashraf-Abdu)
* Email: [ziad.mohamed04@eng-st.cu.edu.eg](mailto:ziad.mohamed04@eng-st.cu.edu.eg)

---

Would you like me to also add a **diagram** (ASCII or simple image link) showing how the **App → Video Processing Server → Helper Server → Hugging Face → Back to App** flow works? That would make it clearer for new contributors.
