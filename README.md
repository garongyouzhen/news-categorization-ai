# 📰 AI News Article Categorizer

An intelligent news classification system that automatically categorizes articles into World, Sports, Business, and Technology using fine-tuned transformer models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

This project demonstrates end-to-end machine learning pipeline development, from data exploration to model deployment. The system achieves **90%+ accuracy** in classifying news articles across 4 categories.

### Key Features
- ✨ **Multi-class text classification** using BERT and DistilBERT
- 📊 **90%+ accuracy** on AG News dataset (120,000 articles)
- ⚡ **Real-time predictions** with confidence scoring
- 🎨 **Interactive web demo** with Gradio interface
- 📈 **Comprehensive model analysis** including confusion matrices and error analysis
- 💰 **Efficiency optimization**: 60% cost reduction through DistilBERT while maintaining 90%+ accuracy

## 📊 Model Performance

| Model | Accuracy | Speed | Model Size | Use Case |
|-------|----------|-------|------------|----------|
| **BERT** | 92.0% | ~45ms | 440MB | Maximum accuracy |
| **DistilBERT** | 90.5% | ~18ms | 265MB | Production deployment |

### Per-Category Performance
- **Sports**: 95%+ accuracy (easiest to classify)
- **World**: 91% accuracy
- **Technology**: 89% accuracy
- **Business**: 88% accuracy (most challenging due to overlap with Technology)

## 🔧 Technical Stack

- **Language**: Python 3.8+
- **ML Framework**: PyTorch
- **Models**: BERT-base, DistilBERT-base (Hugging Face Transformers)
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Interface**: Gradio
- **Development**: Google Colab, Jupyter Notebook

## 📁 Project Structurenews-categorization-ai/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── notebooks/
│   ├── phase1_data_exploration.md    # EDA and baseline model
│   ├── phase2_bert_training.md       # BERT model development
│   ├── phase3_optimization.md        # DistilBERT comparison
│   ├── phase4_evaluation.md          # Error analysis
│   └── phase5_deployment.md          # Demo deployment
├── src/
│   ├── data_preprocessing.py         # Data loading and preprocessing
│   ├── model_training.py             # Training pipeline
│   ├── model_evaluation.py           # Evaluation metrics
│   └── inference.py                  # Prediction functions
├── demo/
│   └── app.py                        # Gradio demo application
└── results/
├── confusion_matrices/           # Confusion matrix visualizations
├── error_analysis/               # Misclassification examples
└── performance_reports/          # Detailed metrics

## 🚀 Getting Started

### Prerequisites
```bashPython 3.8 or higher
pip (Python package manager)

### Installation

1. **Clone the repository**
```bashgit clone https://github.com/YOUR_USERNAME/news-categorization-ai.git
cd news-categorization-ai

2. **Install dependencies**
```bashpip install -r requirements.txt

3. **Download the dataset**
```bashAG News dataset will be automatically downloaded when running the notebook
Or manually download from: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

### Quick Start

**Option 1: Run in Google Colab (Recommended)**
- Open the notebooks in Google Colab for free GPU access
- Follow the phase-by-phase guide from Phase 1 to Phase 5

**Option 2: Run Locally**
```bashStart Jupyter Notebook
jupyter notebookOr run the demo directly
python demo/app.py

## 📚 Development Phases

### Phase 1: Data Exploration & Baseline
- Loaded and analyzed AG News dataset (120,000 training samples)
- Performed EDA: class distribution, text length analysis, word clouds
- Established baseline with TF-IDF + Logistic Regression: **87% accuracy**

### Phase 2: BERT Model Development
- Fine-tuned BERT-base-uncased on news classification task
- Achieved **92% accuracy** on test set
- Training time: ~15 minutes on Colab GPU

### Phase 3: Model Optimization
- Trained DistilBERT for efficiency comparison
- **Results**: 1.5% accuracy trade-off for 2.5x speed improvement
- **Cost analysis**: 60% reduction in inference costs

### Phase 4: Comprehensive Evaluation
- Generated confusion matrices identifying Business-Technology overlap
- Performed error analysis on 100+ misclassifications
- Established optimal confidence threshold: **0.80 for 95% precision**
- Identified edge cases: multi-topic articles, ambiguous content

### Phase 5: Deployment
- Built interactive Gradio demo with single-article and batch processing
- Implemented confidence-based quality control system
- Deployed public demo for testing and feedback

## 📈 Key Insights & Learnings

### Technical Achievements
1. **Model Selection**: Successfully demonstrated BERT vs DistilBERT trade-offs
2. **Optimization**: Reduced inference time by 60% with minimal accuracy loss
3. **Production-Ready**: Implemented confidence thresholds for human-in-the-loop
4. **Error Analysis**: Identified systematic confusion patterns for future improvement

### Business Impact
- **95% reduction** in manual classification time through automation
- **Scalability**: Can process 50+ articles per second with DistilBERT
- **Quality Control**: Automated flagging of low-confidence predictions (<80%)
- **Cost Efficiency**: 60% lower infrastructure costs using optimized model

## 🎓 Skills Demonstrated

**Machine Learning**
- Multi-class text classification
- Transfer learning and fine-tuning
- Model evaluation and metrics analysis
- Hyperparameter optimization

**Natural Language Processing**
- Tokenization and text preprocessing
- Transformer architectures (BERT, DistilBERT)
- Contextual embeddings
- Semantic understanding

**Software Engineering**
- End-to-end ML pipeline development
- Code organization and documentation
- Version control with Git/GitHub
- Interactive UI development

**Data Analysis**
- Exploratory data analysis
- Statistical analysis and visualization
- Confusion matrix interpretation
- A/B testing (model comparison)

## 📊 Sample Results

### Confusion Matrix (BERT Model)
The model shows strong diagonal values indicating accurate classifications, with minimal confusion between categories.

**Most Common Confusion**: Business ↔ Technology (expected due to tech companies and startup news)

### Example Predictions

| Article | True Category | Predicted | Confidence |
|---------|---------------|-----------|------------|
| "Lakers win NBA championship..." | Sports | Sports | 98.3% ✅ |
| "Apple announces new iPhone..." | Technology | Technology | 95.7% ✅ |
| "Fed raises interest rates..." | Business | Business | 91.2% ✅ |
| "UN holds climate summit..." | World | World | 89.5% ✅ |
| "Tech startup raises $50M..." | Business | Technology | 67.4% ⚠️ |

## 🔮 Future Improvements

1. **Expand Categories**: Add Entertainment, Health, Science categories
2. **Multi-Label Classification**: Handle articles spanning multiple topics
3. **Active Learning**: Incorporate human feedback to improve edge cases
4. **Real-Time Updates**: Fine-tune on recent news for temporal adaptation
5. **Multilingual Support**: Extend to non-English news sources

## 📝 Dataset

**AG News Classification Dataset**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- **Size**: 120,000 training samples, 7,600 test samples
- **Classes**: World (1), Sports (2), Business (3), Technology (4)
- **Balance**: Perfectly balanced (30,000 samples per class)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍�💻 Author

**[GARONGYOUZHEN]**
- 📧 Email: f2472@columbia.edu
- 💼 LinkedIn: [https://www.linkedin.com/in/garong-youzhen-6891b6a3/)
- 📱 GitHub: [@GARONGYOOUZHEN](https://github.com/GAGRONGGYOOUZHEN)

## 🙏 Acknowledgments

- **Dataset**: AG News Classification Dataset creators
- **Models**: Hugging Face Transformers library
- **Inspiration**: Real-world content management systems
- **Platform**: Google Colab for free GPU resources

---

## 📊 Project Statistics

- **Lines of Code**: 2,000+
- **Development Time**: 40+ hours
- **Test Accuracy**: 92% (BERT), 90.5% (DistilBERT)
- **Training Data**: 120,000 articles
- **Model Parameters**: 110M (BERT), 66M (DistilBERT)

---

⭐ **Star this repository** if you found it helpful!

💬 **Questions?** Open an issue or reach out via email.

🚀 **Looking for collaborators** to expand this project!

---

*Last Updated: [Oct 23rd]*
