# 📄 Document Manager

**Document Manager** is a powerful, all-in-one web-based tool designed to help users interact with, analyze, transform, and manage their documents seamlessly. Whether you're working with PDFs, Word files, or need advanced AI-powered document processing — this tool has you covered.

---

## 🚀 Key Features

### 🧠 AI-Powered Utilities

#### 1. 💬 Chat with Document (RAG-based)
Interact with your documents using natural language! Ask questions and get answers using **Retrieval-Augmented Generation (RAG)**. Ideal for contracts, reports, manuals, and large documents.

- Upload any document
- Ask queries in plain English
- Get instant, context-aware responses

#### 2. 📂 Document Classification
Automatically detect and label the type or category of uploaded documents using machine learning — such as invoices, legal docs, resumes, etc.

- Multi-class classification
- Works on Images 
- Useful for organizing large datasets

#### 3. 📊 Table Extraction (Structured & Unstructured)
Extract tabular data intelligently from any document:
- Structured tables (digital PDFs)
- Unstructured tables (scanned or image-based PDFs)

> Output can be exported to CSV or Excel.

---

### 🧰 PDF Utility Suite

These six PDF tools make managing your PDFs easier than ever:

#### 4. 📝 Word to PDF Converter
Easily convert `.doc` or `.docx` Word files into clean, compressed PDFs.

#### 5. 🛡️ Redact PDF
Remove or hide sensitive information from PDFs, like emails, phone numbers, or names — automatically or via custom keywords.

#### 6. 🔍 Generate Searchable PDF
Turn scanned/image-based PDFs into **searchable** ones using OCR so that you can search and copy content.

#### 7. 📦 Compress PDF
Reduce PDF file sizes without compromising on quality.

#### 8. ✂️ Split PDF
Split large PDFs into smaller chunks by page ranges or automatically.

#### 9. ➕ Merge PDF
Combine multiple PDFs into a single organized document.

---

## 🧑‍💻 Who Is It For?

- ✅ Researchers & Students
- ✅ Lawyers & Contract Managers
- ✅ Businesses with document-heavy workflows
- ✅ Developers looking to integrate intelligent PDF tools

---

## 🛠️ Tech Stack (Under the Hood)

- **Backend**: Django
- **AI Models**: LLMs (RAG with embedding + vector DB), OCR engines
- **PDF Processing**: PyMuPDF, pikepdf, pdfplumber, docx2pdf, etc.
- **Frontend**: Bootstrap, HTML5, Chart.js (for dashboard/stats)
- **Vector Store**: FAISS  (for retrieval)
- **Database**: PostgreSQL

---

## 📦 How to Use (User Guide)

1. Upload your document.
2. Select the desired operation:
   - Chat
   - Classify
   - Extract table
   - Redact / Compress / Merge etc.
3. For Chat, ask your question.
4. For other utilities, click **Download** to get the final document or data.

---
   

**Built with ❤️ to simplify document management using AI and modern Python tools.**
