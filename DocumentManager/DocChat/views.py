import os
import time
from django.shortcuts import render
from django.core.files.storage import default_storage
from langchain_groq import ChatGroq # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain.chains.combine_documents import create_stuff_documents_chain # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain.chains import create_retrieval_chain # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings # type: ignore
from dotenv import load_dotenv # type: ignore

# Load environment variables
load_dotenv()

# Initialize LangChain components
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def vector_embedding(uploaded_files):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    save_dir = "uploaded_documents"
    os.makedirs(save_dir, exist_ok=True)
    
    all_docs = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(save_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(all_docs)
    
    global vectors
    vectors = FAISS.from_documents(final_documents, embeddings)
    return "Vector Store DB has been re-initialized with new documents."


def document_chat_view(request):
    context = {}
    
    if request.method == "POST":
        if "upload" in request.POST:
            uploaded_files = request.FILES.getlist("pdf_files")
            if uploaded_files:
                message = vector_embedding(uploaded_files)
                context["message"] = message
            else:
                context["error"] = "Please upload at least one PDF file."
        
        elif "ask" in request.POST:
            question = request.POST.get("question")
            if question and "vectors" in globals():
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start = time.process_time()
                response = retrieval_chain.invoke({'input': question})
                elapsed_time = time.process_time() - start
                
                context["response"] = response['answer']
                context["response_time"] = elapsed_time
                context["documents"] = [doc.page_content for doc in response.get("context", [])]
            else:
                context["error"] = "No documents processed or question missing."
    
    return render(request, "document_chat.html", context)
