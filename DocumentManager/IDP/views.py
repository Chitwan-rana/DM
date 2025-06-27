from django.shortcuts import render
from django.conf import settings
from django import forms
import os
import tempfile
import PyPDF2
import google.generativeai as genai
import logging
from django.http import HttpResponse, FileResponse
import pikepdf
from pikepdf import Pdf
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

logger = logging.getLogger(__name__)
     
class SummarizationForm(forms.Form):
    pdf_file = forms.FileField(label='Select a PDF file')
    SUMMARY_CHOICES = [
        ('very_short', 'Very Short (about 100 words)'),
        ('short', 'Short (about 250 words)'),
        ('medium', 'Medium (about 500 words)'),
        ('long', 'Long (about 1000 words)'),
        ('very_long', 'Very Long (about 2000 words)'),
    ]
    summary_length = forms.ChoiceField(choices=SUMMARY_CHOICES, label='Summary Length', initial='medium')

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            for chunk in pdf_file.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        
        text = ""
        with open(temp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page_num].extract_text()
                if page_text:
                    text += page_text + "\n"
        
        os.unlink(temp_file_path)  # Delete the temporary file
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        return None

def get_summary_params(length_option):
    """Get summary parameters based on selected length option."""
    params = {
        'very_short': {'max_words': 100, 'prompt': 'Provide a very brief summary in about 100 words.'},
        'short': {'max_words': 250, 'prompt': 'Provide a short summary in about 250 words.'},
        'medium': {'max_words': 500, 'prompt': 'Provide a comprehensive summary in about 500 words.'},
        'long': {'max_words': 1000, 'prompt': 'Provide a detailed summary in about 1000 words.'},
        'very_long': {'max_words': 2000, 'prompt': 'Provide a very detailed summary in about 2000 words.'},
    }
    return params.get(length_option, params['medium'])

def summarize_text(text, summary_params):
    """Use generative AI to summarize text."""
    try:
        # Configure generative AI
        genai.configure(api_key=settings.GENAI_API_KEY)
        model = genai.GenerativeModel('gemini-1.0-pro')
        
        # Create prompt for summarization
        prompt = f"""
        Summarize the following text. {summary_params['prompt']}:
        
        {text}
        """
        
        # Generate summary
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return None

def summarize_pdf(request):
    """View to summarize PDF using generative AI."""
    context = {
        'form': SummarizationForm(),
        'summary': None,
        'original_text': None,
        'original_word_count': 0,
        'summary_word_count': 0,
        'error': None,
    }
    
    if request.method == 'POST':
        form = SummarizationForm(request.POST, request.FILES)
        context['form'] = form
        
        if form.is_valid():
            try:
                pdf_file = request.FILES['pdf_file']
                summary_length = form.cleaned_data['summary_length']
                
                # Extract text from PDF
                original_text = extract_text_from_pdf(pdf_file)
                if not original_text:
                    context['error'] = "Could not extract text from the PDF. The file might be corrupted or password-protected."
                    return render(request, 'summarize_pdf.html', context)
                
                context['original_text'] = original_text
                context['original_word_count'] = len(original_text.split())
                
                # Get summary parameters based on selected length
                summary_params = get_summary_params(summary_length)
                
                # Generate summary
                summary = summarize_text(original_text, summary_params)
                if not summary:
                    context['error'] = "Failed to generate summary. Please try again later."
                    return render(request, 'summarize_pdf.html', context)
                
                context['summary'] = summary
                context['summary_word_count'] = len(summary.split())
            
            except Exception as e:
                logger.error(f"Error in summarize_pdf view: {str(e)}")
                context['error'] = f"An error occurred: {str(e)}"
    
    return render(request, 'summarize_pdf.html', context)

## MERGE PDFs

class PDFMergeForm(forms.Form):
    pdf_file1 = forms.FileField(label='First PDF file')
    pdf_file2 = forms.FileField(label='Second PDF file')
    output_filename = forms.CharField(label='Output filename', 
                                     initial='merged.pdf',
                                     help_text='Name of the merged PDF file')

def merge_pdfs(pdf_file1, pdf_file2):
    """Merge two PDF files into one."""
    try:
        # Create temporary files for the input PDFs
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file1, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file2, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as output_temp:
            
            # Save the uploaded files to temporary files
            for chunk in pdf_file1.chunks():
                temp_file1.write(chunk)
            for chunk in pdf_file2.chunks():
                temp_file2.write(chunk)
            
            temp_file1_path = temp_file1.name
            temp_file2_path = temp_file2.name
            output_temp_path = output_temp.name
        
        # Create a PDF merger object
        merger = PyPDF2.PdfMerger()
        
        # Add the PDFs to the merger
        merger.append(temp_file1_path)
        merger.append(temp_file2_path)
        
        # Write to the output file
        with open(output_temp_path, 'wb') as output_file:
            merger.write(output_file)
            merger.close()
        
        # Read the merged PDF to return it
        with open(output_temp_path, 'rb') as merged_pdf:
            merged_content = merged_pdf.read()
        
        # Clean up the temporary files
        os.unlink(temp_file1_path)
        os.unlink(temp_file2_path)
        os.unlink(output_temp_path)
        
        return merged_content
        
    except Exception as e:
        logger.error(f"Error merging PDFs: {str(e)}")
        return None

def merge_pdfs_view(request):
    """View to merge two PDF files."""
    context = {
        'form': PDFMergeForm(),
        'error': None,
    }
    
    if request.method == 'POST':
        form = PDFMergeForm(request.POST, request.FILES)
        context['form'] = form
        
        if form.is_valid():
            try:
                pdf_file1 = request.FILES['pdf_file1']
                pdf_file2 = request.FILES['pdf_file2']
                output_filename = form.cleaned_data['output_filename']
                
                # Ensure the filename ends with .pdf
                if not output_filename.lower().endswith('.pdf'):
                    output_filename += '.pdf'
                
                # Merge the PDFs
                merged_pdf_content = merge_pdfs(pdf_file1, pdf_file2)
                if not merged_pdf_content:
                    context['error'] = "Failed to merge the PDF files. Please try again."
                    return render(request, 'merge_pdfs.html', context)
                
                # Create the HTTP response with the merged PDF
                response = HttpResponse(merged_pdf_content, content_type='application/pdf')
                response['Content-Disposition'] = f'attachment; filename="{output_filename}"'
                return response
            
            except Exception as e:
                logger.error(f"Error in merge_pdfs_view: {str(e)}")
                context['error'] = f"An error occurred: {str(e)}"
    
    return render(request, 'merge_pdfs.html', context)


## COMPRESS PDFs


class PDFCompressForm(forms.Form):
    pdf_file = forms.FileField(label='Select a PDF file to compress')
    compression_level = forms.ChoiceField(
        label='Compression Level',
        choices=[
            ('low', 'Low (Better Quality)'),
            ('medium', 'Medium (Balanced)'),
            ('high', 'High (Smaller Size)')
        ],
        initial='medium',
        help_text='Higher compression may affect image quality'
    )
    output_filename = forms.CharField(
        label='Output filename',
        initial='compressed.pdf',
        help_text='Name of the compressed PDF file'
    )

def compress_pdf(pdf_file, compression_level='medium'):
    """Compress a PDF file using pikepdf."""
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as input_temp, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as output_temp:
            
            # Save the uploaded file to a temporary file
            for chunk in pdf_file.chunks():
                input_temp.write(chunk)
            
            input_path = input_temp.name
            output_path = output_temp.name

        # Get original file size
        original_size = os.path.getsize(input_path)
        
        # Open and optimize the PDF
        with pikepdf.Pdf.open(input_path) as pdf:
            # Remove unnecessary data
            if hasattr(pdf, 'remove_unreferenced_resources'):
                pdf.remove_unreferenced_resources()
            
            # Save with compatible compression options
            pdf.save(output_path, 
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
                linearize=True)
        
        # Read the compressed PDF to return it
        with open(output_path, 'rb') as compressed_pdf:
            compressed_content = compressed_pdf.read()
        
        # Get compressed file size
        compressed_size = os.path.getsize(output_path)
        
        # Clean up temporary files
        os.unlink(input_path)
        os.unlink(output_path)
        
        compression_rate = (1 - (compressed_size / original_size)) * 100
        
        return {
            'content': compressed_content,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_rate': compression_rate
        }
        
    except Exception as e:
        logger.error(f"Error compressing PDF: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def compress_pdf_view(request):
    """View to compress a PDF file."""
    context = {
        'form': PDFCompressForm(),
        'error': None,
        'success': False,
    }
    
    if request.method == 'POST':
        form = PDFCompressForm(request.POST, request.FILES)
        context['form'] = form
        
        if form.is_valid():
            try:
                pdf_file = request.FILES['pdf_file']
                compression_level = form.cleaned_data['compression_level']
                output_filename = form.cleaned_data['output_filename']
                
                # Ensure the filename ends with .pdf
                if not output_filename.lower().endswith('.pdf'):
                    output_filename += '.pdf'
                
                # Check file size
                if pdf_file.size > 100 * 1024 * 1024:  # 100MB limit
                    context['error'] = "File is too large. Maximum file size is 100MB."
                    return render(request, 'compress_pdf.html', context)
                
                # Compress the PDF
                compression_result = compress_pdf(pdf_file, compression_level)
                if not compression_result:
                    context['error'] = "Failed to compress the PDF. Please try again."
                    return render(request, 'compress_pdf.html', context)
                
                # Store the compressed content in the session
                request.session['compressed_pdf'] = compression_result['content'].hex()
                request.session['output_filename'] = output_filename
                
                # Format sizes for display
                original_size_kb = compression_result['original_size'] / 1024
                compressed_size_kb = compression_result['compressed_size'] / 1024
                compression_rate = round(compression_result['compression_rate'], 2)
                
                # Add compression stats to the context
                context['success'] = True
                context['original_size'] = round(original_size_kb, 2)
                context['compressed_size'] = round(compressed_size_kb, 2)
                context['compression_rate'] = compression_rate
                context['output_filename'] = output_filename
                
                # Show readable file sizes
                if original_size_kb > 1024:
                    context['original_size_readable'] = f"{round(original_size_kb/1024, 2)} MB"
                else:
                    context['original_size_readable'] = f"{round(original_size_kb, 2)} KB"
                    
                if compressed_size_kb > 1024:
                    context['compressed_size_readable'] = f"{round(compressed_size_kb/1024, 2)} MB"
                else:
                    context['compressed_size_readable'] = f"{round(compressed_size_kb, 2)} KB"
                
            except Exception as e:
                logger.error(f"Error in compress_pdf_view: {str(e)}")
                context['error'] = f"An error occurred: {str(e)}"
    
    return render(request, 'compress_pdf.html', context)

def download_compressed_pdf(request):
    """Download the compressed PDF file."""
    if 'compressed_pdf' not in request.session or 'output_filename' not in request.session:
        return HttpResponse("No compressed PDF found. Please compress a PDF first.", status=404)
    
    # Get the compressed PDF content from the session
    compressed_content = bytes.fromhex(request.session['compressed_pdf'])
    output_filename = request.session['output_filename']
    
    # Create the HTTP response with the compressed PDF
    response = HttpResponse(compressed_content, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{output_filename}"'
    
    # Clear the session data
    del request.session['compressed_pdf']
    del request.session['output_filename']
    
    return response

class PDFSplitForm(forms.Form):
    pdf_file = forms.FileField(label='Select a PDF file to split')
    start_page = forms.IntegerField(
        label='Start Page', 
        min_value=1, 
        initial=1,
        help_text='First page to include in the split'
    )
    end_page = forms.IntegerField(
        label='End Page', 
        min_value=1,
        help_text='Last page to include in the split'
    )
    output_filename = forms.CharField(
        label='Output filename',
        initial='split.pdf',
        help_text='Name of the split PDF file'
    )

class PDFSplitForm(forms.Form):
    pdf_file = forms.FileField(label='Select a PDF file to split')
    start_page = forms.IntegerField(
        label='Start Page', 
        min_value=1, 
        initial=1,
        help_text='First page to include in the split'
    )
    end_page = forms.IntegerField(
        label='End Page', 
        min_value=1,
        help_text='Last page to include in the split'
    )
    output_filename = forms.CharField(
        label='Output filename',
        initial='split.pdf',
        help_text='Name of the split PDF file'
    )


def split_pdf(pdf_file, start_page, end_page):
    """Split a PDF file from start_page to end_page."""
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as input_temp, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as output_temp:
            
            # Save the uploaded file to a temporary file
            for chunk in pdf_file.chunks():
                input_temp.write(chunk)
            
            input_path = input_temp.name
            output_path = output_temp.name
        
        # Get PDF info
        with open(input_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
        
        # Validate page ranges
        start_page = max(1, min(start_page, total_pages))
        end_page = max(start_page, min(end_page, total_pages))
        
        # Create a PDF writer object
        pdf_writer = PyPDF2.PdfWriter()
        
        # Add specified pages to the writer
        with open(input_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(start_page - 1, end_page):  # PyPDF2 uses 0-based indexing
                pdf_writer.add_page(pdf_reader.pages[page_num])
        
        # Write to the output file
        with open(output_path, 'wb') as output_file:
            pdf_writer.write(output_file)
        
        # Read the split PDF to return it
        with open(output_path, 'rb') as split_pdf:
            split_content = split_pdf.read()
        
        # Clean up temporary files
        os.unlink(input_path)
        os.unlink(output_path)
        
        return split_content
        
    except Exception as e:
        logger.error(f"Error splitting PDF: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
def split_pdf_view(request):
    """View to split a PDF file."""
    context = {
        'form': PDFSplitForm(),
        'error': None,
        'success': False,
        'pdf_info': None,
    }
    
    if request.method == 'POST':
        form = PDFSplitForm(request.POST, request.FILES)
        context['form'] = form
        
        if form.is_valid():
            try:
                pdf_file = request.FILES['pdf_file']
                start_page = form.cleaned_data['start_page']
                end_page = form.cleaned_data['end_page']
                output_filename = form.cleaned_data['output_filename']
                
                # Ensure the filename ends with .pdf
                if not output_filename.lower().endswith('.pdf'):
                    output_filename += '.pdf'
                
                # Check file size
                if pdf_file.size > 100 * 1024 * 1024:  # 100MB limit
                    context['error'] = "File is too large. Maximum file size is 100MB."
                    return render(request, 'split_pdf.html', context)
                
                # Get total pages first
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    for chunk in pdf_file.chunks():
                        temp_file.write(chunk)
                    temp_file_path = temp_file.name
                
                with open(temp_file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                
                os.unlink(temp_file_path)
                
                # Reset file pointer to beginning for processing
                pdf_file.seek(0)
                
                # Validate page range
                if start_page < 1 or start_page > total_pages:
                    context['error'] = f"Invalid start page. Please specify a page between 1 and {total_pages}."
                    return render(request, 'split_pdf.html', context)
                
                if end_page < start_page or end_page > total_pages:
                    context['error'] = f"Invalid end page. Please specify a page between {start_page} and {total_pages}."
                    return render(request, 'split_pdf.html', context)
                
                # Split the PDF
                split_content = split_pdf(pdf_file, start_page, end_page)
                if not split_content:
                    context['error'] = "Failed to split the PDF. Please try again."
                    return render(request, 'split_pdf.html', context)
                
                # Store the split PDF content in the session
                request.session['split_pdf'] = split_content.hex()
                request.session['output_filename'] = output_filename
                
                # Add success info to the context
                context['success'] = True
                context['start_page'] = start_page
                context['end_page'] = end_page
                context['total_pages'] = total_pages
                context['output_filename'] = output_filename
                
            except Exception as e:
                logger.error(f"Error in split_pdf_view: {str(e)}")
                context['error'] = f"An error occurred: {str(e)}"
    
    return render(request, 'split_pdf.html', context)


def download_split_pdf(request):
    """Download the split PDF file."""
    if 'split_pdf' not in request.session or 'output_filename' not in request.session:
        return HttpResponse("No split PDF found. Please split a PDF first.", status=404)
    
    # Get the split PDF content from the session
    split_content = bytes.fromhex(request.session['split_pdf'])
    output_filename = request.session['output_filename']
    
    # Create the HTTP response with the split PDF
    response = HttpResponse(split_content, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{output_filename}"'
    
    # Clear the session data
    del request.session['split_pdf']
    del request.session['output_filename']
    
    return response