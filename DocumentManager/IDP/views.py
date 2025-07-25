from django.shortcuts import render
from django.conf import settings
from django import forms
import tempfile
import PyPDF2
import google.generativeai as genai #type:ignore
import logging
import mimetypes
import io
import fitz  # PyMuPDF
import json
import os
import tempfile
import uuid
import ocrmypdf #type: ignore
from pathlib import Path
from django.http import HttpResponse, FileResponse
import pikepdf  #type:ignore
from pikepdf import Pdf  #type:ignore
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.shortcuts import  redirect
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from pdf2docx import Converter #type:ignore
from django.shortcuts import render
from IDP.models import PDFOperationTracker
from django.db.models import Count
from IDP.utils.track_operations import track_pdf_operation 
from django.utils import timezone
from datetime import timedelta
from django.db.models.functions import TruncDate



logger = logging.getLogger(__name__)
     
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
                    return render(request, 'IDP/Merge/merge_pdfs.html', context)
                
                track_pdf_operation('merge', pdf_file1.name)
                track_pdf_operation('merge', pdf_file2.name)
                
                # Create the HTTP response with the merged PDF
                response = HttpResponse(merged_pdf_content, content_type='application/pdf')
                response['Content-Disposition'] = f'attachment; filename="{output_filename}"'
                return response
            
            except Exception as e:
                logger.error(f"Error in merge_pdfs_view: {str(e)}")
                context['error'] = f"An error occurred: {str(e)}"
    
    return render(request, 'IDP/Merge/merge_pdfs.html', context)


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
                if pdf_file.size > 400 * 1024 * 1024:  # 100MB limit
                    context['error'] = "File is too large. Maximum file size is 100MB."
                    return render(request, 'IDP/Compress/compress_pdf.html', context)
                
                # Compress the PDF
                compression_result = compress_pdf(pdf_file, compression_level)
                if not compression_result:
                    context['error'] = "Failed to compress the PDF. Please try again."
                    track_pdf_operation('compress', pdf_file.name)
                    return render(request, 'IDP/Compress/compress_pdf.html', context)
                
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
    
    return render(request, 'IDP/Compress/compress_pdf.html', context)

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

#SPLIT PDFs

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

                if not output_filename.lower().endswith('.pdf'):
                    output_filename += '.pdf'

                if pdf_file.size > 100 * 1024 * 1024:
                    context['error'] = "File is too large. Maximum file size is 100MB."
                    return render(request, 'IDP/Split/split_pdf.html', context)

                # Get total pages
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    for chunk in pdf_file.chunks():
                        temp_file.write(chunk)
                    temp_file_path = temp_file.name

                with open(temp_file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)

                os.unlink(temp_file_path)
                pdf_file.seek(0)

                if start_page < 1 or start_page > total_pages:
                    context['error'] = f"Invalid start page. Please specify a page between 1 and {total_pages}."
                    return render(request, 'IDP/Split/split_pdf.html', context)

                if end_page < start_page or end_page > total_pages:
                    context['error'] = f"Invalid end page. Please specify a page between {start_page} and {total_pages}."
                    return render(request, 'IDP/Split/split_pdf.html', context)

                split_content = split_pdf(pdf_file, start_page, end_page)
                if not split_content:
                    context['error'] = "Failed to split the PDF. Please try again."
                    return render(request, 'IDP/Split/split_pdf.html', context)
                # ✅ Track the operation
                track_pdf_operation('split', pdf_file.name)

                # Store info in session
                request.session['split_pdf'] = split_content.hex()
                request.session['output_filename'] = output_filename
                request.session['total_pages'] = total_pages
                request.session['start_page'] = start_page
                request.session['end_page'] = end_page

                # Redirect to new download page
                return redirect('download_split_pdf_page')

            except Exception as e:
                logger.error(f"Error in split_pdf_view: {str(e)}")
                context['error'] = f"An error occurred: {str(e)}"

    return render(request, 'IDP/Split/split_pdf.html', context)

def download_split_pdf_view(request):
    """Render the download page for the split PDF."""
    if 'split_pdf' not in request.session or 'output_filename' not in request.session:
        return HttpResponse("No split PDF found. Please split a PDF first.", status=404)

    return render(request, 'IDP/Split/download_split_pdf.html', {
        'output_filename': request.session.get('output_filename'),
        'start_page': request.session.get('start_page'),
        'end_page': request.session.get('end_page'),
        'total_pages': request.session.get('total_pages'),
    })


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

#Convert PDF to Searchable Format

def convert_pdf(request):
    """Process the uploaded PDF and convert to searchable format."""
    if request.method == 'POST' and request.FILES.get('pdf_file'):
        # Get the uploaded file
        pdf_file = request.FILES['pdf_file']

        # Validate file is PDF
        if not pdf_file.name.endswith('.pdf'):
            messages.error(request, "Please upload a valid PDF file.")
            return render(request, 'IDP/Searchable/Searchable.html')

        # Create a unique filename
        unique_id = uuid.uuid4().hex
        original_filename = pdf_file.name
        base_name = os.path.splitext(original_filename)[0]

        # Create temp directory
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_pdfs')
        os.makedirs(temp_dir, exist_ok=True)

        # Save paths
        input_path = os.path.join(temp_dir, f"{unique_id}_input.pdf")
        output_path = os.path.join(temp_dir, f"{unique_id}_searchable.pdf")

        # Save the uploaded file to temp path
        fs = FileSystemStorage(location=temp_dir)
        fs.save(f"{unique_id}_input.pdf", pdf_file)

        try:
            # Run OCR
            ocrmypdf.ocr(input_path, output_path, deskew=True)

            # Save result to session
            request.session['output_pdf_path'] = output_path
            request.session['output_filename'] = f"{base_name}_searchable.pdf"

            # ✅ Track operation
            track_pdf_operation('searchable', pdf_file.name)

            return render(request, 'IDP/Searchable/success.html', {
                'original_filename': original_filename,
                'searchable_filename': f"{base_name}_searchable.pdf"
            })

        except Exception as e:
            # Clean up temp files
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)

            error_msg = str(e)

            # Custom message for already searchable PDFs
            if "page already has text!" in error_msg:
                messages.warning(
                    request,
                    "The uploaded PDF already contains selectable text and does not need OCR. "
                    
                )
            else:
                messages.error(request, f"Error processing PDF: {error_msg}")

            return render(request, 'IDP/Searchable/Searchable.html')

    else:
        return render(request, 'IDP/Searchable/Searchable.html')

def download_searchable_pdf(request):
    """Download the converted searchable PDF"""
    if 'output_pdf_path' in request.session and os.path.exists(request.session['output_pdf_path']):
        output_path = request.session['output_pdf_path']
        filename = request.session['output_filename']
        
        # Serve the file
        response = FileResponse(open(output_path, 'rb'), 
                              content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        

        
        return response
    else:
        messages.error(request, "File not found or processing error occurred.")
        return render(request, 'IDP/Searchable/Searchable.html')
    

# Redaction of PDF
TEMP_DIR = os.path.join(settings.MEDIA_ROOT, "temp")

def extract_highlight_images(pdf_path, terms):
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        for term in terms:
            areas = page.search_for(term)
            for rect in areas:
                highlight = page.add_highlight_annot(rect)
                highlight.update()

        img_path = os.path.join(TEMP_DIR, f"preview_page_{page_num}.png")
        pix = page.get_pixmap()
        pix.save(img_path)
        image_paths.append(f"/media/temp/preview_page_{page_num}.png")

    return image_paths

def redact(request):
    if request.method == "POST" and request.FILES.get("pdf"):
        pdf_file = request.FILES["pdf"]
        terms = [t.strip() for t in request.POST.get("terms", "").split(",") if t.strip()]
        session_id = str(uuid.uuid4())
        temp_pdf_path = os.path.join(TEMP_DIR, f"{session_id}.pdf")
        os.makedirs(TEMP_DIR, exist_ok=True)
        default_storage.save(temp_pdf_path, pdf_file)
        # ✅ Track the operation
        track_pdf_operation('redact', pdf_file.name)

        request.session['temp_pdf'] = temp_pdf_path
        request.session['terms'] = terms

        # Create previews
        images = extract_highlight_images(temp_pdf_path, terms)
        return render(request, "IDP/Redaction/Preview.html", {"images": images})

    return render(request, "IDP/Redaction/Redact.html")

def redact_confirm(request):
    temp_pdf = request.session.get('temp_pdf')
    terms = request.session.get('terms', [])

    if not temp_pdf or not os.path.exists(temp_pdf):
        return render(request, "IDP/Redaction/Redact.html", {"error": "No session data found."})

    doc = fitz.open(temp_pdf)

    # Manual redaction boxes from canvas
    if request.method == "POST":
        boxes_json = request.POST.get("boxes_json")
        if boxes_json:
            drawings = json.loads(boxes_json)
            for page_index, boxes in drawings.items():
                page = doc[int(page_index)]
                pix = page.get_pixmap()
                scale_x = page.rect.width / pix.width
                scale_y = page.rect.height / pix.height

                for b in boxes:
                    rect = fitz.Rect(
                        b["x"] * scale_x,
                        b["y"] * scale_y,
                        (b["x"] + b["w"]) * scale_x,
                        (b["y"] + b["h"]) * scale_y
                    )
                    page.add_redact_annot(rect, fill=(0, 0, 0))

        # Apply search-based redaction too (if needed)
        for page in doc:
            for term in terms:
                for rect in page.search_for(term):
                    page.add_redact_annot(rect, fill=(0, 0, 0))
            page.apply_redactions()

    output = io.BytesIO()
    doc.save(output)
    output.seek(0)
    return FileResponse(output, as_attachment=True, filename="redacted.pdf")

# Convert PDF to DOCX

def convert_pdf_view(request):
    """Display the PDF to Word conversion page"""
    return render(request, 'IDP/PDF_TO_DOCX/pdf_to_docx.html')

@csrf_exempt
def upload_pdf(request):
    """Handle PDF upload and return URL for preview"""
    if request.method == 'POST' and request.FILES.get('pdf_file'):
        try:
            pdf_file = request.FILES['pdf_file']
            
            # Validate file type
            if not pdf_file.name.lower().endswith('.pdf'):
                return JsonResponse({'error': 'Please upload a valid PDF file'}, status=400)
            
            # Generate unique filename
            unique_id = uuid.uuid4().hex
            filename = f"{unique_id}_{pdf_file.name}"
            
            # Create media directory if it doesn't exist
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save the file to disk for preview
            pdf_path = os.path.join('uploads', filename)
            full_path = os.path.join(settings.MEDIA_ROOT, pdf_path)
            
            with open(full_path, 'wb+') as destination:
                for chunk in pdf_file.chunks():
                    destination.write(chunk)
                    # ✅ Track the operation
                track_pdf_operation('pdf_to_docx', pdf_file.name)
            
            # Generate URL for preview
            pdf_url = f"{settings.MEDIA_URL}{pdf_path}"
            
            return JsonResponse({
                'success': True,
                'pdf_url': pdf_url,
                'pdf_id': unique_id,
                'pdf_name': pdf_file.name,
                'pdf_path': pdf_path
            })
            
        except Exception as e:
            return JsonResponse({'error': f'Upload failed: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def convert_pdf_to_docx(request):
    """Handle PDF to Word conversion"""
    if request.method == 'POST':
        try:
            pdf_path = request.POST.get('pdf_path')
            
            if not pdf_path:
                return JsonResponse({'error': 'No PDF file specified'}, status=400)
            
            # Generate path for the output DOCX file
            pdf_filename = os.path.basename(pdf_path)
            output_filename = os.path.splitext(pdf_filename)[0].split('_', 1)[1] + '.docx'
            unique_id = uuid.uuid4().hex
            docx_filename = f"{unique_id}_{output_filename}"
            
            # Create converted directory if it doesn't exist
            converted_dir = os.path.join(settings.MEDIA_ROOT, 'converted')
            os.makedirs(converted_dir, exist_ok=True)
            
            # Set paths for conversion
            pdf_full_path = os.path.join(settings.MEDIA_ROOT, pdf_path)
            docx_path = os.path.join('converted', docx_filename)
            docx_full_path = os.path.join(settings.MEDIA_ROOT, docx_path)
            
            # Convert PDF to DOCX
            cv = Converter(pdf_full_path)
            cv.convert(docx_full_path)
            cv.close()
            
            # Generate URL for the converted file
            docx_url = f"{settings.MEDIA_URL}{docx_path}"
            
            return JsonResponse({
                'success': True,
                'docx_url': docx_url,
                'docx_filename': output_filename
            })
            
        except Exception as e:
            return JsonResponse({'error': f'Conversion failed: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


# Statistics View




def stats_view(request):
    filter_type = request.GET.get('filter', 'all')
    now = timezone.now()

    if filter_type == 'today':
        start_date = now.replace(hour=0, minute=0, second=0)
    elif filter_type == 'week':
        start_date = now - timedelta(days=7)
    elif filter_type == 'month':
        start_date = now - timedelta(days=30)
    else:
        start_date = None

    if start_date:
        operations = PDFOperationTracker.objects.filter(timestamp__gte=start_date)
    else:
        operations = PDFOperationTracker.objects.all()

    total_documents = operations.count()

    # Bar & Pie chart data
    stats_by_type = (
        operations.values('operation_type')
        .annotate(count=Count('id'))
        .order_by('-count')
    )

    operation_display = dict(PDFOperationTracker.OPERATION_CHOICES)
    for stat in stats_by_type:
        stat['operation_label'] = operation_display.get(stat['operation_type'], stat['operation_type'])

    # Line chart data (trends over time)
    date_counts = (
        operations.annotate(date=TruncDate('timestamp'))
        .values('date')
        .annotate(count=Count('id'))
        .order_by('date')
    )
    trend_labels = [entry['date'].strftime('%Y-%m-%d') for entry in date_counts]
    trend_counts = [entry['count'] for entry in date_counts]

    context = {
        'total_documents': total_documents,
        'stats_by_type': stats_by_type,
        'operation_labels': [s['operation_label'] for s in stats_by_type],
        'operation_counts': [s['count'] for s in stats_by_type],
        'filter_type': filter_type,
        'trend_labels': json.dumps(trend_labels),
        'trend_counts': json.dumps(trend_counts),
    }

    return render(request, 'IDP/stats.html', context)
