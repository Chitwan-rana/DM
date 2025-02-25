from django.shortcuts import render, redirect
from .forms import LabelForm, ImageForm
from .models import Label, Image
from django.conf import settings
import os
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
import pytesseract
from pytesseract import Output

BASE_DIR = settings.BASE_DIR

def classify(request):
    if request.method == 'POST':
        if 'create_label' in request.POST:
            label_form = LabelForm(request.POST)
            if label_form.is_valid():
                label_form.save()
                return redirect('classify')
            else:
                print("Label Form Errors:", label_form.errors)  # Debugging
        elif 'upload_image' in request.POST:
            image_form = ImageForm(request.POST, request.FILES)
            if image_form.is_valid():
                try:
                    image_instance = image_form.save(commit=False)
                    image_instance.save()  # Save the instance explicitly
                    return redirect('classify')
                except Exception as e:
                    print("Error saving image:", e)  # Debugging
            else:
                print("Image Form Errors:", image_form.errors)  # Debugging
    else:
        label_form = LabelForm()
        image_form = ImageForm()
    
    labels = Label.objects.all()
    images = Image.objects.all()
    
    return render(request, 'Doc_Classify.html', {
        'label_form': label_form,
        'image_form': image_form,
        'labels': labels,
        'images': images,
    })

def train_model(request):
    labels = Label.objects.all()
    labels_dict = {label.name: idx for idx, label in enumerate(labels)}
    dataset = []
    for label in labels:
        images = Image.objects.filter(label=label)
        for image in images:
            dataset.append((image.image.path, labels_dict[label.name]))

    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(labels_dict))

    class DocDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img_path, label = self.dataset[idx]
            image, words, boxes = extract_text_and_bboxes(img_path)
            encoded = processor(
                images=image,
                text=words,
                boxes=boxes,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
            )
            for key in encoded.keys():
                encoded[key] = encoded[key].squeeze(0)
            encoded["labels"] = torch.tensor(label)
            return encoded

    train_dataset = DocDataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**{k: v.to(model.device) for k, v in batch.items() if k != "labels"})
            loss = torch.nn.functional.cross_entropy(outputs.logits, batch["labels"].to(model.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    model.save_pretrained(os.path.join(BASE_DIR, "layoutlmv3_finetuned"))
    processor.save_pretrained(os.path.join(BASE_DIR, "layoutlmv3_finetuned"))

    return redirect('classify')

def extract_text_and_bboxes(image_path):
    image = PILImage.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    words = []
    boxes = []
    width, height = image.size
    for i in range(len(data['text'])):
        if data['text'][i].strip():
            words.append(data['text'][i])
            x_min = int((data['left'][i] / width) * 1000)
            y_min = int((data['top'][i] / height) * 1000)
            x_max = int(((data['left'][i] + data['width'][i]) / width) * 1000)
            y_max = int(((data['top'][i] + data['height'][i]) / height) * 1000)
            x_min, x_max = max(0, x_min), min(1000, x_max)
            y_min, y_max = max(0, y_min), min(1000, y_max)
            boxes.append([x_min, y_min, x_max, y_max])
    return image, words, boxes

import os

def predict_label(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')

        # Create temp directory if it doesn't exist
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        image_path = os.path.join(temp_dir, image_file.name)

        # Save uploaded image
        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        # Process and predict
        image, words, boxes = extract_text_and_bboxes(image_path)

        processor = LayoutLMv3Processor.from_pretrained(os.path.join(BASE_DIR, "layoutlmv3_finetuned"))
        model = LayoutLMv3ForSequenceClassification.from_pretrained(os.path.join(BASE_DIR, "layoutlmv3_finetuned"))

        inputs = processor(
            images=image,
            text=words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )

        model.eval()
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        label_idx = torch.argmax(probs).item()
        confidence = probs[0][label_idx].item() * 100

        labels = Label.objects.all()
        labels_dict = {idx: label.name for idx, label in enumerate(labels)}
        label_name = labels_dict.get(label_idx, "Unknown")

        # Delete temp image after processing
        if os.path.exists(image_path):
            os.remove(image_path)

        return render(request, 'Classify_Predict.html', {
            'label_name': label_name,
            'confidence': confidence,
        })

    return redirect('classify')
