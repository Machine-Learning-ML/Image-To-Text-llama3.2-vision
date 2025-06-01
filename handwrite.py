from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load image
image_path = '/home/kosal/AI/studycircle-ai/image/handnote-10.png'
image = Image.open(image_path).convert('RGB')

# Load processor and model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Preprocess and predict
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Detected text:", generated_text)
