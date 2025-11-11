🎯 COMPLETE LoRA Training Pipeline Guide
Architecture Overview
Airtable Form Submit
    ↓
Job Intake & Validation
    ↓
Workflow 1: Path Normalizer & Dataset Inspector
    ↓
    ├─ IF needs_preprocessing → Workflow 2: Image Preprocessor
    │                              ↓
    └─ ELSE ────────────────────→ Workflow 3: Image Captioner
                                      ↓
                                 Workflow 4: Config Generator
                                      ↓
                                 Workflow 5: Upload to Backblaze
                                      ↓
                                 Workflow 6: Launch RunPod Training
                                      ↓
                                 Workflow 7: Monitor Training (WandB)
                                      ↓
                                 Workflow 8: Post-Training Analysis (VLM)
                                      ↓
                                 Workflow 9: Slack Notification

📋 Airtable Form Configuration
Form Fields (User-Facing):
Required Fields:
├─ dataset_path (long text)
│  └─ Placeholder: "D:/photos/bottles" or "//server/share/images"
├─ lora_token (text, default: "sks_fragrancebottle")
│  └─ Description: "Trigger word for your LoRA (3-10 alphanumeric characters)"
├─ model_base (single select)
│  └─ Options: SDXL, SD1.5, Flux, flux_kontext, wan22, hunyuan, qwen_image
└─ training_type (single select)
   └─ Options: character, object, style, scene, motion_lora

Optional Advanced Fields:
├─ custom_config (long text, JSON format)
│  └─ Description: "Advanced: Override config template with custom JSON"
└─ validation_prompts (long text)
   └─ Description: "One prompt per line for validation images"
   └─ Placeholder: "sks_fragrancebottle on marble counter\nsks_fragrancebottle in natural light"
Backend Fields (Auto-Populated by Workflows):
Job Tracking:
├─ job_id (formula: "job_" & CREATED_TIME())
├─ created_at (created time)
├─ updated_at (last modified time)
└─ status (single select)
   Options: pending, validated, preprocessing, preprocessed, 
            captioning, captioned, config_ready, uploaded, 
            training, training_complete, analyzing, complete, failed

Path Management:
├─ normalized_path (text)
├─ processed_dir (text)
└─ upload_dir (text)

Dataset Info:
├─ image_count (number)
├─ total_size_mb (number)
├─ needs_preprocessing (checkbox)
└─ needs_captioning (checkbox)

Processing Reports (JSON):
├─ inspection_report (long text)
├─ preprocessing_report (long text)
└─ captioning_report (long text)

Captioning Progress:
├─ captioning_progress (percent)
├─ captions_completed (number)
└─ caption_accuracy (percent)

Cloud Storage:
├─ backblaze_dataset_url (URL)
├─ backblaze_config_url (URL)
├─ backblaze_output_url (URL)
└─ upload_complete_at (date)

Training:
├─ runpod_job_id (text)
├─ wandb_url (URL)
├─ training_started_at (date)
├─ training_completed_at (date)
├─ training_progress (percent)
├─ current_step (text, e.g. "1250/2776")
├─ current_loss (number)
└─ eta (text, e.g. "1h 15m")

Results:
├─ best_model_url (URL)
├─ best_model_score (number)
└─ analysis_report_url (URL)

Costs:
├─ estimated_cost (currency, USD)
├─ actual_cost (currency, USD)
└─ captioning_cost (currency, USD)

Error Handling:
├─ error_log (long text)
└─ retry_count (number, default: 0)

🚀 Workflow 0: Job Intake & Validation
Trigger: Airtable webhook on form submission
javascript// n8n Workflow: "0_Job_Intake"

1. Receive Webhook from Airtable
   const formData = $input.item.json;
   
2. Generate job_id
   // Airtable formula already created this
   job_id = formData.job_id;
   
3. Validate Required Fields
   if (!formData.dataset_path) {
     throw new Error("Dataset path is required");
   }
   
   if (!formData.lora_token) {
     // Set default
     lora_token = "sks_fragrancebottle";
   } else {
     lora_token = formData.lora_token;
     // Validate format
     if (!/^[a-zA-Z0-9]{3,10}$/.test(lora_token)) {
       throw new Error("LoRA token must be 3-10 alphanumeric characters");
     }
   }
   
   if (!formData.model_base) {
     throw new Error("Model type is required");
   }
   
   if (!formData.training_type) {
     training_type = "object"; // Default
   } else {
     training_type = formData.training_type;
   }

4. Initialize Job
   UPDATE Airtable record:
   - status = "pending"
   - lora_token = lora_token (if was default)
   - training_type = training_type (if was default)
   - estimated_cost = 2.70 (initial estimate)
   
5. Send Slack Notification
   POST to Slack webhook:
   {
     "channel": "#lora-training-bot",
     "text": "🚀 New training job started",
     "blocks": [
       {
         "type": "section",
         "text": {
           "type": "mrkdwn",
           "text": `*Job ID:* ${job_id}\n*LoRA Token:* ${lora_token}\n*Model:* ${formData.model_base}\n*Type:* ${training_type}\n*Status:* Validating dataset...`
         }
       }
     ]
   }

6. Trigger Workflow 1
   // Call webhook for next workflow
   POST to n8n webhook:
   {
     "job_id": job_id,
     "dataset_path": formData.dataset_path,
     "lora_token": lora_token,
     "model_base": formData.model_base,
     "training_type": training_type,
     "custom_config": formData.custom_config,
     "validation_prompts": formData.validation_prompts
   }

🔍 Workflow 1: Path Normalizer & Dataset Inspector
Purpose: Validate dataset, detect preprocessing/captioning needs
python# n8n Workflow: "1_Path_Normalizer"
# Language: Python (Code node)

import os
import random
from pathlib import Path
from PIL import Image
import json

# INPUT from previous workflow
job_id = $input.item.json.job_id
dataset_path = $input.item.json.dataset_path
lora_token = $input.item.json.lora_token
model_base = $input.item.json.model_base
training_type = $input.item.json.training_type

# ===================================
# 1. NORMALIZE PATH FORMAT
# ===================================
# Convert Windows backslashes to forward slashes
if "\\" in dataset_path:
    normalized_path = dataset_path.replace("\\", "/")
else:
    normalized_path = dataset_path

# ===================================
# 2. VALIDATE PATH EXISTS
# ===================================
source = Path(normalized_path)

if not source.exists():
    raise Exception(f"Path does not exist: {normalized_path}")

if not source.is_dir():
    raise Exception(f"Path is not a directory: {normalized_path}")

# ===================================
# 3. COUNT & VALIDATE IMAGES
# ===================================
image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
image_files = [
    f for f in source.rglob('*') 
    if f.suffix.lower() in image_extensions
]

image_count = len(image_files)

if image_count < 4:
    raise Exception(f"Minimum 4 images required, found {image_count}")

if image_count > 10000:
    raise Exception(f"Maximum 10,000 images exceeded, found {image_count}")

# Calculate total size
total_size_bytes = sum(f.stat().st_size for f in image_files)
total_size_mb = round(total_size_bytes / (1024 * 1024), 2)

# ===================================
# 4. INSPECT DATASET (Sample Analysis)
# ===================================
sample_size = min(10, image_count)
sample = random.sample(image_files, sample_size)

resolutions = []
has_txt_count = 0
sample_captions = []

for img_path in sample:
    # Check resolution
    try:
        img = Image.open(img_path)
        width, height = img.size
        resolutions.append((width, height))
        img.close()
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        continue
    
    # Check for caption file
    txt_path = img_path.with_suffix('.txt')
    if txt_path.exists():
        has_txt_count += 1
        # Read sample caption
        with open(txt_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
            sample_captions.append(caption)

# ===================================
# 5. DETECTION LOGIC
# ===================================

# Analyze resolutions
unique_resolutions = list(set(resolutions))
avg_width = sum(r[0] for r in resolutions) / len(resolutions)
avg_height = sum(r[1] for r in resolutions) / len(resolutions)

# Calculate aspect ratio variance
aspect_ratios = [w/h for w, h in resolutions]
aspect_variance = max(aspect_ratios) - min(aspect_ratios)

# Determine target resolution based on model
target_resolutions = {
    "SD1.5": 512,
    "SDXL": 1024,
    "Flux": 1024,
    "flux_kontext": 1024,
    "wan22": 512,
    "hunyuan": 1024,
    "qwen_image": 1024
}
target_res = target_resolutions.get(model_base, 1024)

# DETECT: Needs Preprocessing?
needs_preprocessing = False
preprocessing_reason = ""

if len(unique_resolutions) > 1:
    needs_preprocessing = True
    preprocessing_reason = f"Mixed resolutions detected: {len(unique_resolutions)} unique sizes"
elif avg_width != target_res or avg_height != target_res:
    needs_preprocessing = True
    preprocessing_reason = f"Images not at target resolution ({target_res}px)"

# DETECT: Needs Captioning?
needs_captioning = False
captioning_reason = ""

caption_coverage = has_txt_count / sample_size

if caption_coverage < 0.8:
    needs_captioning = True
    captioning_reason = f"Only {int(caption_coverage * 100)}% of images have captions"
else:
    # Check if captions contain LoRA token
    token_found = sum(1 for cap in sample_captions if lora_token.lower() in cap.lower())
    token_coverage = token_found / len(sample_captions) if sample_captions else 0
    
    if token_coverage < 0.8:
        needs_captioning = True
        captioning_reason = f"Captions exist but LoRA token '{lora_token}' missing in {int((1-token_coverage)*100)}%"

# ===================================
# 6. GENERATE INSPECTION REPORT
# ===================================
inspection_report = {
    "image_count": image_count,
    "total_size_mb": total_size_mb,
    "sample_size": sample_size,
    "unique_resolutions": len(unique_resolutions),
    "avg_resolution": [int(avg_width), int(avg_height)],
    "target_resolution": target_res,
    "aspect_variance": round(aspect_variance, 3),
    "caption_coverage": round(caption_coverage, 2),
    "needs_preprocessing": needs_preprocessing,
    "preprocessing_reason": preprocessing_reason,
    "needs_captioning": needs_captioning,
    "captioning_reason": captioning_reason,
    "sample_resolutions": resolutions,
    "sample_captions": sample_captions[:3]  # First 3 captions
}

# ===================================
# 7. OUTPUT
# ===================================
return {
    "job_id": job_id,
    "normalized_path": normalized_path,
    "image_count": image_count,
    "total_size_mb": total_size_mb,
    "needs_preprocessing": needs_preprocessing,
    "needs_captioning": needs_captioning,
    "inspection_report": json.dumps(inspection_report, indent=2),
    "status": "validated"
}
Then, in n8n:
javascript// Update Airtable node
UPDATE record WHERE job_id = $json.job_id:
- normalized_path = $json.normalized_path
- image_count = $json.image_count
- total_size_mb = $json.total_size_mb
- needs_preprocessing = $json.needs_preprocessing
- needs_captioning = $json.needs_captioning
- inspection_report = $json.inspection_report
- status = "validated"

// IF node: Route to next workflow
IF $json.needs_preprocessing == true:
  → Trigger Workflow 2
ELSE IF $json.needs_captioning == true:
  → Trigger Workflow 3
ELSE:
  → Trigger Workflow 4

🖼️ Workflow 2: Image Preprocessor
Purpose: Resize, crop, and organize images into aspect buckets
python# n8n Workflow: "2_Image_Preprocessor"
# Language: Python (Code node)

import os
from pathlib import Path
from PIL import Image
import json
import math

# INPUT
job_id = $input.item.json.job_id
normalized_path = $input.item.json.normalized_path
lora_token = $input.item.json.lora_token
model_base = $input.item.json.model_base
image_count = $input.item.json.image_count

# ===================================
# 1. DETERMINE TARGET RESOLUTION
# ===================================
target_resolutions = {
    "SD1.5": 512,
    "SDXL": 1024,
    "Flux": 1024,
    "flux_kontext": 1024,
    "wan22": 512,
    "hunyuan": 1024,
    "qwen_image": 1024
}
target_res = target_resolutions.get(model_base, 1024)

# ===================================
# 2. CREATE OUTPUT DIRECTORY
# ===================================
# Format: {normalized_path}/{job_id}_{lora_token}_{model_base}_v1/
version = "v1"  # Could be incremented if job is rerun
output_folder_name = f"{job_id}_{lora_token}_{model_base}_{version}"
processed_dir = Path(normalized_path) / output_folder_name

os.makedirs(processed_dir, exist_ok=True)

# ===================================
# 3. ANALYZE FOR ASPECT BUCKETING
# ===================================
source = Path(normalized_path)
image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
image_files = [
    f for f in source.rglob('*') 
    if f.suffix.lower() in image_extensions
    and output_folder_name not in str(f)  # Don't process already processed files
]

aspect_ratios = []
for img_path in image_files:
    try:
        img = Image.open(img_path)
        ratio = img.width / img.height
        aspect_ratios.append(ratio)
        img.close()
    except:
        continue

# Calculate variance
import numpy as np
variance = np.std(aspect_ratios)

# Decide on bucketing
use_aspect_bucketing = variance > 0.3

if use_aspect_bucketing:
    # Generate buckets based on target resolution
    if target_res == 512:
        buckets = ["512x512", "512x768", "768x512"]
    elif target_res == 1024:
        buckets = ["1024x1024", "1024x1536", "1536x1024"]
    else:
        buckets = [f"{target_res}x{target_res}"]
else:
    buckets = [f"{target_res}x{target_res}"]

# Create bucket directories
for bucket in buckets:
    os.makedirs(processed_dir / bucket, exist_ok=True)

# ===================================
# 4. HELPER FUNCTIONS
# ===================================
def smart_crop(img, target_w, target_h):
    """Center crop with optional face detection"""
    orig_w, orig_h = img.size
    orig_ratio = orig_w / orig_h
    target_ratio = target_w / target_h
    
    if orig_ratio > target_ratio:
        # Image is wider - crop width
        new_w = int(orig_h * target_ratio)
        left = (orig_w - new_w) // 2
        img_cropped = img.crop((left, 0, left + new_w, orig_h))
    else:
        # Image is taller - crop height
        new_h = int(orig_w / target_ratio)
        top = (orig_h - new_h) // 2
        img_cropped = img.crop((0, top, orig_w, top + new_h))
    
    return img_cropped

def assign_to_bucket(img, buckets):
    """Assign image to closest aspect ratio bucket"""
    img_ratio = img.width / img.height
    
    best_bucket = buckets[0]
    best_diff = float('inf')
    
    for bucket in buckets:
        bucket_w, bucket_h = map(int, bucket.split('x'))
        bucket_ratio = bucket_w / bucket_h
        diff = abs(img_ratio - bucket_ratio)
        
        if diff < best_diff:
            best_diff = diff
            best_bucket = bucket
    
    return best_bucket

# ===================================
# 5. PROCESS IMAGES
# ===================================
processed_count = 0
failed_files = []
bucket_counts = {bucket: 0 for bucket in buckets}

for img_path in image_files:
    try:
        # Read image
        img = Image.open(img_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Determine bucket
        if use_aspect_bucketing:
            bucket = assign_to_bucket(img, buckets)
        else:
            bucket = buckets[0]
        
        # Get target dimensions
        bucket_w, bucket_h = map(int, bucket.split('x'))
        
        # Crop and resize
        img_cropped = smart_crop(img, bucket_w, bucket_h)
        img_resized = img_cropped.resize((bucket_w, bucket_h), Image.LANCZOS)
        
        # Save as PNG
        output_filename = img_path.stem + ".png"
        output_path = processed_dir / bucket / output_filename
        img_resized.save(output_path, "PNG", optimize=True)
        
        img.close()
        img_resized.close()
        
        processed_count += 1
        bucket_counts[bucket] += 1
        
    except Exception as e:
        failed_files.append({
            "file": str(img_path),
            "error": str(e)
        })

# ===================================
# 6. QUALITY CHECK
# ===================================
if processed_count != len(image_files):
    print(f"Warning: Processed {processed_count}/{len(image_files)} images")

# ===================================
# 7. GENERATE REPORT
# ===================================
preprocessing_report = {
    "original_count": len(image_files),
    "processed_count": processed_count,
    "failed_count": len(failed_files),
    "used_aspect_bucketing": use_aspect_bucketing,
    "aspect_variance": float(variance),
    "buckets": bucket_counts,
    "target_resolution": target_res,
    "output_format": "PNG",
    "failed_files": failed_files
}

# ===================================
# 8. OUTPUT
# ===================================
return {
    "job_id": job_id,
    "processed_dir": str(processed_dir),
    "processed_count": processed_count,
    "preprocessing_report": json.dumps(preprocessing_report, indent=2),
    "status": "preprocessed"
}
Then, in n8n:
javascript// Update Airtable
UPDATE record WHERE job_id = $json.job_id:
- processed_dir = $json.processed_dir
- processed_image_count = $json.processed_count
- preprocessing_report = $json.preprocessing_report
- status = "preprocessed"

// Trigger Workflow 3
POST to n8n webhook for Workflow 3

📝 Workflow 3: Image Captioner
Purpose: Generate captions with LoRA token for all images
python# n8n Workflow: "3_Image_Captioner"
# Language: Python (Code node)

import os
from pathlib import Path
from PIL import Image
import base64
import json
import time
from openai import OpenAI

# INPUT
job_id = $input.item.json.job_id
normalized_path = $input.item.json.normalized_path
processed_dir = $input.item.json.get('processed_dir')  # May be None
needs_preprocessing = $input.item.json.needs_preprocessing
lora_token = $input.item.json.lora_token
training_type = $input.item.json.training_type
model_base = $input.item.json.model_base

# OpenAI client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# ===================================
# 1. DETERMINE INPUT DIRECTORY
# ===================================
if needs_preprocessing and processed_dir:
    input_dir = Path(processed_dir)
else:
    input_dir = Path(normalized_path)

# ===================================
# 2. COLLECT ALL IMAGES
# ===================================
image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
image_files = [
    f for f in input_dir.rglob('*') 
    if f.suffix.lower() in image_extensions
]

total_images = len(image_files)

# ===================================
# 3. LOAD PROMPT TEMPLATE
# ===================================
prompts = {
    "character": "Describe this image for character LoRA training. Emphasize pose and positioning in space, lighting direction/quality/intensity and shadows, camera–subject relationship (angle, distance, focal-length feel, depth of field, framing), and surroundings. Do not list facial identity details or unique identifiers; avoid brand/style words.",
    
    "object": "Describe this image for object LoRA training. Focus on environment and optics: lighting direction/quality/intensity and shadows/reflections, camera–subject relationship (angle, distance, depth of field, framing), background context and support surfaces, and general color/material cues visible without naming or identifying the object specifically. Avoid brand names, unique labels, or subjective adjectives.",
    
    "style": "Describe the overall scene succinctly, then the dominant color palette and the quality of light (direction, softness/hardness, contrast, time-of-day feel). Mention composition tendencies or rendering approach if evident (e.g., minimal, high-contrast, soft-gradation). End the caption with: 'in the style of {{STYLE}}'. Do not identify brands or subjects.",
    
    "scene": "Describe the setting and visible elements with attention to spatial layout and depth cues, composition/framing lines, lighting direction/quality/intensity, and palette. Include materials/textures of major surfaces. Avoid identities, brands, and subjective adjectives.",
    
    "motion_lora": "Describe the motion and action in this image. Focus on the type of movement, speed, direction, and any dynamic elements. Describe body positioning, gesture, and flow. Note whether the motion is captured sharply or with motion blur, and describe the energy or intensity of the action."
}

base_prompt = prompts.get(training_type, prompts["object"])
final_prompt = f"IMPORTANT: Begin your description with the trigger token '{lora_token}'. Then provide the rest of the description.\n\n{base_prompt}"

# ===================================
# 4. BATCH PROCESSING WITH RATE LIMITING
# ===================================
batch_size = 50
batches = [image_files[i:i+batch_size] for i in range(0, total_images, batch_size)]

completed = 0
failed = []
caption_lengths = []
token_valid_count = 0

for batch_idx, batch in enumerate(batches):
    print(f"Processing batch {batch_idx + 1}/{len(batches)}...")
    
    for img_path in batch:
        try:
            # Read and encode image
            with open(img_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Determine MIME type
            ext = img_path.suffix.lower().lstrip('.')
            mime = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
            data_uri = f"data:{mime};base64,{img_data}"
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri,
                                "detail": "high"
                            }
                        }
                    ]
                }],
                max_tokens=300
            )
            
            caption = response.choices[0].message.content.strip()
            
            # Validate LoRA token presence
            if lora_token.lower() in caption.lower():
                token_valid_count += 1
            
            # Track caption length
            caption_lengths.append(len(caption.split()))
            
            # Save caption file
            txt_path = img_path.with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            completed += 1
            
        except Exception as e:
            failed.append({
                "file": str(img_path),
                "error": str(e)
            })
    
    # Rate limit buffer (6 seconds between batches)
    if batch_idx < len(batches) - 1:
        time.sleep(6)
    
    # Update progress in Airtable (optional, can be done via HTTP request node)
    progress_pct = (completed / total_images) * 100
    print(f"Progress: {progress_pct:.1f}% ({completed}/{total_images})")

# ===================================
# 5. QUALITY VALIDATION
# ===================================
token_accuracy = token_valid_count / total_images if total_images > 0 else 0
avg_caption_length = sum(caption_lengths) / len(caption_lengths) if caption_lengths else 0

# Calculate cost (OpenAI pricing: ~$0.002 per image for gpt-4o)
cost_per_image = 0.002
captioning_cost = round(total_images * cost_per_image, 2)

# ===================================
# 6. GENERATE REPORT
# ===================================
captioning_report = {
    "total_images": total_images,
    "successful": completed,
    "failed": len(failed),
    "lora_token": lora_token,
    "lora_token_accuracy": round(token_accuracy, 3),
    "avg_caption_length": round(avg_caption_length, 1),
    "captioning_cost_usd": captioning_cost,
    "failed_files": failed
}

# ===================================
# 7. OUTPUT
# ===================================
return {
    "job_id": job_id,
    "captions_completed": completed,
    "caption_accuracy": round(token_accuracy * 100, 1),
    "captioning_cost": captioning_cost,
    "captioning_report": json.dumps(captioning_report, indent=2),
    "status": "captioned"
}
Then, in n8n:
javascript// Update Airtable
UPDATE record WHERE job_id = $json.job_id:
- captions_completed = $json.captions_completed
- caption_accuracy = $json.caption_accuracy
- captioning_cost = $json.captioning_cost
- captioning_report = $json.captioning_report
- status = "captioned"

// Trigger Workflow 4
POST to n8n webhook for Workflow 4

⚙️ Workflow 4: Config Generator
Purpose: Generate training config files from templates
python# n8n Workflow: "4_Config_Generator"
# Language: Python (Code node)

import os
import json
from pathlib import Path

# INPUT
job_id = $input.item.json.job_id
normalized_path = $input.item.json.normalized_path
processed_dir = $input.item.json.get('processed_dir')
needs_preprocessing = $input.item.json.needs_preprocessing
image_count = $input.item.json.image_count
lora_token = $input.item.json.lora_token
model_base = $input.item.json.model_base
training_type = $input.item.json.training_type
custom_config = $input.item.json.get('custom_config')  # Optional
validation_prompts = $input.item.json.get('validation_prompts')  # Optional

# ===================================
# 1. LOAD TEMPLATE
# ===================================
# Template path format: /workspace/templates/{model_base}_{training_type}_config.json
template_path = f"/workspace/templates/{model_base}_{training_type}_config.json"

if os.path.exists(template_path):
    with open(template_path, 'r') as f:
        config_template = json.load(f)
else:
    # Fallback to default template
    template_path = f"/workspace/templates/{model_base}_default_config.json"
    with open(template_path, 'r') as f:
        config_template = json.load(f)

# ===================================
# 2. MERGE CUSTOM CONFIG (if provided)
# ===================================
if custom_config:
    try:
        custom_config_json = json.loads(custom_config)
        # Merge: custom values override template
        config = {**config_template, **custom_config_json}
        config_source = "custom"
    except:
        config = config_template
        config_source = "template"
else:
    config = config_template
    config_source = "template"

# ===================================
# 3. FILL DYNAMIC VALUES
# ===================================
# Instance prompt (LoRA token)
config["instance_prompt"] = lora_token

# Output directory (will be in RunPod container)
config["output_dir"] = f"/output/{job_id}"
config["logging_dir"] = f"/logs/{job_id}"

# ===================================
# 4. CONFIGURE DATA PATHS
# ===================================
# Determine dataset directory
if needs_preprocessing:
    dataset_dir = Path(processed_dir)
else:
    dataset_dir = Path(normalized_path)

# List bucket directories (if aspect bucketing was used)
bucket_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

if bucket_dirs:
    # Multiple buckets
    datasets_config = []
    for bucket_dir in bucket_dirs:
        bucket_name = bucket_dir.name
        try:
            resolution = int(bucket_name.split('x')[0])
        except:
            resolution = 1024  # Default
        
        datasets_config.append({
            "id": bucket_name,
            "type": "local",
            "instance_data_dir": f"/dataset/{bucket_name}",
            "caption_strategy": "filename",
            "resolution": resolution,
            "minimum_image_size": resolution
        })
    
    config["data_backend_config"] = {"datasets": datasets_config}
else:
    # Single directory
    target_res = {
        "SD1.5": 512,
        "SDXL": 1024,
        "Flux": 1024
    }.get(model_base, 1024)
    
    config["data_backend_config"] = {
        "datasets": [{
            "id": "main",
            "type": "local",
            "instance_data_dir": "/dataset",
            "caption_strategy": "filename",
            "resolution": target_res,
            "minimum_image_size": target_res
        }]
    }

# ===================================
# 5. GENERATE VALIDATION PROMPTS
# ===================================
if validation_prompts:
    # User provided prompts
    prompts = [p.strip() for p in validation_prompts.split('\n') if p.strip()]
else:
    # Auto-generate from common patterns
    # Simple approach: use lora_token with common contexts
    prompts = [
        f"{lora_token}",
        f"{lora_token} close-up",
        f"{lora_token} in natural light",
        f"{lora_token} on white background"
    ]

config["validation_prompt"] = prompts
config["num_validation_images"] = min(4, len(prompts))

# ===================================
# 6. WANDB CONFIGURATION
# ===================================
config["report_to"] = "wandb"
config["wandb_project"] = "lora-training"
config["wandb_run_name"] = f"{job_id}_{lora_token}"

# ===================================
# 7. CREATE CONFIG DIRECTORY
# ===================================
# Save configs in normalized_path (user's location)
config_dir = dataset_dir / "configs"
os.makedirs(config_dir, exist_ok=True)

# Save main config
with open(config_dir / "config.json", 'w') as f:
    json.dump(config, f, indent=2)

# Save multidatabackend config
with open(config_dir / "multidatabackend.json", 'w') as f:
    json.dump(config["data_backend_config"], f, indent=2)

# Save user prompt library
with open(config_dir / "user_prompt_library.json", 'w') as f:
    json.dump({"prompts": prompts}, f, indent=2)

# ===================================
# 8. ESTIMATE TRAINING TIME/COST
# ===================================
# Simple estimation (can be refined)
steps_per_epoch = image_count
estimated_epochs = config.get("num_train_epochs", 30)
batch_size = config.get("train_batch_size", 4)
gradient_accumulation = config.get("gradient_accumulation_steps", 2)

total_steps = (image_count * estimated_epochs) / (batch_size * gradient_accumulation)

# Assume ~1.2 seconds per step on 4090
estimated_time_hours = (total_steps * 1.2) / 3600

# Cost: $0.80/hour for 4090
estimated_cost = round(estimated_time_hours * 0.80, 2)

# ===================================
# 9. OUTPUT
# ===================================
return {
    "job_id": job_id,
    "config_dir": str(config_dir),
    "config_source": config_source,
    "estimated_training_time": f"{estimated_time_hours:.1f}h",
    "estimated_cost": estimated_cost,
    "status": "config_ready"
}
Then, in n8n:
javascript// Update Airtable
UPDATE record WHERE job_id = $json.job_id:
- config_dir = $json.config_dir
- estimated_cost = $json.estimated_cost
- status = "config_ready"

// Trigger Workflow 5
POST to n8n webhook for Workflow 5

☁️ Workflow 5: Upload to Backblaze
Purpose: Upload dataset + configs to cloud storage
python# n8n Workflow: "5_Upload_Backblaze"
# Language: Python (Code node)

import os
from pathlib import Path
from b2sdk.v2 import B2Api, InMemoryAccountInfo
import json

# INPUT
job_id = $input.item.json.job_id
normalized_path = $input.item.json.normalized_path
processed_dir = $input.item.json.get('processed_dir')
config_dir = $input.item.json.config_dir
needs_preprocessing = $input.item.json.needs_preprocessing
lora_token = $input.item.json.lora_token
model_base = $input.item.json.model_base

# ===================================
# 1. DETERMINE WHAT TO UPLOAD
# ===================================
if needs_preprocessing:
    dataset_dir = Path(processed_dir)
else:
    dataset_dir = Path(normalized_path)

# ===================================
# 2. AUTHENTICATE WITH BACKBLAZE
# ===================================
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account(
    "production",
    os.environ['B2_APPLICATION_KEY_ID'],
    os.environ['B2_APPLICATION_KEY']
)

# Get bucket
bucket = b2_api.get_bucket_by_name("training-datasets")

# ===================================
# 3. DEFINE REMOTE PATH
# ===================================
# Format: jobs/{job_id}_{lora_token}/
remote_base = f"jobs/{job_id}_{lora_token}"

# ===================================
# 4. UPLOAD DATASET
# ===================================
print("Uploading dataset...")
uploaded_files = 0

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        local_file = os.path.join(root, file)
        relative_path = os.path.relpath(local_file, dataset_dir.parent)
        remote_file = f"{remote_base}/dataset/{relative_path}"
        
        bucket.upload_local_file(
            local_file=local_file,
            file_name=remote_file.replace('\\', '/')
        )
        uploaded_files += 1
        
        if uploaded_files % 10 == 0:
            print(f"Uploaded {uploaded_files} files...")

# ===================================
# 5. UPLOAD CONFIGS
# ===================================
print("Uploading configs...")

for config_file in Path(config_dir).glob('*.json'):
    remote_file = f"{remote_base}/configs/{config_file.name}"
    bucket.upload_local_file(
        local_file=str(config_file),
        file_name=remote_file
    )

# ===================================
# 6. GENERATE URLS
# ===================================
# B2 URLs format: b2://bucket-name/path
dataset_url = f"b2://training-datasets/{remote_base}/dataset/"
config_url = f"b2://training-datasets/{remote_base}/configs/config.json"
output_url = f"b2://training-models/{job_id}/"  # Where model will be saved

# ===================================
# 7. OUTPUT
# ===================================
return {
    "job_id": job_id,
    "backblaze_dataset_url": dataset_url,
    "backblaze_config_url": config_url,
    "backblaze_output_url": output_url,
    "upload_complete_at": datetime.now().isoformat(),
    "files_uploaded": uploaded_files,
    "status": "uploaded"
}
Then, in n8n:
javascript// Update Airtable
UPDATE record WHERE job_id = $json.job_id:
- backblaze_dataset_url = $json.backblaze_dataset_url
- backblaze_config_url = $json.backblaze_config_url
- backblaze_output_url = $json.backblaze_output_url
- upload_complete_at = $json.upload_complete_at
- status = "uploaded"

// Trigger Workflow 6
POST to n8n webhook for Workflow 6
```

---

## 🚀 **Workflow 6-9: Training, Monitoring, Analysis, Notification**

*(These workflows remain mostly as documented earlier, with minor adjustments to use the correct paths)*

### **Key Changes:**
- Dataset URL: `b2://training-datasets/{job_id}_{lora_token}/dataset/`
- Config URL: `b2://training-datasets/{job_id}_{lora_token}/configs/config.json`
- Output URL: `b2://training-models/{job_id}/`

---

## 📊 **Updated File Structure**
```
User's Original Location (e.g., D:/photos/bottles/):
├─ bottle001.jpg          ← Original files
├─ bottle002.jpg
├─ bottle003.jpg
│
└─ job_123_sks_fragrancebottle_SDXL_v1/  ← Created by preprocessing
   ├─ 1024x1024/
   │  ├─ bottle001.png
   │  ├─ bottle001.txt    ← Captions
   │  ├─ bottle002.png
   │  └─ bottle002.txt
   ├─ 1024x1536/
   │  ├─ bottle003.png
   │  └─ bottle003.txt
   └─ configs/
      ├─ config.json
      ├─ multidatabackend.json
      └─ user_prompt_library.json
```

**Then uploaded to Backblaze:**
```
b2://training-datasets/
└─ jobs/
   └─ job_123_sks_fragrancebottle/
      ├─ dataset/
      │  ├─ 1024x1024/
      │  │  ├─ bottle001.png
      │  │  └─ bottle001.txt
      │  └─ 1024x1536/
      │     ├─ bottle003.png
      │     └─ bottle003.txt
      └─ configs/
         ├─ config.json
         ├─ multidatabackend.json
         └─ user_prompt_library.json

✅ UPDATED TODO LIST
Phase 1: Setup (Week 1)

 Create Airtable base with all fields from schema
 Set up Slack webhook
 Install dependencies on n8n server
 Create template directory structure
 Set environment variables

Phase 2: Core Workflows (Week 2-4)

 Build Workflow 1: Path Normalizer
 Build Workflow 2: Image Preprocessor
 Build Workflow 3: Image Captioner
 Build Workflow 4: Config Generator
 Build Workflow 5: Upload to Backblaze

Phase 3: Training Integration (Week 5-6)

 Create SimpleTuner Docker image
 Build Workflow 6: Launch Training
 Build Workflow 7: Monitor Training
 Build Workflow 8: Post-Training Analysis
 Build Workflow 9: Slack Notification

Phase 4: Testing (Week 7)

 End-to-end test with all model types
 Test error handling
 Optimize performance

 What you actually need
1) Containers (compose up)


n8n (webhook + UI)


n8n worker(s) for queue mode (parallel jobs)


Redis (backing the queue)


Your data volume(s) mounted so n8n can see D:\... as a POSIX path inside containers


Python + tools inside the n8n image (Pillow, numpy, b2sdk, wandb, ffmpeg, etc.) so your “Python Code” or “Execute Command” steps work locally


2) Secrets / env


Airtable API + base/table IDs


Slack webhook


Backblaze B2 S3 creds (key id, key, endpoint)


OpenAI key (captioner)


WandB key


RunPod key (training launcher)


n8n encryption key + basic auth


3) Hardware


CPU: 8 vCPU is comfy for preprocessing + 2–3 concurrent jobs


RAM: 16–32 GB (Pillow/NumPy + multiple workers)


Disk: fast SSD; size = your largest in-flight dataset × 2 (processing copies)


GPU: not needed for n8n; training happens on RunPod


OS: Windows host is fine; Docker Desktop binds D:\... into Linux containers



Docker files
docker-compose.yml
version: "3.8"

services:
  redis:
    image: redis:7
    restart: unless-stopped
    ports: ["6379:6379"]

  n8n:
    build: ./n8n
    restart: unless-stopped
    env_file: .env
    ports: ["5678:5678"]
    depends_on: [redis]
    volumes:
      # n8n persistent state
      - n8n_data:/home/node/.n8n
      # your templates, scripts, etc.
      - ./workspace:/workspace
      # bind your Windows dataset path(s)
      - "D:/photos:/data/photos:rw"
      - "D:/ai/scripts:/data/scripts:rw"

  # queue workers (scale these for parallelism)
  n8n-worker:
    build: ./n8n
    restart: unless-stopped
    env_file: .env
    command: ["n8n", "worker"]
    depends_on: [redis]
    volumes:
      - n8n_data:/home/node/.n8n
      - ./workspace:/workspace
      - "D:/photos:/data/photos:rw"
      - "D:/ai/scripts:/data/scripts:rw"

volumes:
  n8n_data:

n8n/Dockerfile
FROM n8nio/n8n:latest

# Install OS tools you’ll want
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python libs for your pipelines
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Optional: place helper scripts in /workspace/scripts
WORKDIR /workspace
USER node

n8n/requirements.txt (starter)
b2sdk==1.30.0
boto3==1.34.131               # optional if you prefer S3 API for B2
botocore==1.34.131
Pillow==10.4.0
numpy==1.26.4
opencv-python==4.10.0.84      # if you want it
wandb==0.17.5
openai==1.51.2
requests==2.32.3
tqdm==4.66.4

.env (example — fill yours)
# n8n basics
N8N_ENCRYPTION_KEY=replace_with_random_32+ chars
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=supersecret

# queue mode
EXECUTIONS_MODE=queue
QUEUE_BULL_REDIS_HOST=redis
QUEUE_BULL_REDIS_PORT=6379
# (optional) hard job timeouts
EXECUTIONS_TIMEOUT=3600
EXECUTIONS_TIMEOUT_MAX=14400

# external URLs (adjust)
N8N_HOST=localhost
WEBHOOK_URL=http://localhost:5678/

# Backblaze S3-style
S3_ENDPOINT=https://s3.us-west-001.backblazeb2.com
S3_FORCE_PATH_STYLE=true
S3_ACCESS_KEY=YOUR_B2_KEY_ID
S3_SECRET_KEY=YOUR_B2_APP_KEY
B2_DEFAULT_BUCKET=training-datasets

# If you also use native b2sdk:
B2_APPLICATION_KEY_ID=YOUR_B2_KEY_ID
B2_APPLICATION_KEY=YOUR_B2_APP_KEY

# APIs
OPENAI_API_KEY=sk-...
AIRTABLE_TOKEN=pat...
RUNPOD_API_KEY=...
WANDB_API_KEY=...
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

Run it:
docker compose up -d --build
# scale workers as needed:
docker compose up -d --scale n8n-worker=3


How queueing works here


Queue mode puts executions into Redis; n8n (main) handles triggers/webhooks/UI, n8n-worker processes jobs.


Scale to more parallel jobs with --scale n8n-worker=N.


Per-workflow concurrency: in n8n → Workflow Settings:


Concurrency (e.g., 2–4)


Retry on fail with backoff




Use Rate Limit nodes between API calls (OpenAI, RunPod).


Use an IF + Wait loop for WandB/RunPod polling, so workers aren’t blocked unnecessarily.



Where you can skip URLs (direct disk access)
Because we bind D:\... into the containers:


Workflow 1: Path Normalizer — point to /data/photos/... (no URLs)


Workflow 2: Preprocessor — read/write under /data/photos/...


Workflow 3: Captioner — read images from /data/..., write .txt right next to them


Workflow 4: Config Generator — write /data/.../configs/*.json


Only when you hit Workflow 5: Upload to Backblaze do you push to cloud and optionally create presigned URLs for downstream consumers. Everything before that can stay purely filesystem-based.

Node choices (pragmatic)


Python work: use an Execute Command node that runs:
python3 /workspace/scripts/your_step.py --arg1 ... --arg2 ...

or keep your inline “Code (Python)” node if you’ve installed a Python runner; the Execute Command pattern is simpler to test and version (scripts live in workspace/scripts).


Backblaze:


For bulk upload: either b2sdk via Python script or n8n’s S3 nodes (endpoint = B2 S3).


For sharing: S3 → Get Presigned URL (easier than b2sdk for expiring links).




Airtable: use n8n nodes for updates; keep your schema exactly as you outlined.


Slack: single “Incoming Webhook” HTTP Request.



Minimal example scripts you can call
/workspace/scripts/inspect_dataset.py
#!/usr/bin/env python3
import sys, json, random
from pathlib import Path
from PIL import Image

dataset_path = Path(sys.argv[1])  # e.g. /data/photos/bottles
model_base = sys.argv[2]          # SDXL, Flux, etc.
lora_token = sys.argv[3]

exts = {'.jpg','.jpeg','.png','.webp'}
files = [p for p in dataset_path.rglob('*') if p.suffix.lower() in exts]
if len(files) < 4:
    print(json.dumps({"error": "Minimum 4 images required"})); sys.exit(1)

sample = random.sample(files, min(10, len(files)))
resolutions, has_txt, caps = [], 0, []
for p in sample:
    try:
        w,h = Image.open(p).size
        resolutions.append((w,h))
    except: pass
    t = p.with_suffix('.txt')
    if t.exists():
        has_txt += 1
        try: caps.append(t.read_text(encoding='utf-8')[:300])
        except: pass

target_map = {"SD1.5":512,"SDXL":1024,"Flux":1024,"flux_kontext":1024,"wan22":512,"hunyuan":1024,"qwen_image":1024}
target = target_map.get(model_base,1024)

needs_pre = (len({r for r in resolutions})>1)
needs_cap = (has_txt/len(sample) if sample else 0) < 0.8

out = {
  "image_count": len(files),
  "target_resolution": target,
  "needs_preprocessing": needs_pre,
  "needs_captioning": needs_cap,
  "inspection_report": {
    "resolutions": resolutions, "sample_captions": caps[:3]
  }
}
print(json.dumps(out))

Call it from an Execute Command node:
python3 /workspace/scripts/inspect_dataset.py "={{$json.normalized_path}}" "={{$json.model_base}}" "={{$json.lora_token}}"

Parse stdout JSON with a small Code(JS) node and set Airtable fields.
/workspace/scripts/upload_b2.py (S3 API route)
#!/usr/bin/env python3
import os, sys, json, mimetypes
import boto3
from pathlib import Path

bucket = os.environ["B2_DEFAULT_BUCKET"]
endpoint = os.environ["S3_ENDPOINT"]
key = os.environ["S3_ACCESS_KEY"]
secret = os.environ["S3_SECRET_KEY"]

dataset_dir = Path(sys.argv[1])    # /data/photos/.../processed or normalized
remote_base = sys.argv[2]          # jobs/{job_id}_{lora_token}

s3 = boto3.client("s3",
  endpoint_url=endpoint,
  aws_access_key_id=key,
  aws_secret_access_key=secret,
)

uploaded = 0
for p in dataset_dir.rglob('*'):
    if p.is_file():
        rel = p.relative_to(dataset_dir)
        keyname = f"{remote_base}/dataset/{rel.as_posix()}"
        s3.upload_file(str(p), bucket, keyname, ExtraArgs={"ContentType": mimetypes.guess_type(p.name)[0] or "application/octet-stream"})
        uploaded += 1

print(json.dumps({"files_uploaded": uploaded, "dataset_url": f"b2://{bucket}/{remote_base}/dataset/"}))

Presign a config or any object later with an S3 → Get Presigned URL node.

Handling multiple requests (the “queue” bit)


Scale workers: docker compose up -d --scale n8n-worker=4 (4 parallel job executors).


Per-workflow:


Add Concurrency and Retry settings.


Wrap rate-limited sections with Rate Limit.




Airtable status: write status transitions at each major step so you can resume/inspect easily.


Idempotency: use job_id-scoped folders (you already do) so re-runs don’t stomp each other.



Where URLs are unavoidable


When a remote service needs access to files (e.g., a RunPod container pulling the dataset directly). In that case, you’ll:


Upload to B2 (Workflow 5),


Give RunPod a B2 S3 URL / presigned URL or let the training image read from B2 using your creds.




For sharing results (Slack message links) use presigned URLs.


Everything else (1–4) can stay local in /data/... with zero URL fuss.

Next actions for you


Save the files above (docker-compose.yml, n8n/Dockerfile, n8n/requirements.txt, .env) and run:
docker compose up -d --build
docker compose up -d --scale n8n-worker=3



In your n8n workflows, replace D:\... with /data/....


Keep your preprocessing/captioning nodes as Python Execute Command steps calling scripts in /workspace/scripts.


Add S3 (B2) nodes for upload and presign in Workflow 5.


Turn on queue mode (already configured) and tune concurrency.


If you want, I can package your current Python blocks into /workspace/scripts, plus a tiny “library” for common tasks (path utils, B2 upload, presign, Airtable updates), and give you a one-file import for the B2 upload/presign chain that slots into Workflow 5.