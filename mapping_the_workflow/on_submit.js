// n8n Workflow triggered via Airtable webhook

1. Create Job Record
   - job_id = "job_" + timestamp
   - status = "pending"
   - created_at = now()
   - All form fields copied to record

2. Validate Input
   IF !dataset_path:
     → Error: "Dataset path required"
   IF !lora_token:
     → Set default: "sks_fragrancebottle"
   IF !model_base:
     → Error: "Model type required"
   IF !training_type:
     → Set default: "object"

3. Initialize Processing Flags
   - needs_preprocessing = null  // Will detect
   - needs_captioning = null     // Will detect
   - config_source = custom_config ? "custom" : "template"

4. Send Slack Notification
   Channel: #lora-training-bot (general channel)
   Message: "🚀 New training job started: [job_id]
            LoRA Token: [lora_token]
            Model: [model_base]
            Status: Downloading dataset..."

5. Trigger Workflow 1: Path Normalizer
```

---

## **Workflow 1: Path Normalizer & Dataset Inspector**

**Purpose:** Download files + detect if preprocessing/captioning needed
```
INPUT: dataset_path from Airtable
OUTPUT: Validated path + detection flags

1. Normalize Path Format
   # Just convert to consistent format, DON'T copy files
   
   path = dataset_path
   
   # Convert Windows backslashes to forward slashes
   if "\\" in path:
       normalized_path = path.replace("\\", "/")
   else:
       normalized_path = path
   
   # That's it! Keep working with files in place.

2. Validate Files Exist
   import os
   from pathlib import Path
   
   source = Path(normalized_path)
   
   if not source.exists():
       → Error: "Path does not exist"
   
   # Count images
   image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
   image_files = [
       f for f in source.rglob('*') 
       if f.suffix.lower() in image_extensions
   ]
   
   image_count = len(image_files)
   
   if image_count < 4:
       → Error: "Minimum 4 images required"

3. Inspect Dataset (sample, don't copy)
   # Sample 10 random images IN PLACE
   sample = random.sample(image_files, min(10, image_count))
   
   resolutions = []
   has_txt_count = 0
   
   for img_path in sample:
       # Read directly from source path
       img = Image.open(img_path)
       resolutions.append((img.width, img.height))
       
       # Check for caption file
       txt_path = img_path.with_suffix('.txt')
       if txt_path.exists():
           has_txt_count += 1
   
   # Detect preprocessing/captioning needs
   needs_preprocessing = (resolution analysis)
   needs_captioning = (has_txt_count < 8)

4. Update Airtable
   - normalized_path = normalized_path  // Just the clean path
   - image_count = image_count
   - needs_preprocessing = needs_preprocessing
   - needs_captioning = needs_captioning
   - status = "validated"

5. Route to Next Step
   if needs_preprocessing:
       → Trigger Workflow 2 (works on normalized_path)
   elif needs_captioning:
       → Trigger Workflow 3 (works on normalized_path)
   else:
       → Trigger Workflow 4
       
4. Validate Files
   files = []
   for file in os.listdir(raw_dir):
       if file.endswith(('.jpg', '.jpeg', '.png', '.webp')):
           files.append(file)
   
   image_count = len(files)
   
   IF image_count < 4:
     → Error: "Minimum 4 images required"
   IF image_count > 10000:
     → Error: "Maximum 10,000 images exceeded"
   
   # Calculate size
   total_size_bytes = sum(os.path.getsize(f"{raw_dir}/{f}") for f in files)
   total_size_mb = total_size_bytes / (1024 * 1024)

5. Inspect Dataset (Detect preprocessing status)
   
   # Sample 10 random images
   sample = random.sample(files, min(10, image_count))
   
   resolutions = []
   has_txt_files = 0
   
   for img_file in sample:
       # Check resolution
       img = Image.open(f"{raw_dir}/{img_file}")
       width, height = img.size
       resolutions.append((width, height))
       
       # Check for caption file
       txt_file = img_file.rsplit('.', 1)[0] + '.txt'
       if os.path.exists(f"{raw_dir}/{txt_file}"):
           has_txt_files += 1
   
   # Analyze
   unique_resolutions = len(set(resolutions))
   avg_width = sum(r[0] for r in resolutions) / len(resolutions)
   avg_height = sum(r[1] for r in resolutions) / len(resolutions)
   
   # DETECTION LOGIC
   needs_preprocessing = True  # Default
   needs_captioning = True     # Default
   
   # Check if already processed
   IF unique_resolutions == 1 AND avg_width in [512, 768, 1024]:
     needs_preprocessing = False
     reason = "All images same target resolution"
   
   # Check if already captioned
   caption_coverage = has_txt_files / len(sample)
   IF caption_coverage > 0.8:
     needs_captioning = False
     # Verify LoRA token present
     sample_caption = open(f"{raw_dir}/{sample[0].rsplit('.', 1)[0]}.txt").read()
     if lora_token.lower() not in sample_caption.lower():
       needs_captioning = True  # Re-caption to add token
       reason = "Captions missing LoRA token"

6. Update Path in Airtable
   # Normalize path for later reference
   normalized_path = job_dir
   
   UPDATE Airtable:
   - normalized_dataset_path = normalized_path
   - raw_directory = raw_dir
   - image_count = image_count
   - total_size_mb = total_size_mb
   - needs_preprocessing = needs_preprocessing
   - needs_captioning = needs_captioning
   - status = "downloaded"

7. Generate Inspection Report
   inspection_report = {
     "image_count": image_count,
     "total_size_mb": total_size_mb,
     "sample_resolutions": resolutions,
     "unique_resolutions": unique_resolutions,
     "caption_coverage": caption_coverage,
     "needs_preprocessing": needs_preprocessing,
     "needs_captioning": needs_captioning,
     "detection_reason": reason
   }
   
   Save to: {job_dir}/reports/inspection_report.json

8. Decision Tree - Route to Next Workflow
   IF needs_preprocessing:
     → Trigger Workflow 2: Image Preprocessor
   ELIF needs_captioning:
     → Trigger Workflow 3: Image Captioner
   ELSE:
     → Trigger Workflow 4: Config Generator
```

---

## **Workflow 2: Image Preprocessor (Simplified)**
```
INPUT: {job_dir}/raw/
OUTPUT: {job_dir}/processed/

1. Load Config
   model_base = from Airtable
   
   # Target resolution based on model
   target_res = {
     "SD1.5": 512,
     "SDXL": 1024,
     "Flux": 1024,
     "flux_kontext": 1024,
     "wan22": 512,
     "hunyuan": 1024,
     "qwen_image": 1024
   }[model_base]

2. Analyze for Aspect Bucketing
   aspect_ratios = []
   for img_file in os.listdir(f"{job_dir}/raw/"):
       img = Image.open(f"{job_dir}/raw/{img_file}")
       ratio = img.width / img.height
       aspect_ratios.append(ratio)
   
   variance = np.std(aspect_ratios)
   
   IF variance > 0.3:
     use_buckets = True
     buckets = generate_buckets(target_res)
     # e.g., [512x512, 512x768, 768x512] for SD1.5
     #       [1024x1024, 1024x1536, 1536x1024] for SDXL
   ELSE:
     use_buckets = False
     buckets = [f"{target_res}x{target_res}"]

3. Process Images
   from PIL import Image, ImageFilter
   
   for img_file in os.listdir(f"{job_dir}/raw/"):
       img = Image.open(f"{job_dir}/raw/{img_file}")
       
       # Auto-crop (center crop + smart crop for faces)
       img = smart_crop(img, target_res)
       
       # Determine bucket
       if use_buckets:
           bucket = assign_to_bucket(img, buckets)
       else:
           bucket = buckets[0]
       
       # Resize
       bucket_w, bucket_h = map(int, bucket.split('x'))
       img_resized = img.resize((bucket_w, bucket_h), Image.LANCZOS)
       
       # Save as PNG
       output_dir = f"{job_dir}/processed/{bucket}"
       os.makedirs(output_dir, exist_ok=True)
       output_path = f"{output_dir}/{img_file.rsplit('.', 1)[0]}.png"
       img_resized.save(output_path, "PNG", optimize=True)

4. Quality Check
   processed_count = sum(
       len(files) for _, _, files in os.walk(f"{job_dir}/processed/")
   )
   
   IF processed_count != image_count:
     → Error: "Processing failed"

5. Generate Report
   preprocessing_report = {
     "original_count": image_count,
     "processed_count": processed_count,
     "buckets": {
       bucket: len(os.listdir(f"{job_dir}/processed/{bucket}"))
       for bucket in buckets
     },
     "target_resolution": target_res,
     "used_aspect_bucketing": use_buckets
   }
   
   Save to: {job_dir}/reports/preprocessing_report.json

6. Update Airtable
   - processed_directory = f"{job_dir}/processed/"
   - processed_image_count = processed_count
   - preprocessing_report = preprocessing_report
   - status = "preprocessed"

7. Trigger Workflow 3: Image Captioner
```

---

## **Workflow 3: Image Captioner (Production Ready)**
```
INPUT: {job_dir}/processed/ (or /raw/ if preprocessing skipped)
OUTPUT: .txt files next to each image

1. Determine Input Directory
   IF needs_preprocessing was True:
     input_dir = f"{job_dir}/processed/"
   ELSE:
     input_dir = f"{job_dir}/raw/"

2. Load All Images
   image_files = []
   for root, dirs, files in os.walk(input_dir):
       for file in files:
           if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
               image_files.append(os.path.join(root, file))
   
   total_images = len(image_files)

3. Load Prompt Template
   training_type = from Airtable
   model_base = from Airtable
   lora_token = from Airtable
   
   # Use your existing prompt templates
   prompts = {
     "character": "Describe this image for character LoRA training...",
     "object": "Describe this image for object LoRA training...",
     "style": "Describe the style...",
     "scene": "Describe the setting...",
     "motion_lora": "Describe the motion and action..."
   }
   
   base_prompt = prompts[training_type]
   final_prompt = f"IMPORTANT: Begin with '{lora_token}'. {base_prompt}"

4. Batch Processing (Smart Rate Limiting)
   from openai import OpenAI
   import base64
   
   client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
   
   batch_size = 50
   batches = [image_files[i:i+batch_size] for i in range(0, len(image_files), batch_size)]
   
   completed = 0
   failed = []
   
   for batch in batches:
       results = []
       
       # Process batch in parallel
       for img_path in batch:
           try:
               # Read and encode image
               with open(img_path, 'rb') as f:
                   img_data = base64.b64encode(f.read()).decode()
               
               # Determine mime type
               ext = img_path.split('.')[-1].lower()
               mime = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
               data_uri = f"data:{mime};base64,{img_data}"
               
               # Call OpenAI
               response = client.chat.completions.create(
                   model="gpt-4o",
                   messages=[{
                       "role": "user",
                       "content": [
                           {"type": "text", "text": final_prompt},
                           {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}}
                       ]
                   }],
                   max_tokens=300
               )
               
               caption = response.choices[0].message.content.strip()
               results.append((img_path, caption, True))
               
           except Exception as e:
               results.append((img_path, str(e), False))
               failed.append(img_path)
       
       # Save after each batch
       for img_path, caption, success in results:
           if success:
               txt_path = img_path.rsplit('.', 1)[0] + '.txt'
               with open(txt_path, 'w') as f:
                   f.write(caption)
               completed += 1
       
       # Rate limit buffer
       time.sleep(6)
       
       # Update progress in Airtable
       progress_pct = (completed / total_images) * 100
       UPDATE Airtable:
       - captioning_progress = progress_pct
       - captions_completed = completed

5. Quality Validation
   # Check LoRA token presence
   token_accuracy = 0
   caption_lengths = []
   
   for img_path in image_files:
       txt_path = img_path.rsplit('.', 1)[0] + '.txt'
       if os.path.exists(txt_path):
           with open(txt_path) as f:
               caption = f.read()
               caption_lengths.append(len(caption.split()))
               if lora_token.lower() in caption.lower():
                   token_accuracy += 1
   
   token_accuracy_pct = token_accuracy / total_images

6. Generate Report
   captioning_report = {
     "total_images": total_images,
     "successful": completed,
     "failed": len(failed),
     "lora_token_accuracy": token_accuracy_pct,
     "avg_caption_length": sum(caption_lengths) / len(caption_lengths),
     "failed_files": failed
   }
   
   Save to: {job_dir}/reports/captioning_report.json

7. Update Airtable
   - captioning_report = captioning_report
   - captions_completed = completed
   - caption_accuracy = token_accuracy_pct
   - status = "captioned"

8. Trigger Workflow 4: Config Generator
```

---

## **Workflow 4: Config Generator (Template-Based)**
```
INPUT: Dataset metadata + Optional custom_config
OUTPUT: Training config files

1. Load Template
   model_base = from Airtable
   training_type = from Airtable
   custom_config = from Airtable (if provided)
   
   # Your pre-tested templates
   template_path = f"/templates/{model_base}_{training_type}_config.json"
   config_template = load_json(template_path)
   
   # Example: /templates/SDXL_object_config.json

2. Merge Custom Config (if provided)
   IF custom_config:
     config = {**config_template, **json.loads(custom_config)}
   ELSE:
     config = config_template

3. Fill Dynamic Values
   config["instance_prompt"] = lora_token
   config["output_dir"] = f"/output/{job_id}"
   
   # Get image count and buckets
   processed_dir = from Airtable
   image_count = count_images(processed_dir)
   buckets = list_subdirectories(processed_dir)
   
   # Update paths
   config["data_backend_config"] = {
     "datasets": [
       {
         "id": bucket,
         "instance_data_dir": f"/dataset/{bucket}",
         "resolution": int(bucket.split('x')[0])
       }
       for bucket in buckets
     ]
   }

4. Generate Validation Prompts
   validation_prompts = from Airtable
   
   IF !validation_prompts:
     # Auto-generate from common caption patterns
     sample_captions = []
     for txt_file in glob(f"{processed_dir}/**/*.txt"):
         with open(txt_file) as f:
             sample_captions.append(f.read())
     
     # Extract common phrases
     prompts = generate_prompts_from_captions(sample_captions, lora_token)
     # e.g., ["sks_fragrancebottle on table", "sks_fragrancebottle close-up"]
   ELSE:
     prompts = validation_prompts.split('\n')
   
   config["validation_prompt"] = prompts

5. Generate WandB Config
   config["report_to"] = "wandb"
   config["logging_dir"] = f"/logs/{job_id}"

6. Save Config Files
   config_dir = f"{job_dir}/configs/"
   os.makedirs(config_dir, exist_ok=True)
   
   # Main config
   with open(f"{config_dir}/config.json", 'w') as f:
       json.dump(config, f, indent=2)
   
   # Multidatabackend
   with open(f"{config_dir}/multidatabackend.json", 'w') as f:
       json.dump(config["data_backend_config"], f, indent=2)
   
   # User prompt library
   with open(f"{config_dir}/user_prompt_library.json", 'w') as f:
       json.dump({"prompts": prompts}, f, indent=2)

7. Update Airtable
   - config_dir = config_dir
   - config_source = "custom" if custom_config else "template"
   - status = "config_ready"

8. Trigger Workflow 5: Upload to Backblaze
```

---

## **Workflow 5: Upload to Backblaze**
```
INPUT: {job_dir}/
OUTPUT: Backblaze URLs

1. Create Package Structure
   package_dir = f"{job_dir}/package/"
   
   # Copy processed images
   shutil.copytree(f"{job_dir}/processed/", f"{package_dir}/dataset/")
   
   # Copy configs
   shutil.copytree(f"{job_dir}/configs/", f"{package_dir}/configs/")
   
   # Copy reports
   shutil.copytree(f"{job_dir}/reports/", f"{package_dir}/metadata/")

2. Upload to Backblaze (Using B2 CLI - No rclone!)
   
   # Install if needed: pip install b2
   from b2sdk.v2 import B2Api, InMemoryAccountInfo
   
   # Authenticate
   info = InMemoryAccountInfo()
   b2_api = B2Api(info)
   b2_api.authorize_account("production", APP_KEY_ID, APP_KEY)
   
   # Get bucket
   bucket = b2_api.get_bucket_by_name("training-datasets")
   
   # Upload directory
   remote_path = f"jobs/{job_id}_{lora_token}/"
   
   for root, dirs, files in os.walk(package_dir):
       for file in files:
           local_file = os.path.join(root, file)
           relative_path = os.path.relpath(local_file, package_dir)
           remote_file = remote_path + relative_path
           
           bucket.upload_local_file(
               local_file=local_file,
               file_name=remote_file
           )
   
   # OR use CLI:
   b2 sync {package_dir} b2://training-datasets/{remote_path}

3. Generate URLs
   dataset_url = f"b2://training-datasets/{remote_path}dataset/"
   config_url = f"b2://training-datasets/{remote_path}configs/config.json"
   output_url = f"b2://training-models/{job_id}/"

4. Update Airtable
   - backblaze_dataset_url = dataset_url
   - backblaze_config_url = config_url
   - backblaze_output_url = output_url
   - upload_complete_at = now()
   - status = "uploaded"

5. Trigger Workflow 6: Launch Training
```

---

## 🔄 **Updated Workflow Triggers Summary**
```
Airtable Form Submit
    ↓
Job Intake (validation + Slack notification)
    ↓
Workflow 1: Path Normalizer & Inspector
    ├─ IF needs_preprocessing → Workflow 2
    ├─ ELIF needs_captioning → Workflow 3
    └─ ELSE → Workflow 4
    
Workflow 2: Image Preprocessor
    ↓
    └─ → Workflow 3
    
Workflow 3: Image Captioner
    ↓
    └─ → Workflow 4
    
Workflow 4: Config Generator
    ↓
    └─ → Workflow 5
    
Workflow 5: Upload to Backblaze
    ↓
    └─ → Workflow 6
    
Workflow 6: Launch RunPod Training
    ↓
    └─ → Workflow 7 (polling loop)
    
Workflow 7: Monitor Training
    ↓
    └─ When complete → Workflow 8
    
Workflow 8: Post-Training Analysis
    ↓
    └─ → Workflow 9
    
Workflow 9: Slack Notification (DONE)
```

---

## 💾 **Updated Airtable Schema**

### **Table: training_jobs**
```
CORE FIELDS:
- job_id (text, primary key, auto: "job_" + timestamp)
- created_at (datetime, auto)
- updated_at (datetime, auto)
- status (single select):
    • pending
    • downloading
    • downloaded
    • preprocessing
    • preprocessed
    • captioning
    • captioned
    • config_ready
    • uploaded
    • training
    • training_complete
    • analyzing
    • complete
    • failed

INPUT FIELDS (from form):
- dataset_path (long text)
- lora_token (text, default: "sks_fragrancebottle")
- model_base (single select):
    • SDXL
    • SD1.5
    • Flux
    • flux_kontext
    • wan22
    • hunyuan
    • qwen_image
- training_type (single select):
    • character
    • object
    • style
    • scene
    • motion_lora
- custom_config (long text, optional)
- validation_prompts (long text, optional)

WORKFLOW STATE:
- normalized_dataset_path (text)
- raw_directory (text)
- processed_directory (text)
- config_dir (text)
- needs_preprocessing (checkbox)
- needs_captioning (checkbox)

DATASET INFO:
- image_count (number)
- total_size_mb (number)
- processed_image_count (number)

REPORTS (JSON):
- inspection_report (long text)
- preprocessing_report (long text)
- captioning_report (long text)

CAPTIONING PROGRESS:
- captioning_progress (percent)
- captions_completed (number)
- caption_accuracy (percent)

CLOUD STORAGE:
- backblaze_dataset_url (URL)
- backblaze_config_url (URL)
- backblaze_output_url (URL)
- upload_complete_at (datetime)

TRAINING:
- runpod_job_id (text)
- wandb_url (URL)
- training_started_at (datetime)
- training_completed_at (datetime)
- training_progress (percent)
- current_step (text, e.g. "1250/2776")
- current_loss (number)
- eta (text, e.g. "1h 15m")

RESULTS:
- best_model_url (URL)
- best_model_score (number)
- analysis_report_url (URL)

COSTS:
- estimated_cost (currency)
- actual_cost (currency)
- captioning_cost (currency)

ERROR HANDLING:
- error_log (long text)
- retry_count (number)
```

---

## 💰 **Updated Cost Breakdown**

### **Per LoRA Training (247 images, SDXL):**
```
Workflow 1: Path Normalization
- n8n compute: $0.01
- Download bandwidth: $0.02
SUBTOTAL: $0.03

Workflow 2: Image Preprocessing
- n8n compute (Python PIL): $0.03
- Temporary storage: $0.01
SUBTOTAL: $0.04

Workflow 3: Image Captioning
- OpenAI API (247 images × $0.002): $0.49
- n8n compute: $0.01
SUBTOTAL: $0.50

Workflow 4: Config Generation
- n8n compute: $0.00
SUBTOTAL: $0.00

Workflow 5: Upload to Backblaze
- Upload bandwidth: $0.01
- Storage (30 days, 2GB): $0.03
SUBTOTAL: $0.04

Workflow 6-7: RunPod Training
- 4090 GPU (2.4 hours × $0.80/hr): $1.92
- Storage during training: $0.01
SUBTOTAL: $1.93

Workflow 8: Post-Training Analysis
- GPT-4V analysis: $0.15
- Download bandwidth: $0.01
SUBTOTAL: $0.16

Workflow 9: Slack Notification
- Free
SUBTOTAL: $0.00

WandB Tracking: Free (team tier)

TOTAL PER LORA: $2.70
```

### **At Scale (100 LoRAs/month):**
```
Direct Costs:
- Path normalization: $3
- Preprocessing: $4
- Captioning: $50
- Backblaze storage: $4
- RunPod training: $193
- VLM analysis: $16
TOTAL: $270/month

Infrastructure:
- n8n self-hosted: $20/month (DigitalOcean)
- Airtable Pro: $20/month
TOTAL: $40/month

GRAND TOTAL: $310/month for 100 LoRAs
COST PER LORA: $3.10

Pricing Strategy:
- Charge $5-10 per LoRA
- Profit margin: 60-70%