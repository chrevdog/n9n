// n8n Workflow triggered via Airtable webhook
- Create job_id (e.g., job_1762875194225)
- Status → "pending"
- Validate all required fields
- Estimate cost & time
- Send confirmation email/Slack
- Trigger Workflow 1
```

---

### **Workflow 1: Path Normalizer & File Sync**

**Purpose:** Convert any path format → download → validate → organize
```
INPUT: dataset_path from Airtable
OUTPUT: Organized local files + metadata

1. Detect Path Type
   IF path starts with "\\\\":
     → Samba/Network Share
     → Mount using smbclient
   ELSE IF path starts with "/Volumes/" OR "/Users/":
     → macOS path
     → Use rsync or rclone - i have had difficulty using r clone in the past wget or back blaze or runpod native cli or api I would ask for alternatives to r clone where possible 
   ELSE IF path starts with "D:/" OR "C:\\":
     → Windows path
     → Use robocopy or rclone - i have had difficulty using r clone in the past wget or back blaze or runpod native cli or api I would ask for alternatives to r clone where possible 
   ELSE IF path starts with "s3://" OR "gs://":
     → Cloud storage
     → Use s3cmd or gsutil 
replaces//updates path in the field for reference later on

// 2. Create Working Directory
//    local_path = /updated local input path from airtable directory/[job_id]_[lora_token]_[modeltype ei flux]/
   
// 3. Download/Copy Files
//    - Use rclone - i have had difficulty using r clone in the past wget or back blaze or runpod native cli or api I would ask for alternatives to r clone where possible  for unified interface (supports all path types!)
//    - Command: rclone - i have had difficulty using r clone in the past wget or back blaze or runpod native cli or api I would ask for alternatives to r clone where possible  copy [normalized_path] [local_path] --include "*.{jpg,jpeg,png,webp}"
//    - Track progress in Airtable
//
   

### **Workflow 2: Image Preprocessor**
```
INPUT: input directory from normalized path 
OUTPUT: input directory from normalized path /[job_id]_[lora_token]_[modeltype ei flux]/

1. Load Config from Airtable
   - target_resolution
   - enable_auto_crop
   - enable_aspect_bucketing
   
   4. Validate Files - streamline this if possible
   - Count files (min 4, max 10,000)
   - Check all files readable
   - Verify image formats
   - Calculate total size
   - Generate file list
    - Check resolution
     - Check aspect ratio
     - Detect faces (optional, for cropping)
     - Calculate blur score
     - Store metadata
    - Sample 10 random images
   - Detect NSFW content (use CLIP or similar)
   - Flag if >30% NSFW
   
3. Generate Processing Plan
   IF aspect_ratio_variance > 0.3 AND aspect_bucketing:
     → Create buckets: 512×512, 512×768, 768×512, 1024×1024
   ELSE:
     → Single resolution: target_resolution
   
   IF auto_crop AND faces_detected:
     → Calculate optimal crop using bbox
   
4. Process Images
   FOR EACH image:
     - Resize/crop according to plan
     from PIL import Image
import os

def process_image(img_path, target_res, bucket):
    img = Image.open(img_path)
    
    # Calculate crop/resize
    if bucket == "512x512":
        # Center crop to square
        img = center_crop_square(img)
        img = img.resize((512, 512), Image.LANCZOS)
    elif bucket == "512x768":
        # Portrait
        img = smart_resize(img, (512, 768))
    
    # Save
    output_path = f"from PIL import Image
import os

def process_image(img_path, target_res, bucket):
    img = Image.open(img_path)
    
    # Calculate crop/resize
    if bucket == "512x512":
        # Center crop to square
        img = center_crop_square(img)
        img = img.resize((512, 512), Image.LANCZOS)
    elif bucket == "512x768":
        # Portrait
        img = smart_resize(img, (512, 768))
    
    # Save
    output_path = f" input directory from normalized path /[job_id]_[lora_token]_[modeltype ei flux]/{os.path.basename(img_path)}"
    img.save(output_path, "JPEG", quality=95)
    
    img.save(output_path, "JPEG", quality=95)
    
    return output_path
     - Save as .png (quality: 100)
     - Maintain filename (update metadata)
     - Copy to appropriate bucket folder
   
   Output structure:
   input directory from normalized path /[job_id]_[lora_token]_[modeltype ei flux]/
     /512x512/
       image001.png
       image002.png
     /512x768/
       image003.png
     /768x512/
       image004.png
   
5. Quality Check
   - Verify all images processed successfully
   - Check no corrupt files
   - Calculate final dataset size
   
6. Generate Report
   preprocessing_report.json:
   {
     "original_count": 247,
     "processed_count": 247,
     "failed_count": 0,
     "buckets": {
       "512x512": 120,
       "512x768": 80,
       "768x512": 47
     },
     "total_size_mb": 890,
     "avg_resolution": [640, 580]
   }
   
7. Update Airtable
   - processed_image_count: 247
   - preprocessing_report: [JSON]
   - status → "preprocessed"
   -path to directory input directory from normalized path /[job_id]_[lora_token]_[modeltype ei flux]/
   
8. Trigger Workflow 3 -- I like a lot of this workflow 3, I have a workflow myself that I cant quite get to work, it as a couple of different QC steps but I would love to work on this in more detail. after we sort out the workflow for the entire pipeline we will move on to this
   
```
INPUT: path to directory input directory from normalized path /[job_id]_[lora_token]_[modeltype ei flux]/ and subfolders if they exist
OUTPUT:  path to directory input directory from normalized path /[job_id]_[lora_token]_[modeltype ei flux]/ and subfolders if they exist

1. Load Images Locally
   - Read from /processed/ directory
   - Convert to base64 data URIs (for OpenAI API)
   
2. Batch Process (Smart Rate Limiting)
   - Calculate: total_images / 60 = batches per minute
   - OpenAI limit: 5,000 requests/min (tier 4)
   - Batch size: min(50, rate_limit_safe_value)
   
   FOR EACH batch:
     - Send 50 images in parallel
     - Wait 6 seconds (rate limit buffer)
     - Save results after each batch (prevent data loss)
   
3. Generate Captions
   FOR EACH image:
     - Send to OpenAI with prompt template
     - Validate LoRA token placement
     - Check caption quality score
     - Retry if failed (3 attempts)
   
4. Save Caption Files
   FOR EACH image:
     - Save as {image_name}.txt next to .jpg
     
   Example:
   /processed/512x512/image001.jpg
   /processed/512x512/image001.txt
   
5. Quality Validation
   - Check all images have captions
   - Validate LoRA token in every caption
   - Calculate quality metrics:
     - Avg caption length: 32 words
     - LoRA token accuracy: 98%
     - Quality score: 0.87
   
6. Generate Caption Report
   captioning_report.json:
   {
     "total_captions": 247,
     "successful": 245,
     "failed": 2,
     "avg_length": 32,
     "lora_token_accuracy": 0.98,
     "quality_score": 0.87,
     "total_cost": 0.49
   }
   
7. Update Airtable
   - caption_status → "complete"
   - captioning_report: [JSON]
   - captioning_cost: 0.49
   - status → "captioned"
   
8. Trigger Workflow 4
```

---

### **Workflow 4: Config Generator** I have templates for these config files I  like where this is at, but I want to develop the other sections first, lets maybe hold with this structure for now unless there is an obvious way to improve simplify or streamline 
```
INPUT: Dataset metadata + Airtable params
OUTPUT: Training config files

1. Analyze Dataset
   total_images = 247
   buckets = {512x512: 120, 512x768: 80, 768x512: 47}
   lora_token = "sks_luneasterstick"
   model_base = "SDXL"
   training_type = "character"
   
2. Calculate Optimal Settings -- skeptical of this, maybe it can reference a table of successful parameters based on dataset types and quality? I would follow your lead on the best way to leverage the tools at hand to simplify parameter configuration while also increasing efficiency and quality. this may be a task that is out of scope for this particular workflow but if this pipeline works I will have the time to develop this more
   # Formula from SimpleTuner best practices
   
   IF total_images < 50:
     repeats = 15
     epochs = 100
   ELIF total_images < 200:
     repeats = 8
     epochs = 50
   ELSE:
     repeats = 3
     epochs = 30
   
   batch_size = 4  # Based on VRAM
   gradient_accumulation = 2
   
   total_steps = (total_images * repeats * epochs) / (batch_size * gradient_accumulation)
   # = (247 * 3 * 30) / (4 * 2) = 2,776 steps
   
   learning_rate = 4e-4  # SDXL defaul

   then I have template files for these different tools so we should probably base them off of things that work, but I love the concept behind generating these scripts It seems incredibly doable I just have no idea the best approach best in terms of production it needs to work and be fool proof and it needs to be easy for wide spread adoption

   Generate config.json
   {
     "model_type": "sdxl",
     "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
     "output_dir": "/output",
     "data_backend": "local",
     "aspect_bucket_alignment": 64,
     "resolution": 1024,
     "resolution_type": "pixel_area",
     "enable_multiresolution_training": true,
     "lora_rank": 32,
     "lora_alpha": 16,
     "lora_dropout": 0.1,
     "learning_rate": 4e-4,
     "lr_scheduler": "constant_with_warmup",
     "lr_warmup_steps": 100,
     "num_train_epochs": 30,
     "train_batch_size": 4,
     "gradient_accumulation_steps": 2,
     "max_train_steps": 2776,
     "save_every_n_steps": 500,
     "validation_prompt": [
       "sks_luneasterstick person standing, neutral pose",
       "sks_luneasterstick person sitting, reading a book",
       "sks_luneasterstick person close-up portrait"
     ],
     "validation_steps": 500,
     "num_validation_images": 4,
     "checkpointing_steps": 500,
     "resume_from_checkpoint": "latest",
     "mixed_precision": "bf16",
     "use_8bit_adam": true,
     "dataloader_num_workers": 4,
     "caption_dropout_probability": 0.1,
     "instance_prompt": "sks_luneasterstick",
     "report_to": "wandb"
   }
   
4. Generate multidatabackend.json
   {
     "datasets": [
       {
         "id": "512x512_bucket",
         "type": "local",
         "instance_data_dir": "/dataset/512x512",
         "caption_strategy": "filename",
         "resolution": 512,
         "minimum_image_size": 512
       },
       {
         "id": "512x768_bucket",
         "type": "local",
         "instance_data_dir": "/dataset/512x768",
         "caption_strategy": "filename",
         "resolution": 512,
         "minimum_image_size": 512
       },
       {
         "id": "768x512_bucket",
         "type": "local",
         "instance_data_dir": "/dataset/768x512",
         "caption_strategy": "filename",
         "resolution": 768,
         "minimum_image_size": 512
       }
     ]
   }
   
5. Generate user_prompt_library.json
   # Based on common patterns from captions
   {
     "prompts": [
       "sks_luneasterstick",
       "sks_luneasterstick person",
       "sks_luneasterstick person standing",
       "sks_luneasterstick person sitting",
       "sks_luneasterstick person portrait",
       "sks_luneasterstick person outdoor",
       "sks_luneasterstick person indoor"
     ]
   }
   
6. Save Config Files
   /tmp/training_jobs/[job_id]/configs/
     config.json
     multidatabackend.json
     user_prompt_library.json
   
7. Generate Training Summary
   training_summary.json:
   {
     "total_steps": 2776,
     "estimated_time_hours": 2.3,
     "estimated_cost_usd": 1.84,
     "checkpoints_saved": 5,
     "validation_images_generated": 20
   }
   
8. Update Airtable
   - config_status → "generated"
   - training_summary: [JSON]
   - estimated_training_cost: 1.84
   - estimated_training_time: "2h 18m"
   - status → "config_ready"
   
9. Trigger Workflow 5
```

---

### **Workflow 5: Package & Upload to Backblaze**
```
INPUT: /tmp/training_jobs/[job_id]/
OUTPUT: Backblaze bucket URL

1. Create Package Structure
   /tmp/training_jobs/[job_id]/package/
     /dataset/
       /512x512/
         image001.jpg
         image001.txt
       /512x768/
         ...
     /configs/
       config.json
       multidatabackend.json
       user_prompt_library.json
     /metadata/
       job_info.json
       preprocessing_report.json
       captioning_report.json
       training_summary.json

       we will figure this out next ^^^ 

       2. Compress (Optional)
   # For faster upload
   tar -czf package.tar.gz package/
   
3. Upload to Backblaze
   Bucket: training-datasets
   Path: jobs/[job_id]_[lora_token]/
   
   Using rclone - i have had difficulty using r clone in the past wget or back blaze or runpod native cli or api I would ask for alternatives to r clone where possible :
   rclone - i have had difficulty using r clone in the past wget or back blaze or runpod native cli or api I would ask for alternatives to r clone where possible  copy /tmp/training_jobs/[job_id]/package/ \
     backblaze:training-datasets/jobs/[job_id]_sks_luneasterstick/ \
     --progress
   
4. Generate URLs
   dataset_url = "b2://training-datasets/jobs/[job_id]/package/"
   # Or use Backblaze native URLs for RunPod to download
   
5. Update Airtable
   - backblaze_url: dataset_url
   - upload_complete_at: timestamp
   - status → "uploaded"
   
6. Cleanup Local Files (Optional)
   # Keep for 24 hours in case training fails
   # Delete after successful training
   
7. Trigger Workflow 6

### **Workflow 6: Launch RunPod Serverless Training**

**Assumption:** You've created custom Docker image: `your-registry/simpletuner-serverless:latest`
```
INPUT: Backblaze dataset URL
OUTPUT: RunPod job_id

1. Prepare RunPod Payload
   {
     "input": {
       "dataset_url": "b2://training-datasets/jobs/[job_id]/package/",
       "config_url": "b2://training-datasets/jobs/[job_id]/package/configs/config.json",
       "output_bucket": "b2://training-models/jobs/[job_id]/",
       "wandb_api_key": "YOUR_WANDB_KEY",
       "wandb_project": "lora-training",
       "wandb_run_name": "[job_id]_sks_luneasterstick",
       "lora_token": "sks_luneasterstick"
     }
   }
   
2. Launch Serverless Endpoint
   POST https://api.runpod.ai/v2/[endpoint_id]/run
   Headers:
     Authorization: Bearer [RUNPOD_API_KEY]
   Body: [payload from step 1]
   
   Response:
   {
     "id": "runpod_job_abc123",
     "status": "IN_QUEUE"
   }
   
3. Update Airtable
   - runpod_job_id: "runpod_job_abc123"
   - training_started_at: timestamp
   - wandb_url: "https://wandb.ai/user/project/runs/[run_id]"
   - status → "training"
   
4. Trigger Workflow 7 (Monitoring)

# In your Docker image: handler.py
import runpod
import subprocess
import os

def download_from_backblaze(url, dest):
    """Download dataset using rclone"""
    subprocess.run([
        "rclone", "copy", url, dest, "--progress"
    ])

def train(job):
    input_data = job['input']
    
    # 1. Download dataset
    download_from_backblaze(
        input_data['dataset_url'],
        "/workspace/dataset"
    )
    
    # 2. Download config
    download_from_backblaze(
        input_data['config_url'],
        "/workspace/config.json"
    )
    
    # 3. Set WandB env
    os.environ['WANDB_API_KEY'] = input_data['wandb_api_key']
    os.environ['WANDB_PROJECT'] = input_data['wandb_project']
    
    # 4. Run training
    subprocess.run([
        "python", "train_sdxl_lora.py",
        "--config", "/workspace/config.json"
    ])
    
    # 5. Upload results
    subprocess.run([
        "rclone", "copy", "/workspace/output/",
        input_data['output_bucket'], "--progress"
    ])
    
    return {
        "status": "success",
        "output_url": input_data['output_bucket']
    }

runpod.serverless.start({"handler": train}) -- when we get to this step of the process I have a lot of python automations for simple tuner and diffusion pipe that might be useful we will need to adjust this reference material for the serverless endpoint

### **Workflow 7: Monitor Training (WandB)**
```
TRIGGER: Polling every 60 seconds while status = "training"

1. Check RunPod Status
   GET https://api.runpod.ai/v2/[endpoint_id]/status/[job_id]
   
   Response:
   {
     "status": "IN_PROGRESS",
     "executionTime": 3600  // seconds
   }
   
2. Fetch WandB Metrics
   Using WandB API:
   GET https://api.wandb.ai/api/v1/runs/[run_id]
   
   Metrics:
   - current_step: 1250 / 2776
   - loss: 0.087
   - learning_rate: 4e-4
   - samples_per_second: 2.1
   - estimated_time_remaining: "1h 15m"
   
3. Download Latest Validation Images
   From WandB or Backblaze (if saving intermediates)
   
4. Update Airtable
   - training_progress: "45%"
   - current_step: "1250/2776"
   - current_loss: 0.087
   - eta: "1h 15m"
   - last_updated: timestamp
   
5. Check for Completion
   IF status = "COMPLETED":
     → Stop polling
     → Update Airtable: status → "training_complete"
     → Trigger Workflow 8
   
6. Check for Failures
   IF status = "FAILED":
     → Log error
     → Update Airtable: status → "failed"
     → Send alert to Slack
     → Stop workflow
```

---