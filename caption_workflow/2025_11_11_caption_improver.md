# Technical Context: n8n Image Captioning Workflow for LoRA Training Pipeline

## Project Overview
I'm building an automated LoRA training pipeline that processes user-submitted datasets through multiple stages: path normalization, image preprocessing, captioning, config generation, cloud upload, RunPod training, and post-training analysis. The captioning workflow is a critical middle stage that needs to be production-ready and integrate seamlessly with the broader system.

---

## Current Workflow: captioner (v6)

### Architecture
**Flow:** Manual Trigger → Setup Config → List Images → Prepare Batch → Read Image → Convert to Base64 → Caption → Optional Subject Removal → Quality Scoring → Optional Refinement → Save .txt Files → Summary Report

### What's Working
- ✅ Manual trigger with editable settings (directory_path, lora_token, prompt_template, enable_refinement)
- ✅ PowerShell-based image file listing (Windows/Linux compatible)
- ✅ Batch preparation that creates individual items per image
- ✅ Base64 data URI conversion for OpenAI Vision API
- ✅ OpenAI GPT-4o captioning with configurable prompt templates (general, object, character, style, scene)
- ✅ LoRA token enforcement in prompts
- ✅ Conditional subject removal (for character/object templates)
- ✅ Quality scoring with 9 validation checks
- ✅ Optional caption refinement based on quality threshold
- ✅ Re-scoring after refinement
- ✅ Writing .txt files next to images
- ✅ Comprehensive summary report with LoRA token validation metrics

### Quality Checks (9 total)
1. `hasMinLength`: Caption >= 10 words
2. `hasMaxLength`: Caption <= 200 words
3. `hasSubject`: Contains subject words (person, character, object, etc.)
4. `hasVisualDetails`: Contains visual detail words (color, light, texture, etc.)
5. `noErrors`: No error language (error, unable, cannot, sorry)
6. `notTooVague`: Doesn't start with generic phrases (This is, The image shows)
7. `hasLighting`: Contains lighting words (light, shadow, bright, dark, etc.)
8. `hasComposition`: Contains composition words (foreground, angle, perspective, etc.)
9. `loraTokenValid`: LoRA token appears exactly once at the start

---

## Integration Requirements for Broader Pipeline

### Input (Future State - from Workflow 2: Image Preprocessor)
The captioner will eventually receive:
```json
{
  "job_id": "job_1762875194225",
  "normalized_path": "D:/photos/bottles",
  "processed_dir": "D:/photos/bottles/job_123_sks_fragrancebottle_SDXL_v1",
  "needs_preprocessing": true,
  "lora_token": "sks_fragrancebottle",
  "model_base": "SDXL",
  "training_type": "object",
  "image_count": 247,
  "custom_config": null,
  "validation_prompts": null
}
```

**Key Points:**
- Should work with either `normalized_path` (if no preprocessing) OR `processed_dir` (if preprocessing happened)
- Should handle nested directory structures (aspect buckets like `512x512/`, `1024x1024/`)
- Must preserve file structure when saving .txt files
- Must track progress and update Airtable (not yet implemented)

### Output (To Workflow 4: Config Generator)
Should produce:
```json
{
  "job_id": "job_1762875194225",
  "captions_completed": 247,
  "caption_accuracy": 98.5,
  "captioning_cost": 0.49,
  "captioning_report": {
    "total_images": 247,
    "successful": 245,
    "failed": 2,
    "lora_token_accuracy": 0.985,
    "avg_caption_length": 32.4,
    "captioning_cost_usd": 0.49,
    "failed_files": [...]
  },
  "status": "captioned"
}
```

### File Structure Handling
Current:
```
D:/ai/scripts/n9n/test_images/post/subject_lora/
├─ image001.jpg
├─ image002.jpg
└─ image003.jpg
```

Future (after preprocessing):
```
D:/photos/bottles/job_123_sks_fragrancebottle_SDXL_v1/
├─ 1024x1024/
│  ├─ bottle001.png
│  ├─ bottle001.txt  ← Must create
│  ├─ bottle002.png
│  └─ bottle002.txt  ← Must create
└─ 1024x1536/
   ├─ bottle003.png
   └─ bottle003.txt  ← Must create
```

**Must handle:**
- Flat directories (current)
- Nested bucket directories (future)
- Mixed file extensions (.jpg, .jpeg, .png, .webp)
- Preserving directory structure when saving .txt files

---

## Known Issues & Desired Improvements

### 1. Platform Compatibility
**Current:** Uses PowerShell (`Get-ChildItem`) which fails on pure Linux systems
**Desired:** Use a cross-platform approach (Python `os.walk()` or platform detection)

### 2. Batch Processing & Rate Limiting
**Current:** Processes images sequentially (one at a time)
**Issues:**
- No rate limiting for OpenAI API (risk of hitting 5,000 req/min limit)
- No batch progress tracking
- No resume capability if workflow fails mid-batch

**Desired:**
- Process in batches of 50 images
- Add 6-second delay between batches (OpenAI rate limit buffer)
- Save after each batch to prevent data loss
- Optional: Update Airtable with progress every batch

### 3. Error Handling & Retry Logic
**Current:** `continueOnFail: true` on OpenAI nodes, but no retry mechanism
**Desired:**
- Retry failed API calls (3 attempts with exponential backoff)
- Collect failed images in a separate list
- Continue processing other images even if some fail
- Generate detailed error report

### 4. Cost Tracking
**Current:** No cost calculation
**Desired:** Calculate cost based on:
- Image count × $0.002 per image (gpt-4o pricing)
- Include in summary report

### 5. Additional QC Checks (User Mentioned)
**Current:** 9 quality checks
**User mentioned:** "I have a couple more qc steps than you have outlined"
**Need to understand:**
- What additional QC checks are needed?
- Are they image-based (resolution, aspect ratio, file size)?
- Are they caption-based (specific keywords, sentiment, length variance)?
- Should they block processing or just flag issues?

### 6. Memory Efficiency
**Current:** Loads all images into memory as base64
**Concern:** Large datasets (1000+ images) could cause memory issues
**Desired:** Process images in streaming fashion or clear memory after each batch

### 7. Nested Directory Support
**Current:** Works with flat directories only
**Future Need:** Must recursively process subdirectories (aspect buckets)

### 8. Progress Reporting
**Current:** Only final summary report
**Desired:**
- Real-time progress updates (console or Airtable)
- ETA calculation
- Current batch / total batches
- Images processed / total images

---

## Technical Constraints

### Environment
- **Platform:** n8n running on Linux (cloud-hosted)
- **Node.js Version:** Compatible with n8n (typically v18+)
- **Available Libraries:** Limited to built-in Node.js modules + what n8n provides
- **Python:** Not directly available (must use executeCommand if needed)

### n8n Specifics
- **Binary Data Handling:** Images converted to base64 in `binary.data` property
- **Item Processing:** Each item in batch has `.json` and optional `.binary` properties
- **Node References:** Can reference previous nodes with `$node['NodeName'].json` or `$('NodeName').item.json`
- **Error Handling:** Use `continueOnFail: true` and check for `error` property in results

### OpenAI API
- **Model:** gpt-4o (vision-capable)
- **Rate Limits:** 5,000 requests/min (Tier 4)
- **Cost:** ~$0.002 per image with "high" detail
- **Max Tokens:** 300 per response (sufficient for captions)
- **Input Format:** Supports base64 data URIs

---

## Current Workflow Strengths (To Preserve)

1. **Modular Design:** Clear separation of concerns (setup, list, prepare, caption, score, save)
2. **Quality Scoring System:** Comprehensive 9-point validation
3. **LoRA Token Validation:** Strict enforcement of token placement
4. **Conditional Processing:** Subject removal and refinement only when needed
5. **Detailed Reporting:** Rich summary with per-image results
6. **File Path Handling:** Handles both Windows and Unix paths

---

## Questions to Address

### For Immediate Testing
1. **What additional QC steps do you need?** (Beyond the 9 current checks)
2. **Should we add batch processing now or later?** (For testing with small datasets, current sequential processing is fine)
3. **Do you want Airtable integration now or after testing?** (Can be added as separate nodes later)
4. **What size datasets are you testing with?** (Determines if memory optimization is urgent)

### For Production Readiness
5. **How should failed images be handled?** (Skip and continue, or halt workflow?)
6. **Should there be a human review step?** (Pause workflow to review sample captions?)
7. **What's the target processing time?** (100 images = ~10 minutes with current sequential approach)
8. **Should captions be editable after generation?** (Save to Airtable for manual edits?)

---

## Specific Improvements Requested

### Priority 1: Make It Work Today
- [ ] Fix any blocking issues preventing execution
- [ ] Verify .txt files are created correctly
- [ ] Confirm quality scoring works as expected
- [ ] Test with your sample dataset (D:/ai/scripts/n9n/test_images/post/subject_lora)

### Priority 2: Production Hardening (This Week)
- [ ] Add cross-platform file listing (replace PowerShell)
- [ ] Implement nested directory support (for aspect buckets)
- [ ] Add batch processing (50 images per batch)
- [ ] Add rate limiting (6-second delays)
- [ ] Implement retry logic (3 attempts)
- [ ] Add cost calculation

### Priority 3: Pipeline Integration (Next Week)
- [ ] Accept input from previous workflow (Workflow 2)
- [ ] Output data for next workflow (Workflow 4)
- [ ] Add Airtable updates (status, progress, results)
- [ ] Add error reporting to Slack
- [ ] Implement resume capability (skip already-captioned images)

---

## Example Test Case

### Input
```
directory_path: D:/ai/scripts/n9n/test_images/post/subject_lora
lora_token: sks_luneasterstick
prompt_template: object
enable_refinement: false
```

### Expected Output
- ✅ All .jpg/.png files in directory are captioned
- ✅ Each caption starts with "sks_luneasterstick"
- ✅ .txt files are created next to each image
- ✅ Quality scores are >= 0.75 (or flagged for manual review)
- ✅ Summary report shows:
  - Total images processed
  - Success/failure count
  - LoRA token accuracy
  - Average caption length
  - Any issues encountered

---

## Code Patterns to Follow

### Error Handling
```javascript
// In OpenAI nodes
continueOnFail: true

// In processing nodes
if (input.error || !input.choices) {
  return {
    image_id: imageId,
    caption: null,
    error: input.error?.message || 'API call failed',
    success: false
  };
}
```

### File Path Handling
```javascript
// Normalize paths
const filepath = `${directoryPath}/${filename}`.replace(/\\/g, '/');

// Create .txt path
const txtPath = filepath.replace(/\.[^/.]+$/, '.txt');
```

### Quality Scoring Pattern
```javascript
const checks = {
  check1: boolean,
  check2: boolean,
  // ... etc
};

const qualityScore = Object.values(checks).filter(Boolean).length / Object.keys(checks).length;
```

---

## Success Criteria

### For Today's Testing
1. Workflow executes without errors
2. .txt files are created in correct locations
3. Captions start with LoRA token
4. Summary report is accurate

### For Production
1. Handles 1000+ images without memory issues
2. Completes 100 images in < 15 minutes
3. Recovers from API failures gracefully
4. Integrates with broader pipeline seamlessly
5. Provides real-time progress updates

---

## Files Attached
- `captioner__6_.json` - Current n8n workflow

## Request
Please review the current workflow and suggest improvements to:
1. Make it robust for production use (error handling, retry logic, batch processing)
2. Ensure it integrates with the broader pipeline architecture described
3. Add any missing QC steps or validation
4. Optimize for performance and memory efficiency
5. Ensure cross-platform compatibility (Linux + Windows paths)

Focus on getting it working reliably TODAY for testing, with a path to production hardening this week.