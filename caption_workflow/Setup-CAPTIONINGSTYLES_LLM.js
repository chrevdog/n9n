// Setup and validation (LoRA-specialized, no hard length bound up front)
const body = $input.item.json.body;
const images = body.images || [];
if (!images || images.length === 0) throw new Error('No images provided');

const cfg = body.config || {};
const config = {
  prompt_template: cfg.prompt_template || 'lora_object', // default to LoRA object
  detail_level: cfg.detail_level || 'high',
  enable_refinement: cfg.enable_refinement !== false,
  quality_threshold: cfg.quality_threshold ?? 0.85,
  max_caption_length: cfg.max_caption_length || 300, // generous first pass

  // tokens
  object_token: cfg.object_token || 'sks_object',
  character_token: cfg.character_token || 'sks_character',
  style_token: cfg.style_token || 'sks_stylelora',

  // QC token cap (applied only after rewrite)
  enforce_token_cap_in_qc: cfg.enforce_token_cap_in_qc !== false,
  token_cap: cfg.token_cap || 75,

  // guardrails (still ban styley/brand words)
  banned_terms: cfg.banned_terms || [
    'brand','logo','cinematic','beautiful','aesthetic','editorial','premium','luxury',
    'instagram','advertisement','ad','famous','award','eco','vegan','stock photo'
  ]
};

const prompts = {
  lora_object:
`Describe this image for object-accurate LoRA training using the token "{{OBJ}}".
One concise sentence preferred (but not required). Avoid brand/style/emotion.
Include lighting environment (direction/quality/intensity) with shadows/reflections; and camera–subject relationship (angle, distance, macro/tele/normal feel, depth of field, framing).`,

  lora_character:
`Describe this image for character-accurate LoRA training using the token "{{CHR}}".
One concise sentence preferred (but not required). Avoid brand/style/emotion.
Include pose/clothing/accessories if visible; lighting environment (direction/quality/intensity); background/context; and camera–subject relationship (angle, distance, focal length feel, depth of field, framing).`,

  lora_style:
`Describe the overall scene, dominant color palette, and quality of light (direction, softness/hardness, contrast, time-of-day feel); avoid brand names; end with: "in the style of {{STYLE}}".`,

  lora_edit:
`Write one imperative instruction that describes ONLY the desired visual change to the input image (subject/region/attributes/lighting), where it applies, and magnitude; explicitly keep everything else unchanged.`
};

const template = (t) => t
  .replaceAll('{{OBJ}}', config.object_token)
  .replaceAll('{{CHR}}', config.character_token)
  .replaceAll('{{STYLE}}', config.style_token);

const chosen = template(prompts[config.prompt_template] || prompts.lora_object);

return {
  images,
  config,
  prompt: chosen,
  job_id: body.job_id || ('job_' + Date.now()),
  started_at: new Date().toISOString()
};
