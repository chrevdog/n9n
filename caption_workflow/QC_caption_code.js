const it = $input.item.json;
if (!it.success) return it;

const mode = it.config.prompt_template;
if (mode !== 'lora_object' && mode !== 'lora_character') {
  // Pass-through for style/edit
  return it;
}

// Instruction to model for rewrite
const system = `Rewrite captions for LoRA training so they only describe environment, lighting, and camera–subject relationship.
Keep the subject token as-is ("${mode==='lora_object' ? it.config.object_token : it.config.character_token}") but remove specific physical traits, materials, and fixed identity details.
Keep one sentence if possible, objective and factual.`;

const user = `Original: ${it.caption}
Rewrite: keep only environment/lighting/camera–subject relationship; preserve the token; remove object/character specifics.`;

try {
  const resp = await $httpRequest({
    method: 'POST',
    url: 'https://api.openai.com/v1/chat/completions',
    authentication: { type: 'predefinedCredentialType', name: 'openAiApi' },
    body: {
      model: 'gpt-4o',
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
      max_tokens: 220,
      temperature: 0.2
    },
    json: true
  });

  const rewritten = resp.choices[0].message.content.trim();
  return { ...it, caption: rewritten, qc_rewritten: true };
} catch (err) {
  return { ...it, qc_rewritten: false, qc_error: err?.message || 'QC rewrite failed' };
}
