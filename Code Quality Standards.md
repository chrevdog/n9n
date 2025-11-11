## Code Quality Standards

### Node.js/JavaScript Conventions
- Use `const` over `let` where possible
- Avoid `var` entirely
- Use arrow functions for callbacks: `array.map(item => ...)`
- Use template literals: `` `${path}/${file}` ``
- Handle errors explicitly: `if (error) { ... }`
- Use `try/catch` for async operations

### n8n-Specific Patterns
- Reference previous nodes: `$node['NodeName'].json`
- Access item data: `$('NodeName').item.json`
- Handle arrays: `$input.all()` for all items
- Binary data: `$binary.data` for base64
- Set node names descriptively: "Caption Images with GPT-4o" not "Function1"

### File System Operations
- Always normalize paths: `path.replace(/\\/g, '/')`
- Check file existence before reading: `fs.existsSync()`
- Use absolute paths in n8n expressions
- Handle ENOENT errors gracefully

### Error Messages
- Be specific: "Failed to write caption file to /path/to/file.txt: Permission denied"
- Include context: filename, operation, error code
- Log to both console and error collection
- Never expose API keys or sensitive data in logs
