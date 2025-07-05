# Setting up HuggingFace Token for Subtitle AI Suite

## Why do you need a HuggingFace token?

The Subtitle AI Suite uses advanced SpeechBrain models from HuggingFace for:
- Speaker recognition and diarization
- Emotion detection
- Gender detection
- Enhanced speaker analysis

Some of these models require authentication to access.

## How to get your HuggingFace token:

1. **Create a HuggingFace account** (if you don't have one):
   - Go to https://huggingface.co/join
   - Sign up for a free account

2. **Generate an access token**:
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Give it a name like "subtitle-ai-suite"
   - Select "Read" permissions
   - Click "Generate a token"
   - **Copy the token immediately** (you won't be able to see it again)

## How to configure the token:

### Option 1: Environment Variable (Recommended)
```bash
export HF_TOKEN=your_token_here
```

Add this to your shell profile (.bashrc, .zshrc, etc.) to make it permanent.

### Option 2: .env File
```bash
# Copy the example file
cp .env.example .env

# Edit .env file and add your token
echo "HF_TOKEN=your_token_here" >> .env
```

### Option 3: Set in current session
```bash
# Linux/Mac
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Windows Command Prompt
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Windows PowerShell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## Verify the token is working:

Run the subtitle AI suite and you should see:
```
Successfully authenticated with HuggingFace using HF_TOKEN
```

Instead of authentication errors.

## Security Notes:

- **Never commit your token to git**
- Keep your token private
- Use environment variables or .env files (which are gitignored)
- Regenerate your token if it gets compromised

## Troubleshooting:

- **"HF_TOKEN not found"**: The environment variable isn't set
- **"Failed to authenticate"**: The token is invalid or expired
- **"401 Client Error"**: You don't have access to the specific model

## What happens without a token?

The system will still work but:
- Some advanced speaker features may be unavailable
- You'll see warning messages
- Basic speaker diarization will still function