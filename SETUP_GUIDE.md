# PhotoRestore App — Complete Setup Guide
## Remini-like Photo Restoration | No GPU Required

---

## 🏗️ Architecture Overview

```
[Android App]  ──POST image──▶  [FastAPI on Render.com]  ──API call──▶  [Replicate.com AI]
                ◀──base64 img──                           ◀──result URL──  GFPGAN + Real-ESRGAN
```

**Cost:** FREE tier for low usage (Render free + Replicate free credits)

---

## STEP 1 — Get a Replicate API Key (Free)

1. Go to https://replicate.com and sign up (free)
2. Go to Account → API Tokens
3. Copy your token — looks like: `r8_xxxxxxxxxxxxxxxx`
4. You get free credits to start. Pay-as-you-go after that (~$0.001 per image)

---

## STEP 2 — Set Up the Python Backend in PyCharm

### 2a. Open the `photo-restore-backend` folder in PyCharm

### 2b. Create a virtual environment
In PyCharm Terminal:
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Mac/Linux
```

### 2c. Install dependencies
```bash
pip install -r requirements.txt
```

### 2d. Test locally
```bash
# Set your Replicate API key (Windows PowerShell)
$env:REPLICATE_API_TOKEN = "r8_your_token_here"

# Run the server
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 — you should see:
```json
{"status": "ok", "message": "PhotoRestore API is running"}
```

### 2e. Test the API with Swagger UI
Open http://localhost:8000/docs
- Click POST /restore → Try it out → Upload a photo → Execute
- You'll see the base64 encoded result

---

## STEP 3 — Deploy Backend to Render.com (FREE hosting)

### 3a. Push backend to GitHub
```bash
cd photo-restore-backend
git init
git add .
git commit -m "Initial PhotoRestore backend"
# Create repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/photo-restore-backend.git
git push -u origin main
```

### 3b. Deploy on Render.com
1. Go to https://render.com → Sign Up (free)
2. New → Web Service → Connect GitHub repo
3. Settings:
   - **Name:** photo-restore-api
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Under **Environment Variables** → Add:
   - Key: `REPLICATE_API_TOKEN`
   - Value: `r8_your_token_here`
5. Click **Create Web Service**
6. Wait ~3 minutes for deploy
7. Your URL will be: `https://photo-restore-api.onrender.com`

### 3c. Test your live API
```
https://photo-restore-api.onrender.com/health
```
Should return: `{"status": "healthy"}`

---

## STEP 4 — Set Up Android App in Android Studio

### 4a. Open `android-app` folder in Android Studio

### 4b. Update your backend URL
Open: `app/src/main/java/com/photorestore/app/api/RetrofitClient.kt`

Change line:
```kotlin
private const val BASE_URL = "https://YOUR_RENDER_APP_NAME.onrender.com/"
```
To your actual Render URL:
```kotlin
private const val BASE_URL = "https://photo-restore-api.onrender.com/"
```

### 4c. Add theme resource
Create `app/src/main/res/values/themes.xml`:
```xml
<resources>
    <style name="Theme.PhotoRestore" parent="Theme.MaterialComponents.DayNight.DarkActionBar">
        <item name="colorPrimary">#7C4DFF</item>
        <item name="colorPrimaryVariant">#5600E8</item>
        <item name="colorOnPrimary">#FFFFFF</item>
    </style>
</resources>
```

### 4d. Sync Gradle
Click "Sync Now" in the yellow banner that appears.

### 4e. Run on your phone
- Enable Developer Options on your Android phone
- Enable USB Debugging
- Connect via USB
- Click ▶ Run in Android Studio

---

## STEP 5 — Using the App

1. Open **PhotoRestore** on your phone
2. Tap **"Pick Photo"** — choose an old/blurry photo
3. Tap **"✨ Restore"**
4. Choose mode:
   - **Face + Full Restore** → Best for faces, old family photos
   - **Enhance Only** → Landscapes, animals, objects
5. Wait 20–60 seconds (Replicate AI processing)
6. View result → **Save** to gallery or **Share**

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| App can't connect | Check your Render URL in RetrofitClient.kt |
| "REPLICATE_API_TOKEN not configured" | Add env var in Render dashboard |
| Timeout error | Render free tier sleeps after inactivity — wait 30s and retry |
| "Gradle sync failed" | File → Invalidate Caches → Restart |
| Build error on `binding` | Enable viewBinding in build.gradle (already done) |

---

## 💰 Cost Breakdown

| Service | Free Tier | After Free |
|---------|-----------|------------|
| Render.com | Free (sleeps after 15min inactivity) | $7/mo to keep awake |
| Replicate.com | ~$1 free credit on signup | ~$0.0023/prediction |
| Total for 100 restorations | FREE | ~$0.23 |

---

## 🚀 Upgrades You Can Add Later

- **Before/After slider** — Show original vs restored side by side
- **Batch processing** — Restore multiple photos at once
- **History screen** — Save and view past restorations (Room DB)
- **Colorization** — Add color to black & white photos (different Replicate model)
- **Remove scratches** — Use `microsoft/bringing-old-photos-back-to-life` on Replicate

---

## 📦 Models Used

| Model | Purpose | Replicate ID |
|-------|---------|-------------|
| GFPGAN v1.4 | Face restoration, detail recovery | `tencentarc/gfpgan` |
| Real-ESRGAN | Super-resolution 2x/4x upscaling | `nightmareai/real-esrgan` |

These are the same class of models that power apps like Remini and Enhancer AI.
