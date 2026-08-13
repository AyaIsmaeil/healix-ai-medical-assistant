# Healix — مساعد طبي (مشروع تخرج)

**ابدأ من:** [`ابدأ_هنا.md`](ابدأ_هنا.md)

خدمة FastAPI: مقابلة سريرية بالعربية → تقييم أولي (مرض + خطورة + تخصص).

```powershell
cd C:\healix-ai-medical-assistant
.\venv\Scripts\Activate.ps1
copy .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Swagger: http://localhost:8000/api/docs
