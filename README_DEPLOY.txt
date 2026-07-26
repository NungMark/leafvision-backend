Render settings:

Build Command:
pip install -r requirements.txt

Start Command:
uvicorn main:app --host 0.0.0.0 --port $PORT

Environment Variables on Render:
SECRET_KEY=leafvision-ai-prod-secret-change-this-long-random-string
FRONTEND_URL=https://your-frontend.vercel.app

After deploy, test:
https://your-backend.onrender.com/health
https://your-backend.onrender.com/docs
