@echo off
echo Starting Medical Chatbot...

cd backend
start /B python -m uvicorn main:app --host 0.0.0.0 --port 8000
echo Backend started.

timeout /t 3

cd ..\frontend
start /B python -m http.server 3000
echo Frontend started.

echo ========================================
echo Medical Chatbot is running!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo ========================================
