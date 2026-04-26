import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    print(f"✅ 成功讀取 API Key: {api_key[:10]}******")
else:
    print("❌ 失敗！找不到 GOOGLE_API_KEY，請檢查 .env 檔案路徑或內容。")