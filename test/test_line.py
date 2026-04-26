# test_line.py
import os
from dotenv import load_dotenv
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, PushMessageRequest, TextMessage
from typing import Any, cast

load_dotenv()

def quick_test():
    conf = Configuration(host="https://api.line.me", access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
    with ApiClient(conf) as api_client:
        line_bot_api = MessagingApi(api_client)
        msg = cast(Any, TextMessage)(text="測試訊息：AI 助理連線成功！")
        req = cast(Any, PushMessageRequest)(to=os.getenv("LINE_USER_ID"), messages=[msg])
        line_bot_api.push_message(req)
        print("✅ 測試成功！請查看手機。")

if __name__ == "__main__":
    quick_test()