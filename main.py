import atexit
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from typing import Any, cast
from urllib.request import urlopen

import yfinance as yf
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    PushMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent


# 載入環境變數
load_dotenv()

# Flask app（讓 Line Webhook 透過 ngrok 打進來）
app = Flask(__name__)

# LineBot API / Webhook 設定
line_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
line_channel_secret = os.getenv("LINE_CHANNEL_SECRET")
configuration = Configuration(host="https://api.line.me", access_token=line_access_token)
handler = WebhookHandler(line_channel_secret)

# --- 1. 初始化工具與模型 ---

# 網路搜尋工具 (需在 .env 設定 SERPER_API_KEY)
search_tool = SerperDevTool()

# Gemini 模型（請確認此 model 在你的專案中可用）
gemini_llm = LLM(
    model="gemini/gemini-3-flash-preview",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

from FinMind.data import DataLoader

@tool("fetch_taiwan_chip_data")
def fetch_taiwan_chip_data(ticker: str):
    """
    抓取台股特有的三大法人籌碼與融資融券數據，並做簡單解讀。
    ticker: 例如 '2330' (注意：FinMind 不需要加 .TW)
    """
    dl = DataLoader()
    # 若有設定 FinMind token，則登入以提高 API 穩定性
    finmind_token = os.getenv("FINMIND_TOKEN")
    if finmind_token:
        try:
            dl.login_by_token(api_token=finmind_token)
        except Exception as e:
            # 不因登入失敗中斷，僅作紀錄
            print(f"⚠️ FinMind 登入失敗：{e}")

    # 動態設定起始日：抓最近約 120 天的資料
    end_date = datetime.today().date()
    start_date = (end_date - timedelta(days=120)).strftime("%Y-%m-%d")

    # 1. 抓取三大法人買賣超
    df_chips = dl.taiwan_stock_institutional_investors(
        stock_id=ticker,
        start_date=start_date,
    )
    if df_chips is None or df_chips.empty:
        return {
            "ticker": ticker,
            "start_date": start_date,
            "error": "無法取得三大法人籌碼資料，可能是代碼有誤或 FinMind 暫時無資料。",
        }

    df_chips = df_chips.sort_values("date")
    latest_chips = df_chips.tail(5)

    # 計算近 5 日外資、投信、自營的總淨買超
    def _net(group):
        return float((group["buy"] - group["sell"]).sum())

    chips_summary = {}
    for inst_name, label in [
        ("Foreign_Investor", "外資"),
        ("Investment_Trust", "投信"),
        ("Dealer", "自營商"),
    ]:
        sub = latest_chips[latest_chips["name"] == inst_name]
        if not sub.empty:
            chips_summary[f"{label}近5日淨買超"] = _net(sub)

    # 投信連三日買超判斷
    it_recent3 = latest_chips[latest_chips["name"] == "Investment_Trust"].tail(3)
    it_consecutive_buy3 = False
    if len(it_recent3) == 3:
        net_flow = it_recent3["buy"] - it_recent3["sell"]
        it_consecutive_buy3 = bool((net_flow > 0).all())

    # 2. 抓取融資融券 (觀察散戶動向)
    df_margin = dl.taiwan_stock_margin_purchase_short_sale(
        stock_id=ticker,
        start_date=start_date,
    )
    margin_info: dict[str, Any] = {}
    if df_margin is not None and not df_margin.empty:
        df_margin = df_margin.sort_values("date")
        latest_margin = df_margin.tail(5)

        # 近 3 日融資變化趨勢
        last3 = latest_margin.tail(3)
        if len(last3) >= 2:
            first_val = float(last3["MarginPurchaseRemain"].iloc[0])
            last_val = float(last3["MarginPurchaseRemain"].iloc[-1])
            if last_val > first_val * 1.02:
                trend = "上升（散戶融資增加，風險可能放大）"
            elif last_val < first_val * 0.98:
                trend = "下降（融資減少，籌碼可能集中）"
            else:
                trend = "持平"
            margin_info = {
                "近3日融資餘額起點": first_val,
                "近3日融資餘額終點": last_val,
                "近3日融資趨勢": trend,
            }
        else:
            margin_info = {"近3日融資趨勢": "資料不足"}
    else:
        margin_info = {"error": "無法取得融資融券資料"}
    
    # 3. 統一整理輸出，方便 Agent 閱讀與引用
    return {
        "ticker": ticker,
        "資料區間": {
            "start_date": start_date,
            "end_date": str(end_date),
        },
        "三大法人近5日摘要": {
            **chips_summary,
            "投信是否連3日淨買超": it_consecutive_buy3,
        },
        # 轉成列表讓 LLM 比較好閱讀
        "近期法人明細": latest_chips[["date", "name", "buy", "sell"]].to_dict(),
        "融資融券摘要": margin_info,
        "解讀建議": (
            "若投信連續買超且融資下降，代表籌碼趨於集中、較有利於上漲；"
            "反之，若投信轉為賣超且融資快速增加，需警惕短線追高與籌碼鬆動風險。"
        ),
    }

# yfinance 股價工具
@tool("fetch_stock_data")
def fetch_stock_data(ticker: str):
    """
    抓取指定美股或台股代碼的即時股價與基本面資訊。
    美股範例: 'NVDA', 'AAPL'；台股範例: '2330.TW', '2454.TW'
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    data = {
        "目前價格": info.get("currentPrice"),
        "52週高點": info.get("fiftyTwoWeekHigh"),
        "52週低點": info.get("fiftyTwoWeekLow"),
        "本益比 (P/E)": info.get("trailingPE"),
        "市值": info.get("marketCap"),
        "分析師建議": info.get("recommendationKey"),
    }
    return f"股票 {ticker} 的即時數據如下：\n{data}"


def run_investment_analysis(stock_target: str) -> str:
    """建立 Crew 並針對指定標的跑完整投資分析流程，回傳報告字串。"""
    alpha_agent = Agent(
        role="策略獵人 Alpha",
        goal="整合即時股價、籌碼與網路新聞，尋找台美股具備動能且風報比合理的進場點。",
        backstory=(
            "你是一名量化與基本面兼具的策略分析師。"
            "在做出任何結論前，你會嚴格依照以下流程："
            "1) 先呼叫工具取得股價、估值與籌碼等客觀數據；"
            "2) 再利用搜尋工具比對近期重大新聞與市場情緒；"
            "3) 對數據與新聞中出現的矛盾之處進行解釋與權重評估；"
            "4) 明確說明你的假設與不確定性，避免絕對語氣與無根據的猜測；"
            "5) 最後以『短線/中長線』情境拆解可能路徑與勝率、風報比。"
            "你非常討厭憑感覺下結論，只信可重複驗證的數據與邏輯推演。"
        ),
        tools=[fetch_taiwan_chip_data, fetch_stock_data, search_tool],
        llm=gemini_llm,
        verbose=True,
    )

    risk_agent = Agent(
        role="風控判官 Risk",
        goal="針對 Alpha 的提案進行壓力測試，找出潛在的利空因素、情境風險與最大虧損範圍。",
        backstory=(
            "你是極度紀律的風控總監，專長是拆解過度樂觀的投資故事。"
            "你會系統性地："
            "1) 逐點檢查 Alpha 引用的每一項數據與新聞是否來自工具輸出的客觀資訊；"
            "2) 對估值、籌碼、流動性與總經情境做壓力測試，假設最壞情境會發生；"
            "3) 明確量化風險（如合理的回撤區間、停損價位與部位上限），不接受模糊的『應該還好』；"
            "4) 一旦關鍵數據不足或衝突，你會要求保守對待，標註為『資訊不足不建議重倉』；"
            "5) 只有在風報比、資金管理與風險集中度都可接受時，才會給出『通過』決策。"
            "你的首要任務是避免大虧損，而不是追求最大報酬。"
        ),
        tools=[fetch_taiwan_chip_data, fetch_stock_data, search_tool],
        llm=gemini_llm,
        verbose=True,
    )

    task1 = Task(
        description=(
            f"調查 {stock_target} 的最新股價數據與過去 24 小時的網路重大新聞。"
            "判斷目前市場情緒是否過熱，並給出買入建議。"
        ),
        expected_output="一份結合數據與新聞情緒的詳細投資計畫。",
        agent=alpha_agent,
    )

    task2 = Task(
        description=(
            f"審核 Alpha 對 {stock_target} 的建議。精簡且列出至少 3 個負面風險點"
            "（例如總經、技術面過熱或地緣政治），最後給予『通過』或『拒絕』的決策。"
        ),
        expected_output="最終決策報告（包含決策理由）。",
        agent=risk_agent,
    )

    crew = Crew(
        agents=[alpha_agent, risk_agent],
        tasks=[task1, task2],
    )

    print("🚀 AI 助理正在討論中...")
    result = crew.kickoff()

    print("\n########################")
    print("## 最終投資決策報告 ##")
    print("########################\n")
    print(result)

    # Crew 可能回傳物件，這裡保險轉成字串
    return str(result)


# --- 2. 定義 Line 發送/回覆函數 ---

def send_line_to_user(content: str):
    """將結果推播至指定的 Line User ID（單純通知用）。"""
    user_id = os.getenv("LINE_USER_ID")

    if not line_access_token or not user_id:
        print("❌ 錯誤：找不到 LINE_CHANNEL_ACCESS_TOKEN 或 LINE_USER_ID")
        return

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        text_message = cast(Any, TextMessage)(
            text=str(content)[:3500],
        )
        push_message_request = cast(Any, PushMessageRequest)(
            to=user_id,
            messages=[text_message],
        )
        try:
            line_bot_api.push_message(push_message_request)
            print("✅ 投資報告已成功傳送到 Line")
        except Exception as e:
            print(f"❌ Line 傳送失敗: {e}")


# --- 3. Line Webhook + Flask 路由設定 ---


@app.route("/callback", methods=["POST"])
def callback():
    """Line Webhook 入口，由 ngrok 對外暴露。"""
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    """收到使用者文字時，跑一次投資分析並把結果推送回去。"""
    user_text = (getattr(event.message, "text", "") or "").strip()
    stock_target = user_text if user_text else "NVIDIA (NVDA)"

    print(f"📩 收到使用者訊息: {user_text!r}，開始分析標的: {stock_target}")
    try:
        result = run_investment_analysis(stock_target)
        send_line_to_user(result)
    except Exception as e:
        error_msg = f"分析過程發生錯誤：{e}"
        print(f"❌ {error_msg}")
        send_line_to_user(error_msg)


# --- 4. 透過 ngrok + Flask 在 8000 埠啟動 ---

def _get_ngrok_public_url(timeout_s: float = 12.0) -> str | None:
    """從 ngrok 本機 API 取得 public_url（需要 ngrok 正在跑）。"""
    deadline = time.time() + timeout_s
    last_err: Exception | None = None

    while time.time() < deadline:
        try:
            # ngrok 的本機 Web/Api 預設在 4040，可用 NGROK_API_URL 覆蓋
            api_url = os.getenv("NGROK_API_URL", "http://127.0.0.1:4040/api/tunnels")
            with urlopen(api_url, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            tunnels = data.get("tunnels", [])
            for t in tunnels:
                public_url = t.get("public_url", "")
                if isinstance(public_url, str) and public_url.startswith("https://"):
                    return public_url
            for t in tunnels:
                public_url = t.get("public_url", "")
                if isinstance(public_url, str) and public_url.startswith("http://"):
                    return public_url
        except Exception as e:
            last_err = e
            time.sleep(0.5)

    if last_err:
        print(f"⚠️ 讀取 ngrok tunnels 失敗（可忽略）：{last_err}")
    return None


def start_ngrok(port: int) -> subprocess.Popen | None:
    """
    使用系統安裝的 ngrok 啟動 http tunnel（Windows ngrok.exe）。
    需求：ngrok 已安裝且可在 PATH 呼叫；或設定 NGROK_PATH 指到 ngrok.exe。
    """
    ngrok_path = os.getenv("NGROK_PATH", "ngrok")

    try:
        # 強制開啟本機 web/api（4040），避免你 config 關掉 web_addr 造成讀不到 tunnels
        # 若你已有 ngrok 在跑且佔用 4040，本次啟動可能會失敗；我們會在外層先嘗試讀取現有 tunnels。
        proc = subprocess.Popen(
            [
                ngrok_path,
                "http",
                str(port),
                "--web-addr=127.0.0.1:4040",
                "--log=stdout",
                "--log-format=json",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(lambda: proc.poll() is None and proc.terminate())
        return proc
    except FileNotFoundError:
        print("⚠️ 找不到 ngrok。請確認已安裝並加入 PATH，或設定 NGROK_PATH 為 ngrok.exe 的完整路徑。")
        return None
    except Exception as e:
        print(f"⚠️ 無法啟動 ngrok（可忽略）：{e}")
        return None


if __name__ == "__main__":
    port = 8000

    # 先嘗試讀取「已在跑」的 ngrok（避免重複啟動導致 4040 佔用/衝突）
    public_url = _get_ngrok_public_url(timeout_s=2.0)

    # 若讀不到，再嘗試自動啟動 ngrok
    if not public_url:
        _ = start_ngrok(port)
        public_url = _get_ngrok_public_url(timeout_s=15.0)

    if public_url:
        print(f"🌐 ngrok 公開網址: {public_url}/callback")
        print("請將上述 URL 設為 Line Webhook URL。")
    else:
        print(
            "⚠️ 仍無法取得 ngrok 公開網址。\n"
            "- 請確認 ngrok 已登入（`ngrok config add-authtoken ...`）\n"
            "- 若你 ngrok 設定把 web_addr 關掉，請改回開啟或設定 NGROK_API_URL\n"
            "- 你也可以改成手動執行 `ngrok http 8000`，然後把網址填到 `<public>/callback`"
        )

    app.run(host="0.0.0.0", port=port)