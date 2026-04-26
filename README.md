# 🤖 AI Investor — LINE 智能投資助手

一個透過 **LINE 聊天機器人** 與你互動的 AI 投資分析工具。結合 CrewAI 多代理人架構與台股 MCP 資料源，讓你直接在 LINE 中查詢股價、執行深度投資分析。

## ✨ 功能亮點

| 功能 | 說明 |
|------|------|
| 🧠 **AI 深度分析** | 雙 Agent 協作（策略獵人 + 風控判官），自動整合股價、籌碼、新聞情緒 |
| 📈 **即時股價查詢** | 透過 MCP Server 取得台股即時報價 |
| 💰 **公司營收** | 查詢單月營收、月增 / 年增率 |
| 🏢 **基本面資訊** | 資本額、負責人、核心產業歸屬 |
| 🛒 **模擬買入** | 紙上交易測試，不動用真金白銀 |
| 📊 **大盤統計** | 近五分鐘委買 / 委賣 / 成交量概覽 |

---

## 🏗️ 技術架構

```
LINE App ──> ngrok (HTTPS tunnel) ──> Flask (port 8000) ──> main.py
                                                              │
                                          ┌───────────────────┼───────────────────┐
                                          ▼                   ▼                   ▼
                                    CrewAI Agents      MCP Client (SSE)     yfinance / FinMind
                                    (Gemini LLM)             │
                                          │                  ▼
                                          │          CasualMarket MCP Server
                                          │            (Docker, port 8001)
                                          ▼
                                  Serper (Google Search)
```

---

## 📋 前置需求

在開始之前，請確認你已備妥以下工具：

| 工具 | 用途 | 安裝指引 |
|------|------|----------|
| **Python 3.10+** | 執行主程式 | [Anaconda](https://www.anaconda.com/download) (推薦) |
| **Docker Desktop** | 執行台股 MCP Server 容器 | [Docker 官網](https://www.docker.com/products/docker-desktop/) |
| **ngrok** | 將本機服務暴露到公網供 LINE Webhook 使用 | [ngrok 官網](https://ngrok.com/download) |
| **LINE Developer 帳號** | 建立 Messaging API Channel | [LINE Developers](https://developers.line.biz/) |

---

## 🚀 快速開始

### 1. Clone 專案

```bash
git clone https://github.com/<your-username>/AI_Investor.git
cd AI_Investor
```

### 2. 建立 Python 環境並安裝套件

```bash
# 建立 conda 虛擬環境（僅需第一次）
conda create -n ai_investor python=3.13 -y

# 啟動環境
conda activate ai_investor

# 安裝所有依賴套件
pip install -r requirements.txt
```

### 3. 設定環境變數

在專案根目錄下建立 `.env` 檔案（此檔案已被 `.gitignore` 排除，不會被提交）：

```bash
cp .env.example .env   # 或手動建立
```

填入以下欄位：

```dotenv
# === Google Gemini (CrewAI 使用的 LLM) ===
GOOGLE_API_KEY=你的_Google_API_Key

# === Serper (Google 搜尋工具) ===
# 申請：https://serper.dev/
SERPER_API_KEY=你的_Serper_API_Key

# === LINE Messaging API ===
# 從 LINE Developers Console 取得
LINE_CHANNEL_ACCESS_TOKEN=你的_Channel_Access_Token
LINE_CHANNEL_SECRET=你的_Channel_Secret
LINE_USER_ID=你的_LINE_User_ID

# === FinMind (台股籌碼資料) ===
# 申請：https://finmindtrade.com/
FINMIND_API_KEY=Bearer 你的_FinMind_JWT_Token

# === MCP Server 連線位址 (通常不需修改) ===
MCP_SERVER_URL=http://localhost:8001/sse
```

> **💡 各 API Key 申請說明：**
>
> | Key | 去哪裡拿 |
> |-----|---------|
> | `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) → 建立 API Key |
> | `SERPER_API_KEY` | [Serper.dev](https://serper.dev/) → 註冊後取得（有免費額度） |
> | `LINE_CHANNEL_*` | [LINE Developers](https://developers.line.biz/) → 建立 Messaging API Channel |
> | `LINE_USER_ID` | LINE Developers Console → Channel → Basic settings → Your user ID |
> | `FINMIND_API_KEY` | [FinMind](https://finmindtrade.com/) → 註冊後至個人頁面取得 Token |

### 4. 啟動服務（需開 3 個終端機）

#### Terminal 1 — 啟動 MCP Server（Docker）

```bash
docker-compose up -d
```

確認容器正在運行：
```bash
docker ps
# 應看到 casualmarket-mcp 容器在 0.0.0.0:8001->8000
```

#### Terminal 2 — 啟動 ngrok 隧道

```bash
ngrok http 8000
```

啟動後會看到類似輸出：
```
Forwarding  https://xxxx-xx-xx.ngrok-free.app -> http://localhost:8000
```

> ⚠️ **重要**：每次 ngrok 重啟都會產生新的 URL，你必須將新 URL 更新到 LINE Developers Console：
>
> **LINE Developers Console** → 你的 Channel → **Messaging API** → **Webhook URL**
>
> 填入：`https://xxxx-xx-xx.ngrok-free.app/callback`

#### Terminal 3 — 啟動主程式

```bash
conda activate ai_investor
python main.py
```

看到以下訊息表示成功啟動：
```
🌐 ngrok 公開網址: https://xxxx.ngrok-free.app/callback
🚀 Waitress 已啟動，監聽埠號: 8000
```

---

## 💬 LINE Bot 指令一覽

在 LINE 聊天室中輸入以下指令即可使用：

### 🧠 AI 深度分析（CrewAI 多 Agent 協作）

由「**策略獵人 Alpha**」整合股價 / 籌碼 / 新聞，再交由「**風控判官 Risk**」進行壓力測試，最終產出完整投資報告。

```
/分析 <股票代碼>
```
> 範例：`/分析 2330` — 深度分析台積電（⏱️ 約需 1~3 分鐘）

### 📊 即時查詢（MCP Server）

| 指令 | 說明 | 範例 |
|------|------|------|
| `/股價 <代碼>` | 即時股價、漲跌幅、成交量 | `/股價 2330` |
| `/營收 <代碼>` | 公司月營收與增減率 | `/營收 2454` |
| `/基本面 <代碼>` | 資本額、負責人、產業分類 | `/基本面 2317` |
| `/買入 <代碼> <數量>` | 模擬紙上買入 | `/買入 2330 1000` |
| `/大盤` | 近 5 分鐘台股市場交易統計 | `/大盤` |

> 💡 輸入任何非指令的文字，Bot 會自動回覆指令清單。

---

## 🗂️ 專案結構

```
AI_Investor/
├── main.py               # 主程式（Flask + LINE Bot + CrewAI + MCP）
├── requirements.txt      # Python 套件依賴清單
├── docker-compose.yml    # CasualMarket MCP Server 容器設定
├── .env                  # 環境變數（API Keys，需自行建立）
├── .env.example          # 環境變數範本
├── .gitignore            # Git 忽略規則
├── test.py               # 測試腳本
├── test_line.py          # LINE Bot 測試腳本
└── LICENSE               # 授權條款
```

---

## ❓ 常見問題 (FAQ)

### MCP Server 連線失敗？
```
無法連線到 MCP Server (http://localhost:8001/sse)
```
→ 確認 Docker Desktop 已啟動，且執行了 `docker-compose up -d`。用 `docker ps` 檢查容器狀態。

### ngrok 網址更新後 LINE Bot 沒反應？
→ 確認已在 [LINE Developers Console](https://developers.line.biz/) 更新 Webhook URL，並確保 Webhook 狀態為 **「使用中」**。

### `/分析` 指令回應很慢？
→ 這是正常的。CrewAI 需要多輪 Agent 協作（呼叫 Gemini API + Serper 搜尋 + yfinance / FinMind 資料），通常需要 1~3 分鐘。Bot 會先回覆「正在分析中...」的提示訊息。

### `pip install` 安裝失敗？
→ 確認你在正確的 conda 環境中（`conda activate ai_investor`），並使用 Python 3.10 以上版本。

---

## 🛑 停止服務

```bash
# 停止 MCP Server 容器
docker-compose down

# 停止 ngrok：在 Terminal 2 按 Ctrl+C

# 停止主程式：在 Terminal 3 按 Ctrl+C
```

---

## 📄 授權

本專案採用 [GNU GPL v3.0](LICENSE) 授權。
