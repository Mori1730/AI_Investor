# AI Investor

主要練習並整合以下技術與生態系統：
- **Gemini API** (Google AI Studio)
- **Serper API** (Google 搜尋整合)
- **LINE Messaging API (LineBot)** (前端聊天機器人介面)
- **Model Context Protocol (MCP)** (官方 Python SDK，異步工具呼叫)
- **CasualMarket MCP Server** (台股專用 MCP 工具包)
- **Docker Compose** (輕量化容器部署)
- **Flask** (本機伺服器)
- **ngrok** (本機埠號外網透傳)

---

## 環境與前置需求 (Environment)

1. **Python 環境**: Anaconda environment (`ai_investor` 環境)
2. **Docker Desktop**: 必須安裝並執行於背景，用於啟動 MCP Server。
3. **環境變數設定檔 `.env`**: 請確保專案目錄下有正確設定好的 `.env` 檔案。

---

## 啟動與使用步驟 (Usage)

專案啟動分為三個階段，請分別開三個不同的**終端機 (Terminal)** 視窗執行：

### 1. 啟動 CasualMarket MCP 伺服器 (Docker)
必須先啟動台股 MCP Server 容器，主程式才不會連線逾時。它會佔用本機的 `8001` Port。
```bash
docker-compose up -d
```
*(停止伺服器時，請使用 `docker-compose down`)*

### 2. 運行 ngrok 進行外部網址穿透
開啟 port 8000 提供給 Flask 與 LineBot 接收 webhook 請求：
```bash
ngrok http 8000
```
> **注意：** 記得將 ngrok 產生的 全新 HTTPS 網址（例如 `https://xxxx.ngrok-free.dev/callback`）更新至 LINE Developer Console 的 Webhook URL 欄位。

### 3. 運行主 Flask 程式
確保你位於 `ai_investor` conda 環境下，啟動 Flask 伺服器：
```bash
conda activate ai_investor
python main.py
```

---

## LINE 小幫手可用指令清單 (Commands)

直接在 LINE 聊天室中輸入以下指令即可觸發對應功能：

### 🧠 Agent 深度投資分析 (CrewAI)
使用複合式的 Agent（策略獵人 Alpha + 風控判官 Risk）進行全面性的資料整合與新聞情緒評估。
- **`/分析 <代碼>`**
  *範例:* `/分析 2330`

### 📊 即時 MCP 單一功能查詢 (CasualMarket)
直接調用本地端的 MCP Server 工具，返回的複雜 JSON 數據已在系統內接上智能中文化標籤以提升閱讀體驗。
- **`/股價 <代碼>`** - 即時股價數據與走勢
  *範例:* `/股價 2330`
- **`/營收 <代碼>`** - 查詢公司單月營收與月增年增率
  *範例:* `/營收 2330`
- **`/基本面 <代碼>`** - 查詢公司資本額、負責人與核心產業歸屬
  *範例:* `/基本面 2330`
- **`/買入 <代碼> <數量>`** - 進行模擬紙上買入測試
  *範例:* `/買入 2330 1000`
- **`/大盤`** - 獲取近五分鐘的整體台股市場交易統計 (委買/委賣/成交)
