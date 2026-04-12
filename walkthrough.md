# CasualMarket MCP 整合完成指引

我們已經成功將 LineBot 的訊息處理邏輯進行升級，現在它不僅能調用 CrewAI 進行深度分析，還能透過特定的指令將請求直接送往 CasualMarket MCP Server 取回單一工具結果！

---

## 一、LineBot 給使用者的最新預設提示訊息

當使用者在 Line 中輸入了系統無法辨識的指令，或是直接點擊/輸入了沒有參數的字詞時，系統會自動預設回傳以下選單提示：

> 🤖 AI Investor 小幫手指令：
> 
> 【Agent 深度分析】
> 🔹 `/分析 <代碼>` (例: `/分析 2330`)
> 
> 【MCP 單一功能查詢】
> 🔹 `/股價 <代碼>` (例: `/股價 2330`)
> 🔹 `/營收 <代碼>` (例: `/營收 2330`)
> 🔹 `/買入 <代碼> <數量>` (例: `/買入 2330 1000`)
> 🔹 `/基本面 <代碼>` (例: `/基本面 2330`)
> 🔹 `/大盤` (查詢即時交易統計)

*註：你可以直接將上述文字做為 Line 官方帳號的「歡迎訊息（Greeting Message）」或是圖文選單按鈕的對應文字，方便使用者一目了然。*

---

## 二、如何正確開啟與測試系統？(指引步驟)

因為目前的架構拆分成 **Python Flask (LineBot)** 與 **CasualMarket (MCP Server)** 兩個端點，請配合以下步驟啟動整個服務：

### 步驟 1：修改環境變數 ( `.env` )
在你的 `AI_Investor` 專案目錄下的 `.env` 檔案中加入這行：
```env
MCP_SERVER_URL=http://localhost:8001/sse
```
這代表我們的 LineBot 去哪裡尋找 CasualMarket。

### 步驟 2：啟動 CasualMarket MCP Server (在 8001 port)
由於我們原本的 `main.py` 佔用了 `8000` 埠，你需要另開一個終端機，將 CasualMarket 跑在 `8001`。如果你手邊已經有 CasualMarket 的 Docker 環境，可以使用這個指令啟動：
```bash
docker run -p 8001:8000 -d sacahan/casualmarket-mcp:latest
# 或依照 CasualMarket 官方方式將內部 8000 對應到本機 8001
```

### 步驟 3：啟動 ngrok
開啟一個新的系統終端機視窗，準備將我們 LineBot 的 8000 port 對外暴露：
```bash
ngrok http 8000
```
把 ngrok 產生的 URL（例如 `https://abc-123.ngrok.app`）複製起來，去 LINE Developer Console 更新 Webhook URL 為 `https://abc-123.ngrok.app/callback` 並開啟 Use webhook。

### 步驟 4：啟動 AI_Investor (LineBot 伺服器)
打開一個系統終端機視窗，進入 `AI_Investor` 專案資料夾並執行：
```bash
# 確保你已經進入 anaconda 等 Python 虛擬環境
python main.py
```

### 步驟 5：打開 Line 開始測試
現在整個系統已經啟動！你可以直接到你的 LineBot 聊天視窗輸入：
1. `/股價 2330` -> 確認 MCP 的回傳速度。
2. `/大盤` -> 確認 MCP 是否正常連線並抓資料。
3. `/分析 2330` -> 確認原有的 CrewAI Agent 仍然能夠正常進行多階段分析！
