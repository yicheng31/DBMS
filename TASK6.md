# Task 6 Extension — Agent & UI Optimization

## Overview

This extension enhances the TransitFlow AI agent (`skeleton/agent.py`) and web interface (`skeleton/ui.py`) with Chinese language support, stronger tool routing, new database tools, and improved user experience. All changes touch database query operations and are documented with inline comments.

---

## Modified Files

### 1. `skeleton/agent.py`

**`# TASK 6 EXTENSION` comment:** Lines 1–3

#### New Database Tools Added

| Tool Name | Database Function Called | Description |
|-----------|------------------------|-------------|
| `get_user_profile` | `query_user_profile()` from `databases/relational/queries.py` | Retrieves logged-in user's profile (name, email, registration date) from PostgreSQL |
| `get_payment_info` | `query_payment_info()` from `databases/relational/queries.py` | Retrieves payment details for a specific booking from PostgreSQL |

These two functions already existed in `databases/relational/queries.py` but were never registered as agent tools. The extension wires them into the agent's tool routing so the LLM can call them when users ask about their account or payment history.

#### New Import Added (Line 49)

```python
from databases.relational.queries import (
    ...
    query_user_profile,    # NEW — used by get_user_profile tool
    query_payment_info,    # NEW — used by get_payment_info tool
    ...
)
```

#### Tool Definitions Added to `TOOLS` List

- `get_user_profile` tool definition (with description, parameters, required fields)
- `get_payment_info` tool definition (with `booking_id` parameter)

#### Tool Execution Added to `_execute_tool()`

```python
elif tool_name == "get_user_profile":
    # Calls query_user_profile() — reads from PostgreSQL registered_users table
    ...

elif tool_name == "get_payment_info":
    # Calls query_payment_info() — reads from PostgreSQL payments table
    ...
```

#### Chinese Station Name Support

Added 30 Chinese station name mappings to `_STATION_INDEX`:

| Chinese Name | Station ID | Network |
|-------------|-----------|---------|
| 中央廣場 | MS01 | Metro |
| 河濱站 | MS02 | Metro |
| 北門站 | MS03 | Metro |
| 榆樹公園站 | MS04 | Metro |
| 西田站 | MS05 | Metro |
| 海港景站 | MS06 | Metro |
| 舊城站 | MS07 | Metro |
| 大學站 | MS08 | Metro |
| 皇后橋站 | MS09 | Metro |
| 公園側站 | MS10 | Metro |
| 綠丘站 | MS11 | Metro |
| 湖岸站 | MS12 | Metro |
| 克利夫頓站 | MS13 | Metro |
| 東威克站 | MS14 | Metro |
| 芬戴爾站 | MS15 | Metro |
| 山頂站 | MS16 | Metro |
| 寬地站 | MS17 | Metro |
| 陽光谷站 | MS18 | Metro |
| 紅木站 | MS19 | Metro |
| 桑頓站 | MS20 | Metro |
| 中央站 | NR01 | National Rail |
| 楓木站 | NR02 | National Rail |
| 舊城交匯站 | NR03 | National Rail |
| 阿什福德站 | NR04 | National Rail |
| 石港站 | NR05 | National Rail |
| 橋港站 | NR06 | National Rail |
| 芬戴爾停靠站 | NR07 | National Rail |
| 煤港站 | NR08 | National Rail |
| 丹摩站 | NR09 | National Rail |
| 蘭福德終點站 | NR10 | National Rail |

#### Chinese Policy Query Translation

New function `_translate_policy_query()` and dictionary `_POLICY_TRANSLATION` that translates Chinese policy keywords to English before embedding search. This solves the problem where Chinese queries could not match English policy documents in the pgvector database.

| Chinese Keyword | English Translation |
|----------------|-------------------|
| 退款 | refund cancellation policy |
| 退票 | refund cancellation policy |
| 取消 | cancellation refund policy |
| 補償 | delay compensation policy |
| 延誤 | delay compensation policy |
| 誤點 | delay compensation policy |
| 行李 | luggage baggage policy |
| 寵物 | pet animal travel policy |
| 腳踏車 | bicycle bike travel policy |
| 兒童 | child fare discount policy |
| 票種 | ticket types single return day pass |
| 政策 | company policy rules |
| 食物 | food drink policy onboard |
| 逃票 | fare evasion penalty |
| 訂票規則 | booking rules policy |

Implementation in `_execute_tool()` for `search_policy`:
```python
# Translate Chinese query to English for better vector similarity matching
raw_query = params["query"]
search_query = _translate_policy_query(raw_query)
embedding = llm.embed(search_query)
docs = query_policy_vector_search(embedding)
# If translated query found nothing, try original as fallback
if not docs and search_query != raw_query:
    embedding = llm.embed(raw_query)
    docs = query_policy_vector_search(embedding)
```

#### Greeting Protection

New function `_is_greeting()` and set `_GREETING_PATTERNS` that detects simple greetings (你好, hello, hi, etc.) and skips all tool calls. This prevents the small LLM (llama3.2:1b) from misrouting greetings to random database tools.

#### Stronger Fallback Logic

The original fallback system only triggered when the LLM selected **no tools**. The extension strengthens it to also trigger when the LLM selects the **wrong tool**.

**Before (original):**
```python
# Only fires when NO tool was selected
elif not tool_calls and _two_stations:
    ...
```

**After (extension):**
```python
# Also fires when WRONG tool was selected
if (not _is_route and _two_stations and any(kw in _lower for kw in _avail_triggers)):
    _expected_tool = "check_national_rail_availability" if o.startswith("NR") else "check_metro_availability"
    if not _tool_selected(_expected_tool, "origin_id", "destination_id"):
        _fallback(_expected_tool, _params, "availability query (override wrong tool)")
```

Same pattern applied to policy queries:
```python
# Policy fallback now overrides wrong tool selections
if any(kw in _lower for kw in _policy_triggers):
    if not _tool_selected("search_policy", "query"):
        _fallback("search_policy", {"query": user_message}, "policy query (override wrong tool)")
```

#### Regex Fix for Chinese Text

Changed station ID extraction regex to work with Chinese characters:

**Before:** `r'\b(MS\d{2}|NR\d{2})\b'` — `\b` word boundary fails when Chinese characters are adjacent to station IDs (e.g., "MS01到MS09")

**After:** `r'(MS\d{2}|NR\d{2})'` — removed `\b` so station IDs are detected regardless of surrounding characters

#### Booking Confirmation Gate

New function `_user_confirmed()` that checks if the user has explicitly confirmed a booking before allowing `make_booking` to execute. Confirmation words include: confirm, yes, 確認, 好, ok, 好的, 沒問題, 訂吧, 訂了.

#### Human-Friendly Error Messages

All error messages changed from English technical messages to friendly Chinese:

| Original | New |
|---------|-----|
| `"No user is currently logged in."` | `"您尚未登入。請點右上角的登入按鈕後再試 😊"` |
| `"User profile not found."` | `"找不到使用者資料，請重新登入。"` |
| `"No metro service found between these stations."` | `"很抱歉，找不到這兩站之間的捷運服務。請確認站點代碼是否正確。"` |
| `{"error": data}` | `{"error": f"訂票失敗：{data}。請稍後再試或聯絡客服。"}` |

---

### 2. `skeleton/ui.py`

#### Welcome Message

Added `WELCOME_MESSAGE` constant displayed when the chat interface first loads:
```
👋 歡迎使用 TransitFlow 智慧交通助理！
🚂 查詢國鐵班次和票價
🚇 查詢捷運路線和票價
🗺️ 規劃最快或最便宜的路線
🎫 訂票和取消訂票（需登入）
📋 查詢退款、行李等相關政策
```

#### Quick-Select Station Buttons

Added `METRO_STATIONS` and `RAIL_STATIONS` lists with commonly used stations as clickable buttons in the sidebar. Clicking a button auto-fills the chat input with the station name and ID.

Metro stations: 中央廣場 MS01, 河濱站 MS02, 北門站 MS03, 大學站 MS08, 皇后橋站 MS09, 東威克站 MS14

National Rail stations: 中央站 NR01, 楓木站 NR02, 舊城交匯站 NR03, 石港站 NR05, 橋港站 NR06, 丹摩站 NR09

#### Login Panel Auto-Close

Login and registration panels now automatically close after successful authentication, providing a cleaner interface.

#### Full Chinese Localization

- Title: `🚂 TransitFlow 智慧交通助理`
- Buttons: 登入, 註冊, 登出, 送出, 清除對話
- Labels: 對話模型, 密碼, 出生年份, 安全問題
- Error messages: all in Chinese
- Example queries: all in Chinese

---

## Testing Evidence

| Test Query | Expected Tool | Result |
|-----------|--------------|--------|
| `NR01到NR05有哪些班次？` | `check_national_rail_availability` | ✅ Correct (via fallback override) |
| `MS01到MS09有哪些捷運？` | `check_metro_availability` | ✅ Correct (via fallback override) |
| `What trains run from NR01 to NR05?` | `check_national_rail_availability` | ✅ Correct (native + fallback) |
| `從MS01到MS14最快怎麼走？` | `find_route` | ✅ Correct (native selection) |
| `退款政策是什麼？` | `search_policy` | ✅ Correct (via fallback + Chinese translation) |
| `What is the refund policy?` | `search_policy` | ✅ Correct (native selection) |
| `你好` | No tool | ✅ Correct (greeting protection) |

---

## Summary of Changes

| Category | Count |
|---------|-------|
| New database tools | 2 (`get_user_profile`, `get_payment_info`) |
| New helper functions | 4 (`_is_greeting`, `_user_confirmed`, `_translate_policy_query`, greeting patterns) |
| Chinese station mappings | 30 |
| Chinese policy translations | 15 |
| Fallback rules enhanced | 3 (availability, policy, route) |
| Files modified | 2 (`skeleton/agent.py`, `skeleton/ui.py`) |
