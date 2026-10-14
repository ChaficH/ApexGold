//+------------------------------------------------------------------+
//|                                                   ApexBridge.mq5 |
//|                                   Copyright 2026, ApexGold Team  |
//|        Cross-Platform High-Speed Bridge for macOS (Wine) & Windows |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, ApexGold Team"
#property link      "https://github.com/apex-gold"
#property version   "2.00"
#property strict
#property description "Guaranteed macOS Wine & Windows Bridge for ApexGold Python Trading Engine"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>

enum ENUM_BRIDGE_MODE
{
   BRIDGE_WEBREQUEST = 0, // High-Speed REST WebRequest (Recommended for macOS Wine & Windows)
   BRIDGE_TCP_SOCKET = 1  // Local TCP Server Socket (Windows / Native)
};

//--- Input Parameters
input group "=== Bridge Configuration ==="
input ENUM_BRIDGE_MODE InpBridgeMode   = BRIDGE_WEBREQUEST; // Bridge Communication Mode
input string           InpServerUrl    = "http://127.0.0.1:8000/api/mt5_bridge/tick"; // REST Bridge URL
input string           InpBridgeToken  = "apex_secret_2026"; // Bridge Auth Token (matches bridge_token in config.yaml)
input string           InpBindIP       = "127.0.0.1";       // TCP Listening IP
input int              InpServerPort   = 15555;             // TCP Listening Port
input int              InpTimerMS      = 100;               // Polling Cycle (100ms)

input group "=== Execution Settings ==="
input ulong            InpMagicNumber  = 15555;             // Magic Number for ApexGold
input ulong            InpDeviation    = 20;                // Max Slippage in Points

//--- Global Handles
int             g_listen_socket = INVALID_HANDLE;
int             g_client_socket = INVALID_HANDLE;
CTrade          g_trade;
CPositionInfo   g_position;
CSymbolInfo     g_symbol_info;
string          g_rx_buffer     = "";

//+------------------------------------------------------------------+
//| Helper: String Extractor for Simple JSON Parsing                 |
//+------------------------------------------------------------------+
string JsonExtractString(const string json, const string key)
{
   string needle = "\"" + key + "\":\"";
   int start = StringFind(json, needle);
   if(start < 0) return "";
   start += StringLen(needle);
   int end = StringFind(json, "\"", start);
   if(end < 0) return "";
   return StringSubstr(json, start, end - start);
}

double JsonExtractDouble(const string json, const string key)
{
   string needle = "\"" + key + "\":";
   int start = StringFind(json, needle);
   if(start < 0) return 0.0;
   start += StringLen(needle);
   
   while(start < StringLen(json) && (StringGetCharacter(json, start) == ' ' || StringGetCharacter(json, start) == '\"'))
      start++;
      
   int end = start;
   while(end < StringLen(json))
   {
      ushort c = StringGetCharacter(json, end);
      if(c == ',' || c == '}' || c == ']' || c == '\"' || c == '\r' || c == '\n') break;
      end++;
   }
   string val_str = StringSubstr(json, start, end - start);
   return StringToDouble(val_str);
}

long JsonExtractLong(const string json, const string key)
{
   return (long)JsonExtractDouble(json, key);
}

ulong JsonExtractULong(const string json, const string key)
{
   string needle = "\"" + key + "\":";
   int start = StringFind(json, needle);
   if(start < 0) return 0;
   start += StringLen(needle);
   while(start < StringLen(json) && (StringGetCharacter(json, start) == ' ' || StringGetCharacter(json, start) == '\"'))
      start++;
   int end = start;
   while(end < StringLen(json))
   {
      ushort c = StringGetCharacter(json, end);
      if(c == ',' || c == '}' || c == ']' || c == '\"' || c == '\r' || c == '\n') break;
      end++;
   }
   string val_str = StringSubstr(json, start, end - start);
   return (ulong)StringToInteger(val_str);
}

//+------------------------------------------------------------------+
//| Execute Received Command                                         |
//+------------------------------------------------------------------+
string ExecuteCommand(const string action, const string json_line)
{
   string act = action;
   StringToUpper(act);

   // 1. ORDER_SEND
   if(act == "ORDER_SEND")
   {
      string sym = JsonExtractString(json_line, "symbol");
      long order_mode = 0;
      if(sym != "")
         SymbolInfoInteger(sym, SYMBOL_ORDER_MODE, order_mode);

      // Auto-fallback: if symbol is un-tradeable, empty, or generic XAUUSD/GOLD, trade the chart's exact symbol (_Symbol)
      if(sym == "" || order_mode == 0 || sym == "XAUUSD" || sym == "GOLD" || sym != _Symbol)
      {
         long chart_mode = 0;
         SymbolInfoInteger(_Symbol, SYMBOL_ORDER_MODE, chart_mode);
         if(chart_mode != 0)
            sym = _Symbol;
      }

      string dir     = JsonExtractString(json_line, "type");
      StringToUpper(dir);
      double vol     = JsonExtractDouble(json_line, "volume");
      double sl      = JsonExtractDouble(json_line, "sl");
      double tp      = JsonExtractDouble(json_line, "tp");
      string comment = JsonExtractString(json_line, "comment");
      if(comment == "") comment = "ApexGold";

      MqlTick tick;
      if(!SymbolInfoTick(sym, tick))
      {
         sym = _Symbol;
         SymbolInfoTick(sym, tick);
      }

      // Clamp volume to broker limits
      double v_min  = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
      double v_max  = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
      double v_step = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
      if(v_min <= 0) v_min = 0.01;
      if(v_max <= 0) v_max = 100.0;
      if(vol < v_min) vol = v_min;
      if(vol > v_max) vol = v_max;
      if(v_step > 0)  vol = MathRound(vol / v_step) * v_step;

      double price   = JsonExtractDouble(json_line, "price");
      bool res = false;
      if(StringFind(dir, "BUY_LIMIT") >= 0)
      {
         if(price <= 0.0) price = tick.ask - 0.50;
         res = g_trade.BuyLimit(vol, price, sym, sl, tp, ORDER_TIME_GTC, 0, comment);
      }
      else if(StringFind(dir, "SELL_LIMIT") >= 0)
      {
         if(price <= 0.0) price = tick.bid + 0.50;
         res = g_trade.SellLimit(vol, price, sym, sl, tp, ORDER_TIME_GTC, 0, comment);
      }
      else if(StringFind(dir, "BUY") >= 0)
         res = g_trade.Buy(vol, sym, tick.ask, sl, tp, comment);
      else if(StringFind(dir, "SELL") >= 0)
         res = g_trade.Sell(vol, sym, tick.bid, sl, tp, comment);

      if(res && g_trade.ResultRetcode() == TRADE_RETCODE_DONE)
      {
         ulong new_ticket = g_trade.ResultOrder();
         PrintFormat("ApexBridge: Order executed successfully! Ticket: %s on %s", IntegerToString(new_ticket), sym);
         return StringFormat("{\"status\":\"OK\",\"ticket\":%s,\"price\":%.5f,\"volume\":%.2f}",
                             IntegerToString(new_ticket), g_trade.ResultPrice(), g_trade.ResultVolume());
      }
      PrintFormat("ApexBridge: Order execution failed on %s! Retcode: %d, Comment: %s", sym, g_trade.ResultRetcode(), g_trade.ResultComment());
      return StringFormat("{\"status\":\"ERROR\",\"error\":\"Order failed [%d]: %s\"}",
                          g_trade.ResultRetcode(), g_trade.ResultComment());
   }

   // 2. ORDER_MODIFY
   if(act == "ORDER_MODIFY")
   {
      ulong ticket = JsonExtractULong(json_line, "ticket");
      double sl    = JsonExtractDouble(json_line, "sl");
      double tp    = JsonExtractDouble(json_line, "tp");
      double price = JsonExtractDouble(json_line, "price");

      if(ticket > 0 && PositionSelectByTicket(ticket))
      {
         if(g_trade.PositionModify(ticket, sl, tp))
         {
            PrintFormat("ApexBridge: Position #%s modified! SL: %.2f, TP: %.2f", IntegerToString(ticket), sl, tp);
            return StringFormat("{\"status\":\"OK\",\"ticket\":%s,\"sl\":%.5f,\"tp\":%.5f}", IntegerToString(ticket), sl, tp);
         }
         return StringFormat("{\"status\":\"ERROR\",\"error\":\"PositionModify failed [%d]: %s\"}",
                             g_trade.ResultRetcode(), g_trade.ResultComment());
      }
      else
      {
         // Check if this ticket is a pending limit or stop order
         if(ticket > 0 && OrderSelect(ticket))
         {
            double ord_price = (price > 0.0) ? price : OrderGetDouble(ORDER_PRICE_OPEN);
            double ord_sl    = (sl > 0.0) ? sl : OrderGetDouble(ORDER_SL);
            double ord_tp    = (tp > 0.0) ? tp : OrderGetDouble(ORDER_TP);
            if(g_trade.OrderModify(ticket, ord_price, ord_sl, ord_tp, ORDER_TIME_GTC, 0, 0))
            {
               PrintFormat("ApexBridge: Pending Order #%s modified! Price: %.2f, SL: %.2f, TP: %.2f", IntegerToString(ticket), ord_price, ord_sl, ord_tp);
               return StringFormat("{\"status\":\"OK\",\"ticket\":%s,\"price\":%.5f,\"sl\":%.5f,\"tp\":%.5f}", IntegerToString(ticket), ord_price, ord_sl, ord_tp);
            }
            return StringFormat("{\"status\":\"ERROR\",\"error\":\"OrderModify failed [%d]: %s\"}",
                                g_trade.ResultRetcode(), g_trade.ResultComment());
         }
         return "{\"status\":\"ERROR\",\"error\":\"Ticket not found in active positions or pending orders\"}";
      }
   }

   // 3. ORDER_CLOSE
   if(act == "ORDER_CLOSE")
   {
      ulong ticket = JsonExtractULong(json_line, "ticket");
      double vol   = JsonExtractDouble(json_line, "volume");

      if(ticket <= 0 || !PositionSelectByTicket(ticket))
      {
         return StringFormat("{\"status\":\"ERROR\",\"error\":\"Position #%s not found in active positions\"}", IntegerToString(ticket));
      }

      bool res = (vol > 0.0) ? g_trade.PositionClosePartial(ticket, vol) : g_trade.PositionClose(ticket);
      if(res)
         return StringFormat("{\"status\":\"OK\",\"ticket\":%s}", IntegerToString(ticket));
      return StringFormat("{\"status\":\"ERROR\",\"error\":\"Close failed [%d]: %s\"}",
                          g_trade.ResultRetcode(), g_trade.ResultComment());
   }

   // 3b. ORDER_CANCEL (Predictive Limit Cancel)
   if(act == "ORDER_CANCEL")
   {
      ulong ticket = JsonExtractULong(json_line, "ticket");
      if(ticket > 0)
      {
         bool res = g_trade.OrderDelete(ticket);
         if(res)
         {
            PrintFormat("ApexBridge: Pending Order #%s predictively cancelled successfully.", IntegerToString(ticket));
            return StringFormat("{\"status\":\"OK\",\"canceled_ticket\":%s}", IntegerToString(ticket));
         }
         return StringFormat("{\"status\":\"ERROR\",\"error\":\"Order delete failed [%d]: %s\"}",
                             g_trade.ResultRetcode(), g_trade.ResultComment());
      }
   }

   // 4. CLOSE_ALL
   if(act == "CLOSE_ALL")
   {
      string sym_filter = JsonExtractString(json_line, "symbol");
      int closed = 0;
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket <= 0) continue;
         if(!PositionSelectByTicket(ticket)) continue;
         if(sym_filter != "" && PositionGetString(POSITION_SYMBOL) != sym_filter) continue;
         if(g_trade.PositionClose(ticket)) closed++;
      }
      return StringFormat("{\"status\":\"OK\",\"closed_count\":%d}", closed);
   }

   return "{\"status\":\"OK\",\"message\":\"No action\"}";
}

//+------------------------------------------------------------------+
//| Helper: Format Timeframe Rates to Compact JSON Array             |
//+------------------------------------------------------------------+
string FormatRatesJson(ENUM_TIMEFRAMES tf, int count)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(_Symbol, tf, 0, count, rates);
   if(copied <= 0) return "[]";
   string json = "[";
   for(int i = copied - 1; i >= 0; i--)
   {
      json += StringFormat("[%s,%.5f,%.5f,%.5f,%.5f,%.1f]",
         IntegerToString(rates[i].time), rates[i].open, rates[i].high, rates[i].low, rates[i].close, (double)rates[i].tick_volume);
      if(i > 0) json += ",";
   }
   json += "]";
   return json;
}

//+------------------------------------------------------------------+
//| Helper: Format Real Closed Deals History from MT5                |
//+------------------------------------------------------------------+
string FormatClosedHistoryJson(int max_deals = 30)
{
   if(!HistorySelect(TimeCurrent() - 86400, TimeCurrent() + 60))
      return "[]";

   int total = HistoryDealsTotal();
   if(total <= 0) return "[]";

   string json = "[";
   int count = 0;
   for(int i = total - 1; i >= 0 && count < max_deals; i--)
   {
      ulong d_ticket = HistoryDealGetTicket(i);
      if(d_ticket == 0) continue;

      ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(d_ticket, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_INOUT) continue;

      ulong pos_id     = (ulong)HistoryDealGetInteger(d_ticket, DEAL_POSITION_ID);
      string sym       = HistoryDealGetString(d_ticket, DEAL_SYMBOL);
      long deal_type   = HistoryDealGetInteger(d_ticket, DEAL_TYPE);
      string pos_dir   = (deal_type == DEAL_TYPE_SELL) ? "BUY" : "SELL";
      double vol       = HistoryDealGetDouble(d_ticket, DEAL_VOLUME);
      double close_p   = HistoryDealGetDouble(d_ticket, DEAL_PRICE);
      double profit    = HistoryDealGetDouble(d_ticket, DEAL_PROFIT);
      double swap      = HistoryDealGetDouble(d_ticket, DEAL_SWAP);
      double comm      = HistoryDealGetDouble(d_ticket, DEAL_COMMISSION);
      long   close_t   = HistoryDealGetInteger(d_ticket, DEAL_TIME);
      string comment   = HistoryDealGetString(d_ticket, DEAL_COMMENT);

      double open_p = close_p;
      long open_t   = close_t;
      if(HistorySelectByPosition(pos_id))
      {
         int p_deals = HistoryDealsTotal();
         for(int j = 0; j < p_deals; j++)
         {
            ulong in_d = HistoryDealGetTicket(j);
            if(in_d > 0 && (ENUM_DEAL_ENTRY)HistoryDealGetInteger(in_d, DEAL_ENTRY) == DEAL_ENTRY_IN)
            {
               open_p = HistoryDealGetDouble(in_d, DEAL_PRICE);
               open_t = HistoryDealGetInteger(in_d, DEAL_TIME);
               long in_type = HistoryDealGetInteger(in_d, DEAL_TYPE);
               pos_dir = (in_type == DEAL_TYPE_BUY) ? "BUY" : "SELL";
               break;
            }
         }
      }

      if(count > 0) json += ",";
      json += StringFormat(
         "{\"ticket\":%s,\"pos_id\":%s,\"symbol\":\"%s\",\"direction\":\"%s\",\"volume\":%.2f,\"open_price\":%.5f,\"close_price\":%.5f,\"open_time\":%s,\"close_time\":%s,\"profit\":%.2f,\"swap\":%.2f,\"commission\":%.2f,\"comment\":\"%s\"}",
         IntegerToString(d_ticket), IntegerToString(pos_id), sym, pos_dir, vol, open_p, close_p,
         IntegerToString(open_t), IntegerToString(close_t), profit, swap, comm, comment
      );
      count++;
   }
   json += "]";
   return json;
}

string GetChartTimeframeString()
{
   ENUM_TIMEFRAMES tf = Period();
   if(tf == PERIOD_M1) return "1m";
   if(tf == PERIOD_M5) return "5m";
   if(tf == PERIOD_M15) return "15m";
   if(tf == PERIOD_M30) return "30m";
   if(tf == PERIOD_H1) return "1H";
   if(tf == PERIOD_H4) return "4H";
   if(tf == PERIOD_D1) return "1D";
   return "1m";
}

//+------------------------------------------------------------------+
//| High-Speed REST WebRequest Loop (Guaranteed macOS Wine Support)  |
//+------------------------------------------------------------------+
void ServiceWebRequestPump()
{
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   double balance     = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity      = AccountInfoDouble(ACCOUNT_EQUITY);
   double margin      = AccountInfoDouble(ACCOUNT_MARGIN);
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   long   leverage    = AccountInfoInteger(ACCOUNT_LEVERAGE);

   // Build open positions array
   string pos_json = "[";
   int total_pos = PositionsTotal();
   int added = 0;
   for(int i = 0; i < total_pos; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      string p_sym = PositionGetString(POSITION_SYMBOL);
      long p_type_int = PositionGetInteger(POSITION_TYPE);
      string p_type = (p_type_int == POSITION_TYPE_BUY) ? "BUY" : "SELL";
      double p_vol  = PositionGetDouble(POSITION_VOLUME);
      double p_open = PositionGetDouble(POSITION_PRICE_OPEN);
      double p_curr = PositionGetDouble(POSITION_PRICE_CURRENT);
      double p_sl   = PositionGetDouble(POSITION_SL);
      double p_tp   = PositionGetDouble(POSITION_TP);
      double p_pnl  = PositionGetDouble(POSITION_PROFIT);
      string p_comm = PositionGetString(POSITION_COMMENT);
      long   p_magic = PositionGetInteger(POSITION_MAGIC);
      long   p_time  = PositionGetInteger(POSITION_TIME);

      if(added > 0) pos_json += ",";
      pos_json += StringFormat("{\"ticket\":%s,\"symbol\":\"%s\",\"type\":\"%s\",\"volume\":%.2f,\"open_price\":%.5f,\"current_price\":%.5f,\"sl\":%.5f,\"tp\":%.5f,\"profit\":%.2f,\"comment\":\"%s\",\"magic\":%s,\"open_time\":%s}",
                               IntegerToString(ticket), p_sym, p_type, p_vol, p_open, p_curr, p_sl, p_tp, p_pnl, p_comm, IntegerToString(p_magic), IntegerToString(p_time));
      added++;
   }
   pos_json += "]";

   // Build active pending limit/stop orders array
   string orders_json = "[";
   int total_orders = OrdersTotal();
   int o_added = 0;
   for(int i = 0; i < total_orders; i++)
   {
      ulong o_ticket = OrderGetTicket(i);
      if(o_ticket <= 0) continue;
      string o_sym = OrderGetString(ORDER_SYMBOL);
      long o_type = OrderGetInteger(ORDER_TYPE);
      string o_dir = "PENDING";
      if(o_type == ORDER_TYPE_BUY_LIMIT) o_dir = "BUY_LIMIT";
      else if(o_type == ORDER_TYPE_SELL_LIMIT) o_dir = "SELL_LIMIT";
      else if(o_type == ORDER_TYPE_BUY_STOP) o_dir = "BUY_STOP";
      else if(o_type == ORDER_TYPE_SELL_STOP) o_dir = "SELL_STOP";

      double o_vol = OrderGetDouble(ORDER_VOLUME_CURRENT);
      double o_p   = OrderGetDouble(ORDER_PRICE_OPEN);
      double o_sl  = OrderGetDouble(ORDER_SL);
      double o_tp  = OrderGetDouble(ORDER_TP);
      string o_comm = OrderGetString(ORDER_COMMENT);
      long   o_magic = OrderGetInteger(ORDER_MAGIC);

      if(o_added > 0) orders_json += ",";
      orders_json += StringFormat("{\"ticket\":%s,\"symbol\":\"%s\",\"type\":\"%s\",\"volume\":%.2f,\"price\":%.5f,\"sl\":%.5f,\"tp\":%.5f,\"comment\":\"%s\",\"magic\":%s}",
                                  IntegerToString(o_ticket), o_sym, o_dir, o_vol, o_p, o_sl, o_tp, o_comm, IntegerToString(o_magic));
      o_added++;
   }
   orders_json += "]";

   // Capture live forming bar
   MqlRates cur_rates[];
   ArraySetAsSeries(cur_rates, true);
   string cur_bar_json = "{}";
   if(CopyRates(_Symbol, PERIOD_M1, 0, 1, cur_rates) > 0)
   {
      cur_bar_json = StringFormat(
         "{\"time\":%s,\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%.1f}",
         IntegerToString(cur_rates[0].time), cur_rates[0].open, cur_rates[0].high, cur_rates[0].low, cur_rates[0].close, (double)cur_rates[0].tick_volume
      );
   }

   // Capture 1:1 broker candles & closed deals from MT5 every 2 seconds
   static ulong last_bars_sync = 0;
   ulong now_ms = GetTickCount64();
   static string bars_m1_json       = "[]";
   static string bars_m5_json       = "[]";
   static string bars_m15_json      = "[]";
   static string bars_h1_json       = "[]";
   static string closed_deals_json  = "[]";
   if(now_ms - last_bars_sync >= 2000)
   {
      last_bars_sync = now_ms;
      bars_m1_json      = FormatRatesJson(PERIOD_M1, 500);
      bars_m5_json      = FormatRatesJson(PERIOD_M5, 300);
      bars_m15_json     = FormatRatesJson(PERIOD_M15, 300);
      bars_h1_json      = FormatRatesJson(PERIOD_H1, 300);
      closed_deals_json = FormatClosedHistoryJson(30);
   }

   // Check if Algo Trading is allowed in MT5
   bool term_trade = (bool)TerminalInfoInteger(TERMINAL_TRADE_ALLOWED);
   bool mql_trade  = (bool)MQLInfoInteger(MQL_TRADE_ALLOWED);
   bool can_auto_trade = term_trade && mql_trade;

   static ulong last_warn_ms = 0;
   if(!can_auto_trade && (now_ms - last_warn_ms > 15000))
   {
      last_warn_ms = now_ms;
      Print("ApexBridge ATTENTION: Algo Trading is DISABLED in MetaTrader 5! Click 'Algo Trading' in top toolbar to turn it GREEN.");
   }

   // Construct JSON payload to POST to Python
   string cur_tf_str = GetChartTimeframeString();
   string token_field = (InpBridgeToken != "") ? StringFormat(",\"token\":\"%s\"", InpBridgeToken) : "";
   string payload = StringFormat(
      "{\"symbol\":\"%s\",\"chart_tf\":\"%s\",\"bid\":%.5f,\"ask\":%.5f,\"spread\":%.2f,\"time\":%s,\"balance\":%.2f,\"equity\":%.2f,\"margin\":%.2f,\"free_margin\":%.2f,\"leverage\":%d,\"algo_trading_allowed\":%s,\"positions\":%s,\"orders\":%s,\"closed_deals\":%s,\"cur_bar\":%s,\"rates_m1\":%s,\"rates_m5\":%s,\"rates_m15\":%s,\"rates_h1\":%s,\"rates\":%s%s}",
      _Symbol, cur_tf_str, tick.bid, tick.ask, (tick.ask - tick.bid), IntegerToString(tick.time), balance, equity, margin, free_margin, leverage,
      (can_auto_trade ? "true" : "false"), pos_json, orders_json, closed_deals_json, cur_bar_json, bars_m1_json, bars_m5_json, bars_m15_json, bars_h1_json, bars_m1_json, token_field
   );

   char post_data[];
   char result_data[];
   string result_headers;
   string headers = "Content-Type: application/json\r\n";
   if(InpBridgeToken != "")
   {
      headers += "X-Bridge-Token: " + InpBridgeToken + "\r\n";
   }

   StringToCharArray(payload, post_data, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayResize(post_data, ArraySize(post_data) - 1);

   // Perform fast WebRequest
   ResetLastError();
   int res = WebRequest("POST", InpServerUrl, headers, 150, post_data, result_data, result_headers);

   if(res == 200 && ArraySize(result_data) > 0)
   {
      string response_str = CharArrayToString(result_data, 0, WHOLE_ARRAY, CP_UTF8);
      string action = JsonExtractString(response_str, "action");
      if(action != "" && action != "NONE")
      {
         PrintFormat("ApexBridge: Executing command from Python: %s", action);
         ExecuteCommand(action, response_str);
      }
   }
   else if(res == 401)
   {
      static ulong last_401_ms = 0;
      if(now_ms - last_401_ms > 10000)
      {
         last_401_ms = now_ms;
         Print("ApexBridge ERROR [401]: Unauthorized! Bridge token does not match python server bridge_token in config.yaml.");
      }
   }
   else if(res == -1)
   {
      int err = GetLastError();
      // Error 4014: URL not allowed in WebRequest list
      if(err == 4014)
      {
         Print("ApexBridge ERROR [4014]: Add http://127.0.0.1:8000 to MT5 Options -> Expert Advisors -> Allow WebRequest!");
      }
   }
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(InpDeviation);

   // Auto-detect broker supported filling mode
   uint filling = (uint)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_IOC) != 0)
      g_trade.SetTypeFilling(ORDER_FILLING_IOC);
   else if((filling & SYMBOL_FILLING_FOK) != 0)
      g_trade.SetTypeFilling(ORDER_FILLING_FOK);
   else
      g_trade.SetTypeFilling(ORDER_FILLING_RETURN);

   EventSetMillisecondTimer(InpTimerMS);

   Print("=========================================================");
   PrintFormat(" ApexBridge v2.1 Started on %s", _Symbol);
   if(InpBridgeMode == BRIDGE_WEBREQUEST)
      PrintFormat(" Mode: REST WebRequest (Target: %s)", InpServerUrl);
   else
      PrintFormat(" Mode: TCP Socket Server (%s:%d)", InpBindIP, InpServerPort);
   Print(" Ready for zero-cost sub-millisecond execution & 1:1 candle streaming.");
   Print("=========================================================");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("ApexBridge EA stopped.");
}

//+------------------------------------------------------------------+
//| Expert timer function (100ms cycle)                              |
//+------------------------------------------------------------------+
void OnTimer()
{
   if(InpBridgeMode == BRIDGE_WEBREQUEST)
      ServiceWebRequestPump();
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   if(InpBridgeMode == BRIDGE_WEBREQUEST)
      ServiceWebRequestPump();
}
