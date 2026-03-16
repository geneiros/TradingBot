using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static class Config
{
    public const int    OhlcHigh       = 2;
    public const int    OhlcLow        = 3;
    public const int    OhlcClose      = 4;
    public const double ProfitTarget   = 1.05;  // take profit at +5%
    public const double StopLoss       = 0.99;  // stop-loss at -1%
    public const double TrailingStop   = 0.99;  // trailing stop: -1% from peak MA
    public const int    WindowSeconds  = 43200; // 12-hour rolling window
    public const int    MaxMaHistory   = 200;   // cap for MA lists

    public static readonly string[] Pairs =
        ["XETHZEUR", "XXBTZEUR", "MANAEUR", "GRTEUR", "LSKEUR", "SCEUR"];
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------
class Trade
{
    [JsonPropertyName("time_stamp")] public string TimeStamp { get; set; } = "";
    [JsonPropertyName("price_usd")]  public double PriceUsd  { get; set; }
    [JsonPropertyName("bought")]     public bool   Bought    { get; set; }
    [JsonPropertyName("sold")]       public bool   Sold      { get; set; }
    [JsonPropertyName("amount")]     public double Amount    { get; set; }
}

class PairData
{
    [JsonPropertyName("high")]   public List<double>       High   { get; set; } = [];
    [JsonPropertyName("low")]    public List<double>       Low    { get; set; } = [];
    [JsonPropertyName("close")]  public List<double>       Close  { get; set; } = [];
    [JsonPropertyName("prices")] public List<List<string>> Prices { get; set; } = [];
}

// ---------------------------------------------------------------------------
// Kraken REST client  (replaces krakenex)
// ---------------------------------------------------------------------------
class KrakenClient
{
    private readonly HttpClient _http = new() { BaseAddress = new Uri("https://api.kraken.com") };
    private string _apiKey    = "";
    private byte[] _apiSecret = [];

    public void LoadKey(string path)
    {
        var lines  = File.ReadAllLines(path);
        _apiKey    = lines[0].Trim();
        _apiSecret = Convert.FromBase64String(lines[1].Trim());
    }

    public async Task<JsonObject> QueryPublicAsync(string method, Dictionary<string, string>? data = null)
    {
        var qs  = data is { Count: > 0 }
            ? "?" + string.Join("&", data.Select(kv => $"{kv.Key}={Uri.EscapeDataString(kv.Value)}"))
            : "";
        var body = await _http.GetStringAsync($"/0/public/{method}{qs}");
        return JsonNode.Parse(body)!.AsObject();
    }

    public async Task<JsonObject> QueryPrivateAsync(string method, Dictionary<string, string>? data = null)
    {
        data ??= [];
        var nonce    = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString();
        data["nonce"] = nonce;

        var postData = string.Join("&", data.Select(kv => $"{kv.Key}={Uri.EscapeDataString(kv.Value)}"));
        var path     = $"/0/private/{method}";

        // Signature: HMAC-SHA512( path_bytes + SHA256(nonce + postData), secret )
        var sha256   = SHA256.HashData(Encoding.UTF8.GetBytes(nonce + postData));
        var pathBytes = Encoding.UTF8.GetBytes(path);
        var hmacInput = new byte[pathBytes.Length + sha256.Length];
        pathBytes.CopyTo(hmacInput, 0);
        sha256.CopyTo(hmacInput, pathBytes.Length);
        var signature = Convert.ToBase64String(new HMACSHA512(_apiSecret).ComputeHash(hmacInput));

        var req = new HttpRequestMessage(HttpMethod.Post, path)
        {
            Content = new StringContent(postData, Encoding.UTF8, "application/x-www-form-urlencoded")
        };
        req.Headers.Add("API-Key",  _apiKey);
        req.Headers.Add("API-Sign", signature);

        var resp = await _http.SendAsync(req);
        return JsonNode.Parse(await resp.Content.ReadAsStringAsync())!.AsObject();
    }
}

// ---------------------------------------------------------------------------
// Bot
// ---------------------------------------------------------------------------
class Bot(KrakenClient kraken)
{
    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = true };

    // ---- Balance ----------------------------------------------------------

    public async Task<Dictionary<string, string>> GetBalanceAsync()
    {
        try
        {
            var resp   = await kraken.QueryPrivateAsync("Balance");
            var errors = resp["error"]?.AsArray();
            if (errors is { Count: > 0 })
            {
                Console.WriteLine($"Balance API error: {errors}");
                return [];
            }
            return resp["result"]!.AsObject()
                .ToDictionary(kv => kv.Key, kv => kv.Value?.ToString() ?? "");
        }
        catch (Exception e)
        {
            Console.WriteLine($"Balance exception: {e.Message}");
            return [];
        }
    }

    // ---- OHLC data --------------------------------------------------------

    public async Task<List<List<string>>> GetCryptoDataAsync(string pair, string since)
    {
        var resp   = await kraken.QueryPublicAsync("OHLC", new() { ["pair"] = pair, ["since"] = since });
        var errors = resp["error"]?.AsArray();
        if (errors is { Count: > 0 })
            throw new Exception($"OHLC error for {pair}: {errors}");

        return resp["result"]![pair]!.AsArray()
            .Select(c => c!.AsArray().Select(v => v!.ToString()).ToList())
            .ToList();
    }

    // ---- Trades file ------------------------------------------------------

    public Dictionary<string, List<Trade>> LoadTrades()
    {
        try
        {
            var json = File.ReadAllText("trades.json");
            return JsonSerializer.Deserialize<Dictionary<string, List<Trade>>>(json, JsonOpts)
                   ?? EmptyTrades();
        }
        catch { return EmptyTrades(); }
    }

    private static Dictionary<string, List<Trade>> EmptyTrades() =>
        Config.Pairs.ToDictionary(p => p, _ => new List<Trade>());

    public void SaveTrade(double close, string name, bool bought, bool sold, double amount)
    {
        var trade = new Trade
        {
            TimeStamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString(),
            PriceUsd  = close,
            Bought    = bought,
            Sold      = sold,
            Amount    = amount
        };
        Console.WriteLine("TRADE:");
        Console.WriteLine(JsonSerializer.Serialize(trade, JsonOpts));
        var trades = LoadTrades();
        trades[name].Add(trade);
        File.WriteAllText("trades.json", JsonSerializer.Serialize(trades, JsonOpts));
    }

    // ---- Market data file -------------------------------------------------

    public Dictionary<string, PairData> LoadCryptoDataFromFile()
    {
        try
        {
            var json = File.ReadAllText("data.json");
            return JsonSerializer.Deserialize<Dictionary<string, PairData>>(json, JsonOpts)
                   ?? MakeCryptoData();
        }
        catch
        {
            var data = MakeCryptoData();
            SaveCryptoData(data);
            return data;
        }
    }

    public static Dictionary<string, PairData> MakeCryptoData() =>
        Config.Pairs.ToDictionary(p => p, _ => new PairData());

    public void SaveCryptoData(Dictionary<string, PairData> data) =>
        File.WriteAllText("data.json", JsonSerializer.Serialize(data, JsonOpts));

    // ---- Available funds --------------------------------------------------

    public async Task<double> GetAvailableFundsAsync()
    {
        var balance = await GetBalanceAsync();
        balance.TryGetValue("ZEUR", out var zusdStr);
        double money = double.TryParse(zusdStr, out var m) ? m : 0.0;

        var trades    = LoadTrades();
        int pairsHeld = Config.Pairs.Count(p =>
            trades[p].Count > 0 && trades[p][^1].Bought && !trades[p][^1].Sold);
        int slotsFree = Config.Pairs.Length - pairsHeld;

        return slotsFree <= 0 ? 0.0 : money / slotsFree;
    }

    // ---- Order placement --------------------------------------------------

    public async Task<bool> PlaceOrderAsync(string pair, string orderType, double volume)
    {
        var resp   = await kraken.QueryPrivateAsync("AddOrder", new()
        {
            ["pair"]      = pair,
            ["type"]      = orderType,
            ["ordertype"] = "market",
            ["volume"]    = volume.ToString("F8"),
        });
        var errors = resp["error"]?.AsArray();
        if (errors is { Count: > 0 })
        {
            Console.WriteLine($"{Capitalize(orderType)} order error for {pair}: {errors}");
            return false;
        }
        Console.WriteLine($"{Capitalize(orderType)} order placed for {pair}: {resp["result"]}");
        return true;
    }

    private static string Capitalize(string s) =>
        s.Length == 0 ? s : char.ToUpper(s[0]) + s[1..];

    // ---- Buy / Sell -------------------------------------------------------

    public async Task BuyCryptoAsync(List<List<string>> cryptoData, string name)
    {
        ClearCryptoData(name);
        double price = double.Parse(cryptoData[^1][Config.OhlcClose]);
        double funds = await GetAvailableFundsAsync();
        if (funds <= 0)
        {
            Console.WriteLine($"No funds available to buy {name}");
            return;
        }
        double amount = funds / price;
        if (await PlaceOrderAsync(name, "buy", amount))
            SaveTrade(price, name, bought: true, sold: false, amount);
    }

    public async Task SellCryptoAsync(List<List<string>> cryptoData, string name)
    {
        ClearCryptoData(name);
        double price   = double.Parse(cryptoData[^1][Config.OhlcClose]);
        var    trades  = LoadTrades();
        var    lastBuy = trades[name].LastOrDefault(t => t.Bought);
        if (lastBuy is null)
        {
            Console.WriteLine($"No buy trade found for {name}, cannot sell");
            return;
        }
        if (await PlaceOrderAsync(name, "sell", lastBuy.Amount))
            SaveTrade(price, name, bought: false, sold: true, lastBuy.Amount);
    }

    // ---- Data management --------------------------------------------------

    public Dictionary<string, PairData> ClearCryptoData(string name)
    {
        var data = LoadCryptoDataFromFile();
        var pd   = data[name];
        pd.High   = [.. pd.High.TakeLast(10)];
        pd.Low    = [.. pd.Low.TakeLast(10)];
        pd.Close  = [.. pd.Close.TakeLast(10)];
        pd.Prices = [.. pd.Prices.TakeLast(10)];
        SaveCryptoData(data);
        return data;
    }

    // ---- Analysis ---------------------------------------------------------

    public async Task CheckDataAsync(
        string name,
        List<List<string>> cryptoData,
        bool shouldBuy,
        Dictionary<string, PairData> mva)
    {
        double high = 0, low = 0, close = 0;
        var pd      = mva[name];
        var last100 = cryptoData.TakeLast(100).ToList();

        foreach (var b in last100)
        {
            if (!pd.Prices.Any(p => p.SequenceEqual(b)))
                pd.Prices.Add(b);
            high  += double.Parse(b[Config.OhlcHigh]);
            low   += double.Parse(b[Config.OhlcLow]);
            close += double.Parse(b[Config.OhlcClose]);
        }
        pd.High.Add(high  / 100);
        pd.Low.Add(low    / 100);
        pd.Close.Add(close / 100);

        // Cap MA lists to prevent unbounded growth
        if (pd.High.Count  > Config.MaxMaHistory) pd.High  = [.. pd.High.TakeLast(Config.MaxMaHistory)];
        if (pd.Low.Count   > Config.MaxMaHistory) pd.Low   = [.. pd.Low.TakeLast(Config.MaxMaHistory)];
        if (pd.Close.Count > Config.MaxMaHistory) pd.Close = [.. pd.Close.TakeLast(Config.MaxMaHistory)];

        SaveCryptoData(mva);

        if (shouldBuy)
            await TryBuyAsync(pd, name, cryptoData);
        else
            await TrySellAsync(pd, name, cryptoData);
    }

    public async Task TryBuyAsync(PairData data, string name, List<List<string>> cryptoData)
    {
        if (CheckOpportunity(data, name, sell: false, buy: true) == true)
        {
            Console.WriteLine("Buy Crypto " + name);
            await BuyCryptoAsync(cryptoData, name);
        }
    }

    public async Task TrySellAsync(PairData data, string name, List<List<string>> cryptoData)
    {
        if (CheckOpportunity(data, name, sell: true, buy: false) == true)
        {
            Console.WriteLine("Sell Crypto " + name);
            await SellCryptoAsync(cryptoData, name);
        }
    }

    // Returns true  → act on the signal
    //         false → no signal (buy path only)
    //         null  → no signal (sell path, no exit conditions met)
    public bool? CheckOpportunity(PairData data, string name, bool sell, bool buy)
    {
        int    count         = 0;
        double previousValue = 0;

        foreach (var mva in data.Close.TakeLast(10))
        {
            if (previousValue == 0)
            {
                previousValue = mva;
            }
            else
            {
                if (mva / previousValue > 1)
                    count = count < 1 ? 1 : count + 1;
                else if (mva / previousValue < 1)
                    count = count > 0 ? -1 : count - 1;
                previousValue = mva;
            }
        }

        if (data.Prices.Count == 0) return buy ? false : null;
        double price = double.Parse(data.Prices[^1][Config.OhlcClose]);

        if (sell)
        {
            double purchasePrice = GetPurchasingPrice(name);
            double peakMa        = data.Close.Count > 0 ? data.Close.Max() : purchasePrice;

            if (price >= purchasePrice * Config.ProfitTarget)
            {
                Console.WriteLine($"Taking profit (+{(int)((Config.ProfitTarget - 1) * 100)}%)");
                return true;
            }
            if (price < purchasePrice * Config.StopLoss)
            {
                Console.WriteLine("Stop-loss triggered (-1%)");
                return true;
            }
            if (price < peakMa * Config.TrailingStop)
            {
                Console.WriteLine("Trailing stop triggered (-1% from peak MA)");
                return true;
            }
            return null;
        }

        if (buy)
        {
            if (count >= 5)
            {
                var areas = data.Close.TakeLast(5).Reverse().Select(mva => mva / price).ToList();
                if (areas.Count > 0 && areas.Average() >= 0.93)
                    return true;
            }
            return false;
        }

        return null;
    }

    public double GetPurchasingPrice(string name)
    {
        var trades = LoadTrades();
        return trades[name][^1].PriceUsd;
    }

    // ---- Main loop --------------------------------------------------------

    public async Task RunAsync(string[] pairs)
    {
        var mva = LoadCryptoDataFromFile();
        while (true)
        {
            string since  = (DateTimeOffset.UtcNow.ToUnixTimeSeconds() - Config.WindowSeconds).ToString();
            var    trades = LoadTrades(); // load once per cycle, shared across all pairs

            foreach (var pair in pairs)
            {
                try
                {
                    var cryptoData = await GetCryptoDataAsync(pair, since);
                    var lastTrade  = trades[pair].Count > 0 ? trades[pair][^1] : null;
                    bool shouldBuy = lastTrade is null || lastTrade.Sold;
                    await CheckDataAsync(pair, cryptoData, shouldBuy, mva);
                    mva = LoadCryptoDataFromFile();
                }
                catch (Exception e)
                {
                    Console.WriteLine($"Error processing {pair}: {e.Message}");
                }
            }
            await Task.Delay(TimeSpan.FromSeconds(20));
        }
    }

    // ---- Init -------------------------------------------------------------

    public static void InitFiles()
    {
        try
        {
            using var f = new FileStream("trades.json", FileMode.CreateNew);
            var empty   = Config.Pairs.ToDictionary(p => p, _ => new List<Trade>());
            JsonSerializer.Serialize(f, empty, new JsonSerializerOptions { WriteIndented = true });
        }
        catch (IOException) { }

        try
        {
            using var f = new FileStream("data.json", FileMode.CreateNew);
            JsonSerializer.Serialize(f, MakeCryptoData(), new JsonSerializerOptions { WriteIndented = true });
        }
        catch (IOException) { }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
Bot.InitFiles();
var krakenClient = new KrakenClient();
krakenClient.LoadKey("kraken.key");
await new Bot(krakenClient).RunAsync(Config.Pairs);
