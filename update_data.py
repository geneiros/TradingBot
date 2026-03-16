"""
Backfill data.json with ~2 months of hourly OHLC data from Kraken.
Uses the same MA processing logic as bot.py's check_data().
"""
import json
import time
import urllib.request
import urllib.parse

# ---- config ---------------------------------------------------------------
PAIRS         = ('XETHZUSD', 'XXBTZUSD', 'MANAUSD', 'GRTUSD', 'LSKUSD', 'SCUSD')
INTERVAL      = 60          # minutes per candle
MONTHS_BACK   = 2
MA_WINDOW     = 100         # candles per MA data point (matches bot.py)
MAX_MA_HISTORY = 200
OHLC_HIGH, OHLC_LOW, OHLC_CLOSE = 2, 3, 4

# ---------------------------------------------------------------------------

def fetch_ohlc(pair: str, since: int, interval: int) -> list:
    """Fetch OHLC candles from Kraken; returns list of candle lists."""
    params  = urllib.parse.urlencode({'pair': pair, 'since': since, 'interval': interval})
    url     = f'https://api.kraken.com/0/public/OHLC?{params}'
    with urllib.request.urlopen(url, timeout=30) as r:
        resp = json.loads(r.read())
    if resp.get('error'):
        raise RuntimeError(f'OHLC error for {pair}: {resp["error"]}')
    candles = resp['result'].get(pair) or resp['result'].get(list(resp['result'].keys())[0])
    last    = int(resp['result'].get('last', 0))
    return candles, last


def fetch_all_candles(pair: str, since: int, interval: int) -> list:
    """Paginate through Kraken OHLC until we reach the present."""
    all_candles = []
    cursor      = since
    while True:
        print(f'  Fetching {pair} since {cursor} …', flush=True)
        candles, last = fetch_ohlc(pair, cursor, interval)
        if not candles:
            break
        all_candles.extend(candles)
        # Kraken returns `last` as the timestamp of the last returned candle;
        # stop when it's within one interval of now
        if last >= int(time.time()) - interval * 60:
            break
        cursor = last
        time.sleep(1)   # be polite to the API
    # de-duplicate by candle timestamp (first field)
    seen, unique = set(), []
    for c in all_candles:
        t = c[0]
        if t not in seen:
            seen.add(t)
            unique.append(c)
    return sorted(unique, key=lambda c: c[0])


def build_pair_data(candles: list) -> dict:
    """
    Replicate the bot's check_data() MA logic over a full history.
    For every window of MA_WINDOW consecutive candles, emit one MA entry.
    """
    prices = []
    highs, lows, closes = [], [], []

    for i in range(MA_WINDOW, len(candles) + 1):
        window = candles[i - MA_WINDOW : i]
        h = sum(float(b[OHLC_HIGH])  for b in window) / MA_WINDOW
        l = sum(float(b[OHLC_LOW])   for b in window) / MA_WINDOW
        c = sum(float(b[OHLC_CLOSE]) for b in window) / MA_WINDOW
        highs.append(h)
        lows.append(l)
        closes.append(c)

    # Prices: store the raw candles as lists of strings (matches bot.py format)
    prices = [[str(v) for v in b] for b in candles]

    # Cap to MAX_MA_HISTORY
    return {
        'high':   highs[-MAX_MA_HISTORY:],
        'low':    lows[-MAX_MA_HISTORY:],
        'close':  closes[-MAX_MA_HISTORY:],
        'prices': prices,
    }


def main():
    since = int(time.time()) - MONTHS_BACK * 30 * 24 * 3600
    data  = {}

    for pair in PAIRS:
        print(f'\n=== {pair} ===')
        try:
            candles = fetch_all_candles(pair, since, INTERVAL)
            print(f'  {len(candles)} candles fetched')
            if len(candles) < MA_WINDOW:
                print(f'  WARNING: fewer than {MA_WINDOW} candles, MA will be empty')
            data[pair] = build_pair_data(candles)
            print(f'  MA entries: {len(data[pair]["close"])}  prices: {len(data[pair]["prices"])}')
        except Exception as e:
            print(f'  ERROR: {e}')
            data[pair] = {'high': [], 'low': [], 'close': [], 'prices': []}

    with open('data.json', 'w') as f:
        json.dump(data, f, indent=4)
    print('\ndata.json updated successfully.')


if __name__ == '__main__':
    main()
