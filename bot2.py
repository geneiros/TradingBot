import krakenex
import time
import json

k = None  # set in __main__

# OHLC candle field indices
_OHLC_CLOSE = 4

# Trading thresholds
_STOP_LOSS      = 0.99   # stop-loss at -1%
_MIN_PROFIT     = 1.03   # SMA crossdown sell only if >= +3% profit
_SMA_FAST       = 20     # fast SMA period (minutes)
_SMA_SLOW       = 50     # slow SMA period (minutes)
_OHLC_INTERVAL  = 1      # 1-minute candles
# Fetch enough candles to always cover SMA50 + 1 previous bar
_FETCH_MINUTES  = (_SMA_SLOW + 5) * _OHLC_INTERVAL

_PAIRS = ('XETHZEUR', 'XXBTZEUR', 'MANAEUR', 'GRTEUR', 'LSKEUR', 'SCEUR')


# ---------------------------------------------------------------------------
# Kraken helpers
# ---------------------------------------------------------------------------

def get_balance():
    try:
        resp = k.query_private('Balance')
        if resp.get('error'):
            print(f'Balance API error: {resp["error"]}')
            return {}
        return resp['result']
    except Exception as e:
        print(f'Balance exception: {e}')
        return {}


def get_crypto_data(pair, since):
    ret = k.query_public('OHLC', data={
        'pair':     pair,
        'interval': _OHLC_INTERVAL,
        'since':    since,
    })
    if ret.get('error'):
        raise RuntimeError(f'OHLC error for {pair}: {ret["error"]}')
    return ret['result'][pair]


# ---------------------------------------------------------------------------
# Trade persistence
# ---------------------------------------------------------------------------

def load_trades():
    try:
        with open('trades2.json', 'r') as f:
            return json.load(f)
    except Exception:
        trades = {pair: [] for pair in _PAIRS}
        save_trades(trades)
        return trades


def save_trades(trades):
    with open('trades2.json', 'w') as f:
        json.dump(trades, f, indent=4)


def save_trade(price, name, bought, sold, amount):
    trade = {
        'time_stamp': str(int(time.time())),
        'price_usd':  price,
        'bought':     bought,
        'sold':       sold,
        'amount':     amount,
    }
    print('TRADE:')
    print(json.dumps(trade, indent=4))
    trades = load_trades()
    trades[name].append(trade)
    save_trades(trades)


def get_purchasing_price(name):
    trades = load_trades()
    last_buy = next((t for t in reversed(trades[name]) if t['bought']), None)
    if last_buy is None:
        return None
    return float(last_buy['price_usd'])


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

def _place_order(pair, order_type, volume):
    resp = k.query_private('AddOrder', {
        'pair':      pair,
        'type':      order_type,
        'ordertype': 'market',
        'volume':    f'{volume:.8f}',
    })
    if resp.get('error'):
        print(f'{order_type.capitalize()} order error for {pair}: {resp["error"]}')
        return False
    print(f'{order_type.capitalize()} order placed for {pair}: {resp["result"]}')
    return True


def get_available_funds():
    balance  = get_balance()
    money    = float(balance.get('ZEUR', 0))
    trades   = load_trades()
    pairs_held = sum(
        1 for pair in _PAIRS
        if trades[pair] and trades[pair][-1]['bought'] and not trades[pair][-1]['sold']
    )
    slots_free = len(_PAIRS) - pairs_held
    if slots_free <= 0:
        return 0.0
    return money / slots_free


def buy_crypto(candles, name):
    price = float(candles[-1][_OHLC_CLOSE])
    funds = get_available_funds()
    if funds <= 0:
        print(f'No funds available to buy {name}')
        return
    amount = funds / price
    if _place_order(name, 'buy', amount):
        save_trade(price, name, True, False, amount)


def sell_crypto(candles, name):
    price    = float(candles[-1][_OHLC_CLOSE])
    trades   = load_trades()
    last_buy = next((t for t in reversed(trades[name]) if t['bought']), None)
    if last_buy is None:
        print(f'No buy trade found for {name}, cannot sell')
        return
    amount = float(last_buy['amount'])
    if _place_order(name, 'sell', amount):
        save_trade(price, name, False, True, amount)


# ---------------------------------------------------------------------------
# SMA crossover logic
# ---------------------------------------------------------------------------

def sma(closes, period):
    """Simple moving average of the last `period` values."""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def check_signals(name, candles, should_buy):
    """
    Evaluate SMA20/SMA50 crossover signals on 1-minute closes.

    Buy  : SMA20 crosses ABOVE SMA50  (previous SMA20 < SMA50, current SMA20 > SMA50)
    Sell : SMA20 crosses BELOW SMA50  AND price >= buy_price * 1.03
    Stop : price < buy_price * 0.99   (always, regardless of crossover)
    """
    closes = [float(c[_OHLC_CLOSE]) for c in candles]

    # Need at least SMA_SLOW + 1 bars to detect a crossover
    if len(closes) < _SMA_SLOW + 1:
        print(f'{name}: not enough candles ({len(closes)}) for SMA{_SMA_SLOW}')
        return

    # Current and previous bar SMAs
    sma20_now  = sma(closes,        _SMA_FAST)
    sma50_now  = sma(closes,        _SMA_SLOW)
    sma20_prev = sma(closes[:-1],   _SMA_FAST)
    sma50_prev = sma(closes[:-1],   _SMA_SLOW)

    if None in (sma20_now, sma50_now, sma20_prev, sma50_prev):
        return

    price = closes[-1]
    print(f'{name} | price={price:.4f}  SMA20={sma20_now:.4f}  SMA50={sma50_now:.4f}')

    if should_buy:
        # Golden cross: SMA20 crosses above SMA50
        if sma20_prev <= sma50_prev and sma20_now > sma50_now:
            print(f'{name}: SMA20 crossed ABOVE SMA50 → BUY')
            buy_crypto(candles, name)

    else:
        purchase_price = get_purchasing_price(name)
        if purchase_price is None:
            return

        # Stop-loss: -1% from purchase price (always checked first)
        if price < purchase_price * _STOP_LOSS:
            print(f'{name}: stop-loss triggered (-1%) → SELL')
            sell_crypto(candles, name)
            return

        # Death cross: SMA20 crosses below SMA50, but only if profit >= +3%
        if sma20_prev >= sma50_prev and sma20_now < sma50_now:
            if price >= purchase_price * _MIN_PROFIT:
                print(f'{name}: SMA20 crossed BELOW SMA50 with ≥3% profit → SELL')
                sell_crypto(candles, name)
            else:
                profit_pct = (price / purchase_price - 1) * 100
                print(f'{name}: death cross detected but profit {profit_pct:.2f}% < 3%, holding')


# ---------------------------------------------------------------------------
# Main bot loop
# ---------------------------------------------------------------------------

def bot(k_api, pairs):
    global k
    k = k_api
    while True:
        since = str(int(time.time() - _FETCH_MINUTES * 60))
        trades = load_trades()
        for pair in pairs:
            try:
                candles    = get_crypto_data(pair, since)
                last_trade = trades[pair][-1] if trades[pair] else None
                should_buy = last_trade is None or last_trade['sold']
                check_signals(pair, candles, should_buy)
            except Exception as e:
                print(f'Error processing {pair}: {e}')
        time.sleep(60)  # re-evaluate every minute


def _init_files():
    try:
        with open('trades2.json', 'x') as f:
            json.dump({p: [] for p in _PAIRS}, f, indent=4)
    except FileExistsError:
        pass


if __name__ == '__main__':
    _init_files()
    k = krakenex.API()
    k.load_key('kraken.key')
    bot(k, _PAIRS)
