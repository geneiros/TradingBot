"""
sma_crossover_bot.py — SMA-20 / SMA-50 crossover strategy

Buy signal : SMA20 crosses ABOVE SMA50 (golden cross)
Sell signal: SMA20 crosses BELOW SMA50 (death cross) AND price > buy price
Stop-loss  : price drops below buy_price × 0.99 (hard exit, same as bot.py)

Uses separate state files (sma_state.json, sma_trades.json) so it can
run alongside bot.py without conflicts.
"""

import krakenex
import time
import json

k = None  # set in __main__

# OHLC candle field index
_OHLC_CLOSE = 4

# Strategy parameters
_SMA_SHORT      = 20
_SMA_LONG       = 50
_STOP_LOSS      = 0.99    # hard stop at -1% from entry price
_WINDOW_SECONDS = 3600    # 1 hour of 1-min candles → ~60 candles, enough for SMA50

_PAIRS = ('XETHZEUR', 'XXBTZEUR', 'MANAEUR', 'GRTEUR', 'LSKEUR', 'SCEUR')

_STATE_FILE  = 'sma_state.json'
_TRADES_FILE = 'sma_trades.json'


# ── Kraken helpers ─────────────────────────────────────────────────────────────

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


def get_ohlc(pair, since):
    ret = k.query_public('OHLC', data={'pair': pair, 'since': since})
    if ret.get('error'):
        raise RuntimeError(f'OHLC error for {pair}: {ret["error"]}')
    return ret['result'][pair]


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
    balance    = get_balance()
    money      = float(balance.get('ZEUR', 0))
    trades     = load_trades()
    pairs_held = sum(
        1 for pair in _PAIRS
        if trades[pair] and trades[pair][-1]['bought'] and not trades[pair][-1]['sold']
    )
    slots_free = len(_PAIRS) - pairs_held
    if slots_free <= 0:
        return 0.0
    return money / slots_free


# ── Trade persistence ──────────────────────────────────────────────────────────

def load_trades():
    try:
        with open(_TRADES_FILE) as f:
            return json.load(f)
    except Exception:
        return {pair: [] for pair in _PAIRS}


def save_trade(price, name, bought, sold, amount):
    trade = {
        'time_stamp': str(int(time.time())),
        'price_usd':  price,
        'bought':     bought,
        'sold':       sold,
        'amount':     amount,
    }
    print(f'TRADE: {json.dumps(trade, indent=4)}')
    trades = load_trades()
    trades[name].append(trade)
    with open(_TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=4)


def get_purchase_price(name):
    trades = load_trades()
    last_buy = next((t for t in reversed(trades[name]) if t['bought']), None)
    if last_buy is None:
        raise ValueError(f'No buy trade found for {name}')
    return float(last_buy['price_usd'])


# ── State persistence (SMA crossover requires prev_above across cycles) ────────

def load_state():
    try:
        with open(_STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return _make_state()


def _make_state():
    return {
        pair: {
            'prev_above': None,   # None = unknown (not enough data yet)
            'peak_price': 0.0,
        }
        for pair in _PAIRS
    }


def save_state(state):
    with open(_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)


# ── Buy / Sell execution ───────────────────────────────────────────────────────

def buy_crypto(name, price, state):
    funds = get_available_funds()
    if funds <= 0:
        print(f'[{name}] No funds available to buy')
        return
    amount = funds / price
    if _place_order(name, 'buy', amount):
        save_trade(price, name, True, False, amount)
        state[name]['peak_price'] = price   # reset peak on new entry


def sell_crypto(name, price, state):
    trades   = load_trades()
    last_buy = next((t for t in reversed(trades[name]) if t['bought']), None)
    if last_buy is None:
        print(f'[{name}] No buy trade found, cannot sell')
        return
    amount = float(last_buy['amount'])
    if _place_order(name, 'sell', amount):
        save_trade(price, name, False, True, amount)
        state[name]['peak_price'] = 0.0


# ── Signal logic ───────────────────────────────────────────────────────────────

def _sma(closes, n):
    return sum(closes[-n:]) / n


def check_pair(name, ohlc_candles, should_buy, state):
    closes = [float(c[_OHLC_CLOSE]) for c in ohlc_candles]

    if len(closes) < _SMA_LONG:
        print(f'[{name}] Waiting for data ({len(closes)}/{_SMA_LONG} candles)')
        return

    sma20 = _sma(closes, _SMA_SHORT)
    sma50 = _sma(closes, _SMA_LONG)
    price = closes[-1]

    curr_above = sma20 > sma50
    prev_above = state[name]['prev_above']   # True | False | None

    # Crossover flags (require a known previous state to avoid false signals on startup)
    golden_cross = (prev_above is False) and curr_above
    death_cross  = (prev_above is True)  and (not curr_above)

    state[name]['prev_above'] = curr_above

    arrow = '▲' if curr_above else '▼'
    print(f'[{name}] price={price:.6f}  SMA{_SMA_SHORT}={sma20:.6f}  '
          f'SMA{_SMA_LONG}={sma50:.6f}  {arrow}')

    if should_buy:
        if golden_cross:
            print(f'[{name}] Golden cross → BUY')
            buy_crypto(name, price, state)
        else:
            print(f'[{name}] BUY mode — waiting for golden cross')
    else:
        # Update running peak for informational purposes
        if price > state[name].get('peak_price', 0.0):
            state[name]['peak_price'] = price

        try:
            buy_price  = get_purchase_price(name)
            profit_pct = (price / buy_price - 1) * 100
            peak       = state[name].get('peak_price', 0.0)
            print(f'[{name}] HOLD | buy={buy_price:.6f}  profit={profit_pct:+.4f}%  '
                  f'peak={peak:.6f}')
        except ValueError:
            buy_price  = 0.0
            profit_pct = 0.0

        if buy_price > 0 and price < buy_price * _STOP_LOSS:
            print(f'[{name}] Stop-loss triggered ({profit_pct:+.4f}%) → SELL')
            sell_crypto(name, price, state)
        elif death_cross and buy_price > 0 and price > buy_price:
            print(f'[{name}] Death cross + profitable ({profit_pct:+.4f}%) → SELL')
            sell_crypto(name, price, state)
        elif death_cross:
            print(f'[{name}] Death cross but price ({profit_pct:+.4f}%) — holding')


# ── Main loop ─────────────────────────────────────────────────────────────────

def bot(k_api, pairs):
    global k
    k = k_api

    while True:
        since  = str(int(time.time() - _WINDOW_SECONDS))
        state  = load_state()
        trades = load_trades()
        print('─' * 60)

        for pair in pairs:
            try:
                ohlc      = get_ohlc(pair, since)
                last      = trades[pair][-1] if trades[pair] else None
                should_buy = last is None or last['sold']
                check_pair(pair, ohlc, should_buy, state)
            except Exception as e:
                print(f'Error processing {pair}: {e}')

        save_state(state)
        time.sleep(20)


def _init_files():
    try:
        with open(_TRADES_FILE, 'x') as f:
            json.dump({p: [] for p in _PAIRS}, f, indent=4)
    except FileExistsError:
        pass
    try:
        with open(_STATE_FILE, 'x') as f:
            json.dump(_make_state(), f, indent=4)
    except FileExistsError:
        pass


if __name__ == '__main__':
    _init_files()
    k = krakenex.API()
    k.load_key('kraken.key')
    bot(k, _PAIRS)
