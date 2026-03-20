import krakenex
import time
import decimal
import json

k = None  # set in __main__

# OHLC candle field indices
_OHLC_HIGH = 2
_OHLC_LOW = 3
_OHLC_CLOSE = 4

# Trading thresholds
_PROFIT_TARGET = 1.03   # take profit at +3%
_STOP_LOSS = 0.99       # stop-loss at -1%
_TRAILING_STOP = 0.99   # trailing stop: -1% from peak MA
_WINDOW_SECONDS = 43200  # 12-hour rolling window
_MAX_MA_HISTORY = 200   # cap for MA lists

_PAIRS = ('XETHZEUR', 'XXBTZEUR', 'MANAEUR', 'GRTEUR', 'LSKEUR', 'SCEUR')


def now():
    return decimal.Decimal(time.time())


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
    ret = k.query_public('OHLC', data={'pair': pair, 'since': since})
    if ret.get('error'):
        raise RuntimeError(f'OHLC error for {pair}: {ret["error"]}')
    return ret['result'][pair]


def get_purchasing_price(name):
    trades = load_trades()
    return trades[name][-1]['price_usd']


def load_trades():
    trades = {}
    try:
        with open('trades.json', 'r') as f:
            trades = json.load(f)
    except Exception:
        for pair in _PAIRS:
            trades[pair] = []
    return trades


def save_crypto_data(data):
    with open('data.json', 'w') as f:
        json.dump(data, f, indent=4)


def load_crypto_data_from_file():
    data = {}
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
    except Exception:
        data = make_crypto_data(data)
        save_crypto_data(data)
    return data


def make_crypto_data(data):
    for name in _PAIRS:
        data[name] = {
            'high': [],
            'low': [],
            'close': [],
            'prices': []
        }
    return data


def save_trade(close, name, bought, sold, amount):
    trade = {
        'time_stamp': str(int(time.time())),
        'price_usd': close,
        'bought': bought,
        'sold': sold,
        'amount': amount
    }
    print('TRADE:')
    print(json.dumps(trade, indent=4))
    trades = load_trades()
    trades[name].append(trade)
    with open('trades.json', 'w') as f:
        json.dump(trades, f, indent=4)


def _place_order(pair, order_type, volume):
    """Place a market order. Returns True on success, False on API error."""
    resp = k.query_private('AddOrder', {
        'pair': pair,
        'type': order_type,
        'ordertype': 'market',
        'volume': f'{volume:.8f}',
    })
    if resp.get('error'):
        print(f'{order_type.capitalize()} order error for {pair}: {resp["error"]}')
        return False
    print(f'{order_type.capitalize()} order placed for {pair}: {resp["result"]}')
    return True


def buy_crypto(crypto_data, name):
    clear_crypto_data(name)
    price = float(crypto_data[-1][_OHLC_CLOSE])
    funds = get_available_funds()
    if funds <= 0:
        print(f'No funds available to buy {name}')
        return
    amount = funds / price
    if _place_order(name, 'buy', amount):
        save_trade(price, name, True, False, amount)


def sell_crypto(crypto_data, name):
    clear_crypto_data(name)
    price = float(crypto_data[-1][_OHLC_CLOSE])
    # Get amount from last buy trade (avoids Kraken balance key naming issues)
    trades = load_trades()
    last_buy = next((t for t in reversed(trades[name]) if t['bought']), None)
    if last_buy is None:
        print(f'No buy trade found for {name}, cannot sell')
        return
    amount = float(last_buy['amount'])
    if _place_order(name, 'sell', amount):
        save_trade(price, name, False, True, amount)


def clear_crypto_data(name):
    data = load_crypto_data_from_file()
    for key in data[name]:
        data[name][key] = data[name][key][-10:]
    save_crypto_data(data)
    return data


def delete_entries(data, key):
    return data[key][-10:]


def get_available_funds():
    balance = get_balance()
    money = float(balance.get('ZEUR', 0))
    # Count how many of our pairs are currently held (have an open buy)
    trades = load_trades()
    pairs_held = sum(
        1 for pair in _PAIRS
        if trades[pair] and trades[pair][-1]['bought'] and not trades[pair][-1]['sold']
    )
    slots_free = len(_PAIRS) - pairs_held
    if slots_free <= 0:
        return 0.0
    return money / slots_free


def bot(k_api, pairs):
    global k
    k = k_api
    mva = load_crypto_data_from_file()
    while True:
        since = str(int(time.time() - _WINDOW_SECONDS))
        trades = load_trades()  # load once per cycle, shared across all pairs
        for pair in pairs:
            try:
                crypto_data = get_crypto_data(pair, since)
                last_trade = trades[pair][-1] if trades[pair] else None
                should_buy = last_trade is None or last_trade['sold']
                check_data(pair, crypto_data, should_buy, mva)
                mva = load_crypto_data_from_file()
            except Exception as e:
                print(f'Error processing {pair}: {e}')
        time.sleep(20)


def check_data(name, crypto_data, should_buy, mva):
    high = 0
    low = 0
    close = 0
    for b in crypto_data[-100:]:
        if b not in mva[name]['prices']:
            mva[name]['prices'].append(b)
        high  += float(b[_OHLC_HIGH])
        low   += float(b[_OHLC_LOW])
        close += float(b[_OHLC_CLOSE])
    mva[name]['high'].append(high / 100)
    mva[name]['low'].append(low / 100)
    mva[name]['close'].append(close / 100)
    # Cap MA lists to prevent unbounded growth
    for key in ('high', 'low', 'close'):
        if len(mva[name][key]) > _MAX_MA_HISTORY:
            mva[name][key] = mva[name][key][-_MAX_MA_HISTORY:]
    save_crypto_data(mva)

    if should_buy:
        try_buy(mva[name], name, crypto_data)
    else:
        try_sell(mva[name], name, crypto_data)


def try_buy(data, name, crypto_data):
    price = float(crypto_data[-1][_OHLC_CLOSE])
    print(f'[{name}] BUY mode  | price={price:.2f}')
    make_trade = check_opportunity(data, name, crypto_data, False, True)
    if make_trade:
        print('Buy Crypto ' + name)
        buy_crypto(crypto_data, name)


def check_opportunity(data, name, crypto_data, sell, buy):
    count = 0
    previous_value = 0
    trends = []
    for mva in data['close'][-10:]:
        if previous_value == 0:
            previous_value = mva
        else:
            if mva / previous_value > 1:
                if count < 1:
                    count = 1
                else:
                    count += 1
                trends.append('UPTREND')
            elif mva / previous_value < 1:
                trends.append('DOWNTREND')
                if count > 0:
                    count = -1
                else:
                    count -= 1
            else:
                trends.append('NOTREND')
            previous_value = mva

    areas = []
    price = float(crypto_data[-1][_OHLC_CLOSE])  # always use live price

    if sell:
        purchase_price = float(get_purchasing_price(name))
        profit_pct = (price / purchase_price - 1) * 100
        peak_ma = max(data['close']) if data['close'] else purchase_price
        print(f'[{name}] SELL mode | price={price:.2f} buy={purchase_price:.2f} profit={profit_pct:+.2f}% | peak_ma={peak_ma:.2f}')
        if price >= purchase_price * _PROFIT_TARGET:
            print(f'Taking profit at +3% → SELL')
            return True
        if price < purchase_price * _STOP_LOSS:
            print('Stop-loss triggered (-1%)')
            return True
        if price < peak_ma * _TRAILING_STOP:
            print('Trailing stop triggered (-1% from peak MA)')
            return True

    for mva in reversed(data['close'][-5:]):
        areas.append(mva / price)

    if buy:
        counter = 0
        if count >= 5:
            for area in areas:
                counter += area
            if len(areas) > 0 and counter / len(areas) >= 0.93:
                return True
        return False


def try_sell(data, name, crypto_data):
    make_trade = check_opportunity(data, name, crypto_data, True, False)
    if make_trade:
        print('Sell Crypto ' + name)
        sell_crypto(crypto_data, name)


def get_pairs():
    return _PAIRS


def _init_files():
    try:
        with open('trades.json', 'x') as f:
            json.dump({p: [] for p in _PAIRS}, f, indent=4)
    except FileExistsError:
        pass
    try:
        with open('data.json', 'x') as f:
            json.dump(make_crypto_data({}), f, indent=4)
    except FileExistsError:
        pass


if __name__ == '__main__':
    _init_files()
    k = krakenex.API()
    k.load_key('kraken.key')
    bot(k, _PAIRS)
