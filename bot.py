import krakenex
import time
import decimal
import json

k = None  # set in __main__


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
        for crypto in get_pairs():
            trades[crypto] = []
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
    for name in get_pairs():
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


def buy_crypto(crypto_data, name):
    clear_crypto_data(name)
    price = float(crypto_data[-1][4])
    funds = get_available_funds()
    if funds <= 0:
        print(f'No funds available to buy {name}')
        return
    amount = funds / price
    resp = k.query_private('AddOrder', {
        'pair': name,
        'type': 'buy',
        'ordertype': 'market',
        'volume': f'{amount:.8f}',
    })
    if resp.get('error'):
        print(f'Buy order error for {name}: {resp["error"]}')
        return
    print(f'Buy order placed for {name}: {resp["result"]}')
    save_trade(price, name, True, False, amount)


def sell_crypto(crypto_data, name):
    clear_crypto_data(name)
    price = float(crypto_data[-1][4])
    # Get amount from last buy trade (avoids Kraken balance key naming issues)
    trades = load_trades()
    last_buy = next((t for t in reversed(trades[name]) if t['bought']), None)
    if last_buy is None:
        print(f'No buy trade found for {name}, cannot sell')
        return
    amount = float(last_buy['amount'])
    resp = k.query_private('AddOrder', {
        'pair': name,
        'type': 'sell',
        'ordertype': 'market',
        'volume': f'{amount:.8f}',
    })
    if resp.get('error'):
        print(f'Sell order error for {name}: {resp["error"]}')
        return
    print(f'Sell order placed for {name}: {resp["result"]}')
    save_trade(price, name, False, True, amount)


def clear_crypto_data(name):
    data = load_crypto_data_from_file()
    for key in data[name]:
        data[name][key] = delete_entries(data[name], key)
    save_crypto_data(data)
    return data


def delete_entries(data, key):
    clean_array = []
    for entry in data[key][-10:]:
        clean_array.append(entry)
    return clean_array


def get_available_funds():
    balance = get_balance()
    money = float(balance.get('ZUSD', 0))
    # Count how many of our pairs are currently held (have an open buy)
    trades = load_trades()
    pairs_held = sum(
        1 for pair in get_pairs()
        if trades[pair] and trades[pair][-1]['bought'] and not trades[pair][-1]['sold']
    )
    slots_free = len(get_pairs()) - pairs_held
    if slots_free <= 0:
        return 0.0
    return money / slots_free


def bot(k_api, pairs):
    global k
    k = k_api
    mva = load_crypto_data_from_file()
    while True:
        since = str(int(time.time() - 43200))  # rolling 12h window each cycle
        for pair in pairs:
            try:
                trades = load_trades()
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
        high  += float(b[2])
        low   += float(b[3])
        close += float(b[4])
    mva[name]['high'].append(high / 100)
    mva[name]['low'].append(low / 100)
    mva[name]['close'].append(close / 100)
    save_crypto_data(mva)

    if should_buy:
        try_buy(mva[name], name, crypto_data)
    else:
        try_sell(mva[name], name, crypto_data)


def try_buy(data, name, crypto_data):
    make_trade = check_opportunity(data, name, False, True)
    if make_trade:
        print('Buy Crypto ' + name)
        buy_crypto(crypto_data, name)


def check_opportunity(data, name, sell, buy):
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
    # FIX 1: use close price [4] instead of low price [3]
    price = float(data['prices'][-1][4])
    for mva in reversed(data['close'][-5:]):
        if sell:
            purchase_price = float(get_purchasing_price(name))
            peak_ma = max(data['close']) if data['close'] else purchase_price
            if price >= (purchase_price * 1.05):
                print('Taking profit (+5%)')
                return True
            if price < purchase_price * 0.99:
                print('Stop-loss triggered (-1%)')
                return True
            if price < peak_ma * 0.99:
                print('Trailing stop triggered (-1% from peak MA)')
                return True
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
    make_trade = check_opportunity(data, name, True, False)
    if make_trade:
        print('Sell Crypto ' + name)
        sell_crypto(crypto_data, name)


def get_pairs():
    return ('XETHZUSD', 'XXBTZUSD', 'MANAUSD', 'GRTUSD', 'LSKUSD', 'SCUSD')


def _init_files():
    import os
    if not os.path.exists('trades.json'):
        with open('trades.json', 'w') as f:
            json.dump({p: [] for p in get_pairs()}, f, indent=4)
    if not os.path.exists('data.json'):
        save_crypto_data(make_crypto_data({}))


if __name__ == '__main__':
    _init_files()
    k = krakenex.API()
    k.load_key('kraken.key')
    pairs = get_pairs()
    bot(k, pairs)
