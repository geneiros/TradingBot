"""
sim_kraken.py — in-memory stand-in for the Kraken API used by simulations.

FakeKraken implements just enough of krakenex.API (query_public 'OHLC',
query_private 'Balance' / 'AddOrder') for bot.py to run unmodified on
synthetic data. Market orders fill instantly at the last close of the
current candle window, with an optional taker fee.
"""

import random
import time

_BASE_ASSET = {
    'XETHZEUR': 'XETH', 'XXBTZEUR': 'XXBT', 'MANAEUR': 'MANA',
    'GRTEUR': 'GRT', 'LSKEUR': 'LSK', 'SCEUR': 'SC',
}

# Approximate EUR base prices for each pair
BASE_PRICES = {
    'XETHZEUR': 1850.0, 'XXBTZEUR': 37000.0, 'MANAEUR': 0.46,
    'GRTEUR': 0.14, 'LSKEUR': 1.40, 'SCEUR': 0.0028,
}

_DRIFTS = {'up': (0.0003, 0.0015), 'down': (-0.0015, -0.0003), 'flat': (-0.0005, 0.0005)}


def generate_ohlc(base_price, n=200, trend='up', vol=0.004, rng=random):
    """Return n 1-minute candles in Kraken's OHLC layout (strings, like the API)."""
    candles, price = [], base_price
    t0 = int(time.time()) - n * 60
    lo, hi = _DRIFTS[trend]
    for i in range(n):
        open_ = price
        price *= 1 + rng.uniform(lo, hi) + rng.uniform(-vol, vol)
        price = max(price, 1e-8)
        close = price
        high = max(open_, close) * (1 + abs(rng.uniform(0, vol)))
        low = min(open_, close) * (1 - abs(rng.uniform(0, vol)))
        vwap = (high + low + close) / 3
        candles.append([
            t0 + i * 60,
            f'{open_:.8f}', f'{high:.8f}', f'{low:.8f}', f'{close:.8f}',
            f'{vwap:.8f}', f'{rng.uniform(10, 500):.2f}', rng.randint(50, 500),
        ])
    return candles


class FakeKraken:
    """Minimal krakenex.API replacement backed by pre-generated candles."""

    def __init__(self, history, eur=1000.0, window=100, fee=0.0):
        self.history = history          # {pair: [candles]}
        self.window = window            # candles returned per OHLC call
        self.fee = fee                  # taker fee as a fraction (e.g. 0.0026)
        self.cursor = window            # index (exclusive) of the "current" candle
        self.balance = {'ZEUR': eur}
        self.orders = []

    # -- simulation control ------------------------------------------------
    def advance(self, steps=1):
        self.cursor += steps

    def candles(self, pair):
        end = min(self.cursor, len(self.history[pair]))
        return self.history[pair][max(0, end - self.window):end]

    def price(self, pair):
        return float(self.candles(pair)[-1][4])

    def total_value(self):
        total = self.balance.get('ZEUR', 0.0)
        for pair, asset in _BASE_ASSET.items():
            qty = self.balance.get(asset, 0.0)
            if qty and pair in self.history:
                total += qty * self.price(pair)
        return total

    # -- krakenex.API interface -------------------------------------------
    def query_public(self, method, data=None):
        if method != 'OHLC':
            return {'error': [f'EGeneral:Unsupported method {method}'], 'result': {}}
        pair = data['pair']
        return {'error': [], 'result': {pair: self.candles(pair), 'last': self.cursor}}

    def query_private(self, method, data=None):
        if method == 'Balance':
            return {'error': [], 'result': {k: f'{v:.10f}' for k, v in self.balance.items()}}
        if method == 'AddOrder':
            return self._add_order(data)
        return {'error': [f'EGeneral:Unsupported method {method}'], 'result': {}}

    def _add_order(self, data):
        pair, side, volume = data['pair'], data['type'], float(data['volume'])
        asset = _BASE_ASSET[pair]
        price = self.price(pair)
        cost = volume * price
        if side == 'buy':
            total = cost * (1 + self.fee)
            # allow for the 8-decimal rounding of the requested volume
            if total > self.balance['ZEUR'] * (1 + 1e-6) + 1e-9:
                return {'error': ['EOrder:Insufficient funds'], 'result': {}}
            self.balance['ZEUR'] = max(0.0, self.balance['ZEUR'] - total)
            self.balance[asset] = self.balance.get(asset, 0.0) + volume
        else:
            held = self.balance.get(asset, 0.0)
            if volume > held * (1 + 1e-9) + 1e-12:
                return {'error': ['EOrder:Insufficient funds'], 'result': {}}
            volume = min(volume, held)
            self.balance[asset] = held - volume
            self.balance['ZEUR'] += volume * price * (1 - self.fee)
        self.orders.append((self.cursor, pair, side, volume, price))
        return {'error': [], 'result': {'descr': {'order': f'{side} {volume:.8f} {pair} @ market'},
                                        'txid': [f'SIM-{len(self.orders)}']}}
