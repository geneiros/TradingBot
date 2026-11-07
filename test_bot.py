import sys
import json
import unittest
from unittest.mock import patch, mock_open, MagicMock

# Mock krakenex before importing bot (not installed in test environment)
sys.modules['krakenex'] = MagicMock()

import bot


class TestGetBalance(unittest.TestCase):
    def setUp(self):
        bot.k = MagicMock()

    def test_returns_balance_from_kraken(self):
        bot.k.query_private.return_value = {
            'error': [],
            'result': {'ZUSD': '500.0', 'XETH': '1.5'}
        }
        result = bot.get_balance()
        self.assertEqual(result, {'ZUSD': '500.0', 'XETH': '1.5'})
        bot.k.query_private.assert_called_once_with('Balance')

    def test_returns_empty_on_api_error(self):
        bot.k.query_private.return_value = {'error': ['EGeneral:Invalid key']}
        result = bot.get_balance()
        self.assertEqual(result, {})

    def test_returns_empty_on_exception(self):
        bot.k.query_private.side_effect = Exception('network error')
        result = bot.get_balance()
        self.assertEqual(result, {})


class TestGetPurchasingPrice(unittest.TestCase):
    def test_returns_last_trade_price(self):
        mock_trades = {
            'XETHZEUR': [
                {'price_usd': 100.0, 'bought': True, 'sold': False, 'amount': 1.0, 'time_stamp': '100'},
                {'price_usd': 150.0, 'bought': True, 'sold': False, 'amount': 1.0, 'time_stamp': '200'},
            ]
        }
        with patch('bot.load_trades', return_value=mock_trades):
            result = bot.get_purchasing_price('XETHZEUR')
        self.assertEqual(result, 150.0)

    def test_returns_first_trade_price_when_single_trade(self):
        mock_trades = {
            'XETHZEUR': [
                {'price_usd': 200.0, 'bought': True, 'sold': False, 'amount': 1.0, 'time_stamp': '100'},
            ]
        }
        with patch('bot.load_trades', return_value=mock_trades):
            result = bot.get_purchasing_price('XETHZEUR')
        self.assertEqual(result, 200.0)


class TestLoadTrades(unittest.TestCase):
    def test_loads_from_file(self):
        mock_data = {'XETHZEUR': [{'price_usd': 100.0}]}
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_data))):
            result = bot.load_trades()
        self.assertEqual(result, mock_data)

    def test_initializes_empty_on_invalid_json(self):
        with patch('builtins.open', mock_open(read_data='bad json')):
            result = bot.load_trades()
        for pair in bot.get_pairs():
            self.assertIn(pair, result)
            self.assertEqual(result[pair], [])


class TestDeleteEntries(unittest.TestCase):
    def test_keeps_last_10_entries(self):
        data = {'close': list(range(20))}
        result = bot.delete_entries(data, 'close')
        self.assertEqual(result, list(range(10, 20)))

    def test_short_list_kept_intact(self):
        data = {'close': [1, 2, 3]}
        result = bot.delete_entries(data, 'close')
        self.assertEqual(result, [1, 2, 3])

    def test_exactly_10_entries_unchanged(self):
        data = {'close': list(range(10))}
        result = bot.delete_entries(data, 'close')
        self.assertEqual(result, list(range(10)))


class TestGetAvailableFunds(unittest.TestCase):
    def _make_trades(self, held_pairs):
        """Build a trades dict where held_pairs are currently bought (not sold)."""
        trades = {p: [] for p in bot.get_pairs()}
        for p in held_pairs:
            trades[p] = [{'bought': True, 'sold': False, 'amount': 1.0,
                          'price_usd': 100.0, 'time_stamp': '0'}]
        return trades

    def test_divides_equally_when_no_cryptos_owned(self):
        with patch('bot.get_balance', return_value={'ZEUR': '600.0'}), \
             patch('bot.load_trades', return_value=self._make_trades([])):
            result = bot.get_available_funds()
        # 6 slots free → 600 / 6 = 100
        self.assertAlmostEqual(result, 100.0)

    def test_accounts_for_owned_cryptos(self):
        with patch('bot.get_balance', return_value={'ZEUR': '400.0'}), \
             patch('bot.load_trades', return_value=self._make_trades(['XETHZEUR'])):
            result = bot.get_available_funds()
        # 1 held → 5 slots free → 400 / 5 = 80
        self.assertAlmostEqual(result, 80.0)

    def test_returns_zero_when_all_slots_full(self):
        all_pairs = list(bot.get_pairs())
        with patch('bot.get_balance', return_value={'ZEUR': '0.0'}), \
             patch('bot.load_trades', return_value=self._make_trades(all_pairs)):
            result = bot.get_available_funds()
        self.assertEqual(result, 0.0)


class TestGetPairs(unittest.TestCase):
    def test_returns_tuple_of_six_pairs(self):
        pairs = bot.get_pairs()
        self.assertEqual(len(pairs), 6)

    def test_contains_expected_pairs(self):
        pairs = bot.get_pairs()
        self.assertIn('XETHZEUR', pairs)
        self.assertIn('XXBTZEUR', pairs)


class TestMakeCryptoData(unittest.TestCase):
    def test_creates_structure_for_all_pairs(self):
        result = bot.make_crypto_data({})
        for pair in bot.get_pairs():
            self.assertIn(pair, result)
            self.assertIn('high', result[pair])
            self.assertIn('low', result[pair])
            self.assertIn('close', result[pair])
            self.assertIn('prices', result[pair])

    def test_initializes_empty_lists(self):
        result = bot.make_crypto_data({})
        for pair in bot.get_pairs():
            self.assertIsInstance(result[pair]['close'], list)
            self.assertEqual(len(result[pair]['close']), 0)


class TestCheckOpportunity(unittest.TestCase):
    """check_opportunity(data, name, crypto_data, sell, buy).

    crypto_data candles follow Kraken's OHLC layout:
    [time, open, high, low, close, ...].
    """

    @staticmethod
    def _candles(price, open_price=None):
        if open_price is None:
            open_price = price
        return [[0, open_price, price, price, price]] * 3

    @staticmethod
    def _data(close_mas, peak_price=0.0):
        return {'close': close_mas, 'peak_price': peak_price}

    # ── Buy signal ──────────────────────────────────────────────────────────
    def _uptrend(self):
        return [95.0 + i * 0.5 for i in range(15)]  # last MAs ~ 101..102

    def test_no_buy_signal_on_flat_trend(self):
        result = bot.check_opportunity(self._data([100.0] * 15), 'XETHZEUR',
                                       self._candles(100.0), sell=False, buy=True)
        self.assertFalse(result)

    def test_no_buy_signal_on_downtrend(self):
        close_mas = [100.0 - i for i in range(15)]
        result = bot.check_opportunity(self._data(close_mas), 'XETHZEUR',
                                       self._candles(85.0), sell=False, buy=True)
        self.assertFalse(result)

    def test_buy_signal_on_uptrend_with_price_near_ma(self):
        result = bot.check_opportunity(self._data(self._uptrend()), 'XETHZEUR',
                                       self._candles(103.0, open_price=102.5),
                                       sell=False, buy=True)
        self.assertTrue(result)

    def test_no_buy_signal_on_uptrend_price_overbought(self):
        # price > ~3% above the MA → stretched, don't buy
        result = bot.check_opportunity(self._data(self._uptrend()), 'XETHZEUR',
                                       self._candles(110.0, open_price=109.0),
                                       sell=False, buy=True)
        self.assertFalse(result)

    def test_no_buy_signal_on_bearish_candle(self):
        # all conditions met but current candle closes below its open
        result = bot.check_opportunity(self._data(self._uptrend()), 'XETHZEUR',
                                       self._candles(103.0, open_price=103.5),
                                       sell=False, buy=True)
        self.assertFalse(result)

    def test_no_buy_signal_with_too_few_consecutive_rises(self):
        close_mas = [100.0] * 11 + [100.5, 101.0, 101.5, 102.0]  # only 4 rises
        result = bot.check_opportunity(self._data(close_mas), 'XETHZEUR',
                                       self._candles(102.0, open_price=101.9),
                                       sell=False, buy=True)
        self.assertFalse(result)

    # ── Sell signal ─────────────────────────────────────────────────────────
    def _sell(self, price, peak_price=0.0, buy_price=100.0):
        with patch('bot.get_purchasing_price', return_value=buy_price):
            return bot.check_opportunity(self._data([100.0] * 15, peak_price),
                                         'XETHZEUR', self._candles(price),
                                         sell=True, buy=False)

    def test_take_profit_at_3_percent(self):
        self.assertTrue(self._sell(103.0, peak_price=103.0))

    def test_holds_below_take_profit(self):
        self.assertFalse(self._sell(101.4, peak_price=101.4))

    def test_stop_loss_at_1_percent_loss(self):
        self.assertTrue(self._sell(98.9, peak_price=100.0))

    def test_holds_within_1_percent_loss(self):
        self.assertFalse(self._sell(99.6, peak_price=100.0))

    def test_break_even_stop_after_peak_reached_1_percent(self):
        # peak hit +1.2%, price fell back below entry → sell at break-even
        self.assertTrue(self._sell(99.9, peak_price=101.2))

    def test_break_even_holds_while_above_entry(self):
        self.assertFalse(self._sell(100.3, peak_price=101.2))

    def test_break_even_replaces_minus_1_percent_stop(self):
        # once break-even is armed, a -1% price also triggers the sale
        self.assertTrue(self._sell(98.5, peak_price=101.2))

    def test_trailing_stop_triggers_on_drop_from_peak(self):
        # peak +2% (> +1.5%), price 1.1% below peak
        self.assertTrue(self._sell(100.9, peak_price=102.0))

    def test_trailing_stop_holds_on_small_pullback(self):
        self.assertFalse(self._sell(101.5, peak_price=102.0))


class TestCheckData(unittest.TestCase):
    def _candle(self, i, price):
        return [i, str(price), str(price), str(price), str(price)]

    def test_moving_average_uses_actual_candle_count(self):
        mva = bot.make_crypto_data({})
        candles = [self._candle(i, 10.0 + i) for i in range(10)]  # mean close = 14.5
        with patch('bot.save_crypto_data'), patch('bot.try_buy'):
            bot.check_data('XETHZEUR', candles, True, mva)
        self.assertAlmostEqual(mva['XETHZEUR']['close'][-1], 14.5)

    def test_stored_candles_are_capped(self):
        mva = bot.make_crypto_data({})
        with patch('bot.save_crypto_data'), patch('bot.try_buy'):
            for start in range(0, 500, 50):
                candles = [self._candle(i, 1.0) for i in range(start, start + 100)]
                bot.check_data('XETHZEUR', candles, True, mva)
        self.assertLessEqual(len(mva['XETHZEUR']['prices']), bot._MAX_MA_HISTORY)

    def test_empty_ohlc_is_skipped(self):
        mva = bot.make_crypto_data({})
        with patch('bot.save_crypto_data') as save, patch('bot.try_buy') as buy:
            bot.check_data('XETHZEUR', [], True, mva)
        save.assert_not_called()
        buy.assert_not_called()


class TestNow(unittest.TestCase):
    def test_returns_decimal(self):
        import decimal
        result = bot.now()
        self.assertIsInstance(result, decimal.Decimal)

    def test_returns_positive_timestamp(self):
        result = bot.now()
        self.assertGreater(result, 0)


if __name__ == '__main__':
    unittest.main()
