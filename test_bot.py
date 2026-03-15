import sys
import json
import unittest
from unittest.mock import patch, mock_open, MagicMock

# Mock krakenex before importing bot (not installed in test environment)
sys.modules['krakenex'] = MagicMock()

import bot


class TestGetBalance(unittest.TestCase):
    def test_returns_balance_from_file(self):
        mock_data = {'ZUSD': '500.0', 'XETH': '1.5'}
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_data))):
            result = bot.get_balance()
        self.assertEqual(result, mock_data)

    def test_returns_default_on_invalid_json(self):
        with patch('builtins.open', mock_open(read_data='invalid json')):
            result = bot.get_balance()
        self.assertEqual(result['ZUSD'], '1000.0')
        self.assertIn('EUR.HOLD', result)

    def test_returns_default_on_empty_file(self):
        with patch('builtins.open', mock_open(read_data='')):
            result = bot.get_balance()
        self.assertEqual(result['ZUSD'], '1000.0')


class TestUpdateBalance(unittest.TestCase):
    def test_buy_deducts_funds_and_adds_crypto(self):
        mock_balance = {'ZUSD': '1000.0'}
        with patch('bot.get_balance', return_value=mock_balance), \
             patch('bot.save_balance'):
            result = bot.update_balance(2.0, 'XETHZUSD', 100.0, sold=False)
        self.assertAlmostEqual(float(result['ZUSD']), 800.0)
        self.assertEqual(result['XETH'], '2.0')

    def test_sell_adds_funds_and_removes_crypto(self):
        mock_balance = {'ZUSD': '800.0', 'XETH': '2.0'}
        with patch('bot.get_balance', return_value=mock_balance), \
             patch('bot.save_balance'):
            result = bot.update_balance(2.0, 'XETHZUSD', 100.0, sold=True)
        self.assertAlmostEqual(float(result['ZUSD']), 1000.0)
        self.assertNotIn('XETH', result)

    def test_save_balance_called_after_update(self):
        mock_balance = {'ZUSD': '1000.0'}
        with patch('bot.get_balance', return_value=mock_balance), \
             patch('bot.save_balance') as mock_save:
            bot.update_balance(1.0, 'XXBTZUSD', 200.0, sold=False)
        mock_save.assert_called_once()


class TestSaveBalance(unittest.TestCase):
    def test_writes_json_to_file(self):
        data = {'ZUSD': '500.0'}
        m = mock_open()
        with patch('builtins.open', m):
            bot.save_balance(data)
        m.assert_called_once_with('balance.json', 'w')
        handle = m()
        written = ''.join(call.args[0] for call in handle.write.call_args_list)
        self.assertIn('ZUSD', written)


class TestGetPurchasingPrice(unittest.TestCase):
    def test_returns_last_trade_price(self):
        mock_trades = {
            'XETHZUSD': [
                {'price_usd': 100.0, 'bought': True, 'sold': False, 'amount': 1.0, 'time_stamp': '100'},
                {'price_usd': 150.0, 'bought': True, 'sold': False, 'amount': 1.0, 'time_stamp': '200'},
            ]
        }
        with patch('bot.load_trades', return_value=mock_trades):
            result = bot.get_purchasing_price('XETHZUSD')
        self.assertEqual(result, 150.0)

    def test_returns_first_trade_price_when_single_trade(self):
        mock_trades = {
            'XETHZUSD': [
                {'price_usd': 200.0, 'bought': True, 'sold': False, 'amount': 1.0, 'time_stamp': '100'},
            ]
        }
        with patch('bot.load_trades', return_value=mock_trades):
            result = bot.get_purchasing_price('XETHZUSD')
        self.assertEqual(result, 200.0)


class TestLoadTrades(unittest.TestCase):
    def test_loads_from_file(self):
        mock_data = {'XETHZUSD': [{'price_usd': 100.0}]}
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
    def test_divides_equally_when_no_cryptos_owned(self):
        # balance has ZUSD + EUR.HOLD = 2 non-crypto entries, 0 cryptos owned
        mock_balance = {'ZUSD': '600.0', 'EUR.HOLD': '0.0'}
        with patch('bot.get_balance', return_value=mock_balance):
            result = bot.get_available_funds()
        # cryptos_not_owned = 6 - (2 - 2) = 6; 600 / 6 = 100
        self.assertAlmostEqual(result, 100.0)

    def test_accounts_for_owned_cryptos(self):
        mock_balance = {'ZUSD': '400.0', 'EUR.HOLD': '0.0', 'XETH': '1.0'}
        with patch('bot.get_balance', return_value=mock_balance):
            result = bot.get_available_funds()
        # cryptos_not_owned = 6 - (3 - 2) = 5; 400 / 5 = 80
        self.assertAlmostEqual(result, 80.0)


class TestGetPairs(unittest.TestCase):
    def test_returns_tuple_of_six_pairs(self):
        pairs = bot.get_pairs()
        self.assertEqual(len(pairs), 6)

    def test_contains_expected_pairs(self):
        pairs = bot.get_pairs()
        self.assertIn('XETHZUSD', pairs)
        self.assertIn('XXBTZUSD', pairs)


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
    def _make_data(self, close_prices, current_price):
        # Index [4] = close price (FIX 1: was [3] low price)
        prices = [[None, None, None, current_price * 0.99, current_price]] * 15
        return {'close': close_prices, 'prices': prices}

    def test_no_buy_signal_on_flat_trend(self):
        data = self._make_data([100.0] * 15, 100.0)
        result = bot.check_opportunity(data, 'XETHZUSD', sell=False, buy=True)
        self.assertFalse(result)

    def test_no_buy_signal_on_downtrend(self):
        close_prices = [100.0 - i for i in range(15)]
        data = self._make_data(close_prices, 90.0)
        result = bot.check_opportunity(data, 'XETHZUSD', sell=False, buy=True)
        self.assertFalse(result)

    def test_buy_signal_on_uptrend_with_price_near_ma(self):
        # MA values trending up AND current price close to MA (within ~5%) → buy
        close_prices = [95.0 + i * 0.5 for i in range(15)]  # MA ~102
        data = self._make_data(close_prices, 103.0)  # price within 5% of MA
        result = bot.check_opportunity(data, 'XETHZUSD', sell=False, buy=True)
        self.assertTrue(result)

    def test_no_buy_signal_on_uptrend_price_overbought(self):
        # MA trending up but current price is >5% above MA → overbought, don't buy
        close_prices = [95.0 + i * 0.5 for i in range(15)]  # MA ~102
        data = self._make_data(close_prices, 115.0)  # price >12% above MA
        result = bot.check_opportunity(data, 'XETHZUSD', sell=False, buy=True)
        self.assertFalse(result)

    def test_sell_triggers_above_3_percent_profit(self):
        # take-profit at +3%
        data = self._make_data([100.0] * 15, 104.0)
        with patch('bot.get_purchasing_price', return_value=100.0):
            result = bot.check_opportunity(data, 'XETHZUSD', sell=True, buy=False)
        self.assertTrue(result)

    def test_sell_does_not_trigger_below_3_percent_profit(self):
        # price only 2% above purchase — below the 3% take-profit threshold
        data = self._make_data([100.0] * 15, 102.0)
        with patch('bot.get_purchasing_price', return_value=100.0):
            result = bot.check_opportunity(data, 'XETHZUSD', sell=True, buy=False)
        self.assertIsNone(result)

    def test_sell_triggers_at_1_percent_loss(self):
        # stop-loss at -1%
        data = self._make_data([100.0] * 15, 98.5)
        with patch('bot.get_purchasing_price', return_value=100.0):
            result = bot.check_opportunity(data, 'XETHZUSD', sell=True, buy=False)
        self.assertTrue(result)

    def test_sell_does_not_trigger_within_1_percent_loss(self):
        # loss of only 0.5% — within the 1% stop-loss buffer
        data = self._make_data([100.0] * 15, 99.6)
        with patch('bot.get_purchasing_price', return_value=100.0):
            result = bot.check_opportunity(data, 'XETHZUSD', sell=True, buy=False)
        self.assertIsNone(result)

    def test_trailing_stop_triggers_on_drop_from_peak_ma(self):
        # Peak MA was 110, current price dropped to 107 (< 110 * 0.98 = 107.8)
        close_prices = [105.0 + i for i in range(6)] + [110.0] * 9  # peak MA = 110
        data = self._make_data(close_prices, 107.0)
        with patch('bot.get_purchasing_price', return_value=100.0):
            result = bot.check_opportunity(data, 'XETHZUSD', sell=True, buy=False)
        self.assertTrue(result)


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
