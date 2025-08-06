# Jesse Testing Guide

## Overview
Jesse's testing framework contains 442 tests across 27 test files. These tests ensure the reliability and correctness of Jesse's algorithmic trading functionality.

## Test Categories
- **backtest**: 2 tests
- **broker**: 4 tests
- **candle_service**: 6 tests
- **completed_trade**: 4 tests
- **conflicting_orders**: 3 tests
- **dynamic_numpy_array**: 7 tests
- **exchange**: 3 tests
- **helpers**: 79 tests
- **import_candles**: 3 tests
- **indicators**: 178 tests
- **isolated_backtest**: 7 tests
- **math_utils**: 1 tests
- **metrics**: 3 tests
- **order**: 4 tests
- **parent_strategy**: 76 tests
- **position**: 17 tests
- **research**: 1 tests
- **router**: 1 tests
- **spot_mode**: 13 tests
- **state_candle**: 6 tests
- **state_exchanges**: 1 tests
- **state_orderbook**: 2 tests
- **state_orders**: 4 tests
- **state_ticker**: 2 tests
- **state_trades**: 1 tests
- **utils**: 14 tests

## Running Tests
To run all tests:
```
pytest
```

To run a specific test file:
```
pytest tests/test_file.py
```

To run a specific test:
```
pytest tests/test_file.py::test_function
```

## Common Testing Patterns
1. **Candle testing**: Tests for candle data manipulation and analysis
2. **Strategy testing**: Tests for trading strategies and execution
3. **Backtest testing**: Tests for backtesting functionality
4. **Indicator testing**: Tests for technical indicators

## Test Structure
Most Jesse tests follow this pattern:
1. Set up test data (often using candles)
2. Execute the functionality being tested
3. Assert the expected outcomes

## Best Practices
1. Always write tests for new functionality
2. Ensure tests are isolated and don't depend on each other
3. Use Jesse's testing utilities when possible
4. Verify both normal operation and edge cases

## Test Data
Test data is often generated using Jesse's factory functions:
- `candles_from_close_prices()`
- `range_candles()`

## Resources
- [Jesse Documentation](https://docs.jesse.trade)
- [Jesse Github Repository](https://github.com/jesse-ai/jesse)
