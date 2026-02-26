# Reports

This directory contains all generated reports from strategy experiments and backtests.

## Structure

All experiment outputs are saved here:
- **Strategy charts** - Interactive HTML visualizations with signals
- **Backtest reports** - Performance metrics and statistics

## Files

### Chart Files
Format: `{strategy_name}_{symbol}_{params}.html`

Example: `sma_crossover_EURUSD_20_50.html`

### Backtest Reports
Format: `{strategy_name}_backtest_{timestamp}.txt`

Example: `sma_crossover_backtest_20260226_120530.txt`

## Git

Report files are gitignored by default to keep the repository clean.
