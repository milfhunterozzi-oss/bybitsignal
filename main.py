import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BYBIT_BASE_URL = "https://api.bybit.com"
INSTRUMENTS_URL = f"{BYBIT_BASE_URL}/v5/market/instruments-info"
TICKERS_URL = f"{BYBIT_BASE_URL}/v5/market/tickers"
KLINES_URL = f"{BYBIT_BASE_URL}/v5/market/kline"
DAY_MS = 24 * 60 * 60 * 1000
STATE_FILE = Path(__file__).parent / "signals_state.json"


@dataclass(frozen=True)
class Config:
    universe_size: int = int(os.environ.get("UNIVERSE_SIZE", "50"))
    long_window_days: int = int(os.environ.get("LONG_WINDOW_DAYS", "30"))
    short_window_days: int = int(os.environ.get("SHORT_WINDOW_DAYS", "7"))
    request_pause_seconds: float = float(
        os.environ.get("REQUEST_PAUSE_SECONDS", "0.30")
    )
    run_mode: str = os.environ.get("RUN_MODE", "daily").lower()
    rebalance_hour_utc: int = int(os.environ.get("REBALANCE_HOUR_UTC", "0"))
    telegram_enabled: bool = (
        os.environ.get("TELEGRAM_ENABLED", "true").lower() == "true"
    )
    min_turnover_usd: float = float(
        os.environ.get("MIN_TURNOVER_USD", "10000000")
    )
    min_listing_days: int = int(os.environ.get("MIN_LISTING_DAYS", "30"))
    universe_candidate_multiplier: int = int(
        os.environ.get("UNIVERSE_CANDIDATE_MULTIPLIER", "3")
    )


SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
)
SESSION.proxies.update({
    "http": "http://8.219.229.53:5060",
    "https": "http://8.219.229.53:5060",
})

def retry_on_network_error(max_attempts: int = 5, base_delay: float = 2.0, max_delay: float = 30.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.ConnectionError, requests.Timeout) as exc:
                    if attempt == max_attempts:
                        raise
                    log.warning(
                        "Network error on attempt %d/%d: %s — retrying in %.1fs",
                        attempt, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    if status == 429:
                        retry_after = float(
                            exc.response.headers.get("Retry-After", delay)
                        )
                        log.warning(
                            "Rate limited (429) on attempt %d/%d — sleeping %.1fs",
                            attempt, max_attempts, retry_after,
                        )
                        time.sleep(retry_after)
                        delay = min(delay * 2, max_delay)
                    elif status is not None and status >= 500:
                        if attempt == max_attempts:
                            raise
                        log.warning(
                            "Server error %s on attempt %d/%d — retrying in %.1fs",
                            status, attempt, max_attempts, delay,
                        )
                        time.sleep(delay)
                        delay = min(delay * 2, max_delay)
                    else:
                        raise
            return None
        return wrapper
    return decorator


@retry_on_network_error()
def request_json(url: str, params: dict | None = None, timeout: int = 20):
    response = SESSION.get(url, params=params, timeout=timeout)
    if response.status_code in {403, 451}:
        raise RuntimeError(
            "Bybit blocked this IP/location for the requested endpoint. "
            "Прокси в код не добавлен, чтобы не рисковать блокировкой проекта."
        )
    response.raise_for_status()
    data = response.json()
    if data.get("retCode") != 0:
        raise RuntimeError(
            f"Bybit API error {data.get('retCode')}: {data.get('retMsg')}"
        )
    return data


def short_name(symbol: str) -> str:
    """BTCUSDT → BTC"""
    return symbol.removesuffix("USDT")


def get_trading_usdt_perpetuals(min_listing_days: int) -> dict[str, datetime]:
    """Return {symbol: listing_datetime} for active USDT LinearPerpetual contracts
    that are at least min_listing_days old."""
    symbols: dict[str, datetime] = {}
    cursor = None
    cutoff = datetime.now(timezone.utc) - timedelta(days=min_listing_days)

    while True:
        params = {"category": "linear", "quoteCoin": "USDT", "limit": 1000}
        if cursor:
            params["cursor"] = cursor

        data = request_json(INSTRUMENTS_URL, params=params)
        result = data.get("result", {})

        for item in result.get("list", []):
            if (
                item.get("contractType") == "LinearPerpetual"
                and item.get("quoteCoin") == "USDT"
                and item.get("status") == "Trading"
            ):
                launch_ms = item.get("launchTime")
                if launch_ms:
                    try:
                        launch_dt = datetime.fromtimestamp(
                            int(launch_ms) / 1000, tz=timezone.utc
                        )
                        if launch_dt <= cutoff:
                            symbols[item["symbol"]] = launch_dt
                    except (TypeError, ValueError):
                        pass

        cursor = result.get("nextPageCursor")
        if not cursor:
            break

    log.info(
        "Tradable symbols with >= %d days listing history: %d",
        min_listing_days, len(symbols),
    )
    return symbols


def get_top_liquid_symbols(
    limit: int, config: Config
) -> tuple[list[str], dict[str, float]]:
    """Return (candidate_list, {symbol: price_change_pct_24h})."""
    tradable = get_trading_usdt_perpetuals(config.min_listing_days)

    data = request_json(TICKERS_URL, params={"category": "linear"})
    tickers = data.get("result", {}).get("list", [])

    rows: list[tuple[str, float]] = []
    changes: dict[str, float] = {}

    for item in tickers:
        symbol = item.get("symbol")
        if symbol not in tradable:
            continue
        try:
            quote_volume = float(item.get("turnover24h", 0))
        except (TypeError, ValueError):
            continue
        if quote_volume < config.min_turnover_usd:
            continue

        try:
            change_pct = float(item.get("price24hPcnt", 0)) * 100
        except (TypeError, ValueError):
            change_pct = 0.0

        changes[symbol] = change_pct
        rows.append((symbol, quote_volume))

    rows.sort(key=lambda row: row[1], reverse=True)
    candidates = [s for s, _ in rows[: limit * config.universe_candidate_multiplier]]

    log.info(
        "Candidate pool (top %d by 24h turnover, min %.0fM USDT): %d symbols",
        limit * config.universe_candidate_multiplier,
        config.min_turnover_usd / 1e6,
        len(candidates),
    )
    return candidates, changes


def fetch_closed_daily_klines(symbol: str, required_days: int) -> pd.DataFrame:
    data = request_json(
        KLINES_URL,
        params={
            "category": "linear",
            "symbol": symbol,
            "interval": "D",
            "limit": required_days + 5,
        },
    )
    raw = data.get("result", {}).get("list", [])
    if not raw:
        return pd.DataFrame()

    frame = pd.DataFrame(
        raw,
        columns=["start_time", "open", "high", "low", "close", "volume", "turnover"],
    )
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    frame["start_time"] = pd.to_numeric(frame["start_time"], errors="coerce")
    frame["quote_volume"] = pd.to_numeric(frame["turnover"], errors="coerce")
    frame = frame.loc[
        frame["start_time"] + DAY_MS <= now_ms,
        ["start_time", "quote_volume"],
    ].dropna()
    frame = frame.sort_values("start_time")
    return frame.tail(required_days)


def calculate_ema_volume_ratio(symbol: str, config: Config) -> tuple[float, float] | None:
    """Return (ema_ratio, avg_7d_turnover) or None if insufficient history."""
    frame = fetch_closed_daily_klines(symbol, config.long_window_days)
    if len(frame) < config.long_window_days:
        return None

    vol = frame["quote_volume"]

    ema_short = float(
        vol.ewm(span=config.short_window_days, adjust=False).mean().iloc[-1]
    )
    ema_long = float(
        vol.ewm(span=config.long_window_days, adjust=False).mean().iloc[-1]
    )

    if ema_long <= 0:
        return None

    avg_7d = float(vol.tail(config.short_window_days).mean())
    return float(ema_short / ema_long), avg_7d


def select_final_universe(
    kline_data: dict[str, tuple[float, float]],
    universe_size: int,
) -> list[str]:
    """Re-rank by 7d avg turnover from klines and return top universe_size."""
    ranked = sorted(
        kline_data.keys(),
        key=lambda s: kline_data[s][1],
        reverse=True,
    )
    selected = ranked[:universe_size]
    log.info("Final universe (%d symbols) selected by 7d avg kline turnover.", len(selected))
    return selected


def build_portfolio(ratios: dict[str, float]) -> tuple[pd.Series, pd.Series, pd.Series]:
    ranking = pd.Series(ratios, name="volume_ratio").sort_values(ascending=False)
    midpoint = len(ranking) // 2
    if midpoint == 0:
        raise RuntimeError("Not enough valid symbols to build a long/short portfolio")
    longs = ranking.iloc[:midpoint]
    shorts = ranking.iloc[midpoint:].sort_values(ascending=True)
    return ranking, longs, shorts


def load_previous_state() -> dict:
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            log.warning("Could not load previous state: %s", exc)
    return {}


def save_current_state(longs: pd.Series, shorts: pd.Series) -> None:
    state = {
        "longs": list(longs.index),
        "shorts": list(shorts.index),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with STATE_FILE.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        log.warning("Could not save state: %s", exc)


def build_diff_section(
    current_set: set[str],
    previous_set: set[str],
    side: str,
) -> str:
    added = current_set - previous_set
    removed = previous_set - current_set
    if not added and not removed:
        return ""
    lines = [f"\n<i>Изменения {side}:</i>"]
    for s in sorted(added):
        lines.append(f"  ➕ {short_name(s)}")
    for s in sorted(removed):
        lines.append(f"  ➖ {short_name(s)}")
    return "\n".join(lines)


def format_symbol_list(series: pd.Series) -> str:
    return "\n".join(
        f"• {short_name(symbol)}: R {value:.2f}" for symbol, value in series.items()
    )


def format_change(pct: float) -> str:
    if pct >= 0:
        return f"↑{pct:.1f}%"
    return f"↓{abs(pct):.1f}%"


def format_position_list(
    series: pd.Series,
    changes: dict[str, float],
    marker: str,
) -> str:
    lines = []
    for symbol, value in series.items():
        change_str = format_change(changes.get(symbol, 0.0))
        lines.append(
            f"{marker} <b>{short_name(symbol)}</b> — R {value:.2f} — {change_str}"
        )
    return "\n".join(lines)


def build_signal_message(
    universe: list[str],
    ratios: dict[str, float],
    changes: dict[str, float],
    config: Config,
    previous_state: dict,
) -> str:
    ranking, longs, shorts = build_portfolio(ratios)

    log.info("TOP LONG:\n%s", format_symbol_list(longs))
    log.info("TOP SHORT:\n%s", format_symbol_list(shorts))

    save_current_state(longs, shorts)

    prev_longs = set(previous_state.get("longs", []))
    prev_shorts = set(previous_state.get("shorts", []))

    long_diff = build_diff_section(set(longs.index), prev_longs, "LONG") if prev_longs else ""
    short_diff = build_diff_section(set(shorts.index), prev_shorts, "SHORT") if prev_shorts else ""

    return (
        f"📅 <b>Bybit Futures Volume Ranking</b>\n\n"
        f"📈 <b>LONG</b>\n"
        f"{format_position_list(longs, changes, '🟢')}{long_diff}\n\n"
        f"📉 <b>SHORT</b>\n"
        f"{format_position_list(shorts, changes, '🔴')}{short_diff}"
    )


def split_telegram_message(message: str, limit: int = 3900) -> list[str]:
    chunks: list[str] = []
    current = ""
    for line in message.splitlines(keepends=True):
        if len(current) + len(line) > limit:
            if current:
                chunks.append(current.rstrip())
                current = ""
            if len(line) > limit:
                chunks.append(line[:limit].rstrip())
                current = line[limit:]
            else:
                current = line
        else:
            current += line
    if current:
        chunks.append(current.rstrip())
    return chunks


def _send_telegram_blocking(message: str, config: Config) -> bool:
    if not config.telegram_enabled:
        log.info("Telegram is disabled by TELEGRAM_ENABLED=false")
        log.info("\n%s", message)
        return False

    token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    if not token or not chat_id:
        log.warning(
            "Telegram message was not sent: TG_BOT_TOKEN and TG_CHAT_ID are not configured."
        )
        log.info("\n%s", message)
        return False

    chunks = split_telegram_message(message)
    for index, chunk in enumerate(chunks, start=1):
        text = chunk
        if len(chunks) > 1:
            text = f"{chunk}\n\nЧасть {index}/{len(chunks)}"

        try:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=20,
            )
            if response.status_code != 200:
                log.error("Telegram error: %s %s", response.status_code, response.text)
                return False
        except Exception as exc:
            log.error("Telegram send failed: %s", exc)
            return False

    log.info("Signal sent to Telegram")
    return True


def send_telegram(message: str, config: Config) -> None:
    thread = threading.Thread(
        target=_send_telegram_blocking,
        args=(message, config),
        daemon=True,
        name="telegram-sender",
    )
    thread.start()
    thread.join(timeout=60)
    if thread.is_alive():
        log.warning("Telegram send timed out after 60s")


def run_strategy(config: Config) -> None:
    log.info(
        "Starting Volume Ranking Strategy — universe=%d, EMA%d/EMA%d, mode=%s",
        config.universe_size,
        config.short_window_days,
        config.long_window_days,
        config.run_mode,
    )

    previous_state = load_previous_state()

    candidates, changes = get_top_liquid_symbols(config.universe_size, config)
    if not candidates:
        raise RuntimeError("Could not build Bybit USDT perpetual candidate pool")

    log.info("Fetching klines for %d candidate symbols…", len(candidates))

    kline_data: dict[str, tuple[float, float]] = {}
    total = len(candidates)

    for index, symbol in enumerate(candidates, start=1):
        try:
            metrics = calculate_ema_volume_ratio(symbol, config)
        except Exception as exc:
            log.warning("[%d/%d] %s: skipped (%s)", index, total, symbol, exc)
            metrics = None

        if metrics is None:
            log.debug("[%d/%d] %s: skipped, not enough clean history", index, total, symbol)
        else:
            ema_ratio, avg_7d = metrics
            kline_data[symbol] = (ema_ratio, avg_7d)
            log.info(
                "[%d/%d] %s: R=%.3f  7d_avg_vol=$%.0fK",
                index, total, symbol, ema_ratio, avg_7d / 1000,
            )

        time.sleep(config.request_pause_seconds)

    if len(kline_data) < 2:
        raise RuntimeError("Not enough usable data to generate signals")

    universe = select_final_universe(kline_data, config.universe_size)
    ratios = {s: kline_data[s][0] for s in universe if s in kline_data}

    if len(ratios) < 2:
        raise RuntimeError("Not enough symbols in final universe to generate signals")

    message = build_signal_message(universe, ratios, changes, config, previous_state)
    send_telegram(message, config)


def seconds_until_next_rebalance(hour_utc: int) -> int:
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=hour_utc, minute=0, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(days=1)
    return max(60, int((next_run - now).total_seconds()))


def main() -> None:
    config = Config()
    if config.short_window_days >= config.long_window_days:
        raise RuntimeError("SHORT_WINDOW_DAYS must be smaller than LONG_WINDOW_DAYS")

    if config.run_mode == "daily":
        while True:
            try:
                run_strategy(config)
            except Exception as exc:
                error_message = f"⚠️ <b>Bybit strategy error</b>\n{exc}"
                log.error(error_message)
                send_telegram(error_message, config)
            sleep_seconds = seconds_until_next_rebalance(config.rebalance_hour_utc)
            next_run = datetime.now(timezone.utc) + timedelta(seconds=sleep_seconds)
            log.info("Next rebalance: %s UTC", next_run.strftime("%d.%m.%Y %H:%M"))
            time.sleep(sleep_seconds)
    else:
        try:
            run_strategy(config)
        except Exception as exc:
            error_message = f"⚠️ <b>Bybit strategy error</b>\n{exc}"
            log.error(error_message)
            send_telegram(error_message, config)


if __name__ == "__main__":
    main()
