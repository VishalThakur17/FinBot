import os
import datetime as dt
from typing import List, Dict, Any
import io
import base64
import math  

from flask import Flask, request, jsonify, send_from_directory
from dateutil.relativedelta import relativedelta
from openai import OpenAI
import yfinance as yf
import pandas as pd
# IMPORTANT: use a non-GUI backend for matplotlib
import matplotlib
matplotlib.use("Agg")    
import matplotlib.pyplot as plt

from crypto_utils import CryptoUtils
from metals_utils import MetalsUtils 
from bonds_utils import BondsUtils

try:
    import PyPDF2  # optional but recommended for PDF transcripts
except ImportError:
    PyPDF2 = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""
)

openai_client = OpenAI()  # picks up OPENAI_API_KEY from env


# ---------------------------------------------------------------------------
# Helper: resolve company name -> ticker using OpenAI
# ---------------------------------------------------------------------------

def resolve_symbol_from_company(company: str) -> str:
    """
    Use a small OpenAI model to map a company name like "Apple" to a stock ticker like "AAPL".
    If the user already typed a probable ticker (short, all caps, no spaces), we just return it.
    If we can't confidently map it, we return the uppercased input.
    """
    raw = company.strip()
    if not raw:
        return ""

    # If it already looks like a ticker (e.g., AAPL, MSFT), just normalize & return
    if raw.isalpha() and " " not in raw and 1 <= len(raw) <= 5:
        return raw.upper()

    system_prompt = (
        "You convert company names to their primary stock ticker symbol on major exchanges. "
        "Examples: Apple -> AAPL, Microsoft -> MSFT, Tesla -> TSLA. "
        "If the input already looks like a ticker symbol, just return it. "
        "If you don't know, return the original input unchanged. "
        "Return ONLY the ticker string, with no explanation."
    )
    user_prompt = f"Company name or ticker: {company}\nReturn only the ticker symbol."

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        symbol = resp.choices[0].message.content.strip().upper()
        symbol = symbol.split()[0]
        return symbol
    except Exception as e:
        print(f"[resolve_symbol_from_company] Error: {e}")
        return raw.upper()


# ---------------------------------------------------------------------------
# Helper: historical prices from Yahoo Finance for forecast feature
# ---------------------------------------------------------------------------

def get_historical_prices(symbol: str) -> List[Dict[str, Any]]:
    """
    Fetch last ~12 months of DAILY closing prices for given symbol
    using Yahoo Finance via yfinance.
    Returns: list of { "date": "YYYY-MM-DD", "close": float }, sorted by date.
    """
    symbol = symbol.strip().upper()
    data = yf.download(symbol, period="1y", interval="1d", progress=False)

    if data is None or data.empty:
        raise RuntimeError(f"No historical data found for symbol '{symbol}' from Yahoo Finance.")

    if "Close" not in data.columns:
        raise RuntimeError(f"No 'Close' prices available for symbol '{symbol}'.")

    data = data[["Close"]].dropna()
    if data.empty:
        raise RuntimeError(f"No usable closing price data for symbol '{symbol}' from Yahoo Finance.")

    points: List[Dict[str, Any]] = []
    for idx, row in data.iterrows():
        close_value = float(row["Close"])
        date_str = idx.date().isoformat()
        points.append({"date": date_str, "close": close_value})

    points.sort(key=lambda x: x["date"])
    return points


# ---------------------------------------------------------------------------
# Helper: scenario-based forecast using OpenAI (forecast feature)
# ---------------------------------------------------------------------------

def call_openai_scenario_forecast(
    symbol: str,
    company: str,
    scenario: str,
    historical: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ask ChatGPT to:
      - Generate 3 monthly forecast prices (next 3 months)
      - Provide a short explanation
    """
    if not historical:
        raise ValueError("No historical data passed to OpenAI forecast.")

    closes = [p["close"] for p in historical]
    dates = [p["date"] for p in historical]

    last_price = closes[-1]
    min_price = min(closes)
    max_price = max(closes)
    first_price = closes[0]
    trend_pct = ((last_price - first_price) / first_price) * 100 if first_price != 0 else 0

    trend_description = "roughly flat"
    if trend_pct > 8:
        trend_description = "upward"
    elif trend_pct < -8:
        trend_description = "downward"

    system_prompt = (
        "You are a cautious financial analysis assistant. "
        "Given REAL recent stock data (from Yahoo Finance) and a hypothetical scenario, you will "
        "produce a plausible but clearly non-guaranteed 3-month price forecast "
        "and a short narrative explanation.\n\n"
        "Important:\n"
        "- You are NOT giving financial advice.\n"
        "- Use the provided historical context as the basis for your reasoning.\n"
        "- Make conservative, realistic forecasts (no extreme jumps unless the scenario clearly suggests it).\n"
        "- Output ONLY valid JSON with keys 'forecast_prices' and 'explanation'.\n"
        "- 'forecast_prices' must be a list of exactly 3 numbers (floats).\n"
        "- 'explanation' should be 3–6 sentences and mention that this is a scenario-based, "
        "non-guaranteed projection and not financial advice."
    )

    user_prompt = f"""
Company input: {company}
Resolved ticker symbol: {symbol}

Historical data (real, from Yahoo Finance):
- Date range: from {dates[0]} to {dates[-1]}
- Last closing price: {last_price:.2f}
- 12-month low: {min_price:.2f}
- 12-month high: {max_price:.2f}
- Overall trend over the last year: {trend_description} (approx {trend_pct:.1f}% change)

Scenario to analyze:
\"\"\"{scenario}\"\"\"


Task:
1. Based on the recent performance and the scenario, produce a plausible forecast of the stock's closing price at the end of:
   - Month 1
   - Month 2
   - Month 3

2. Provide a short explanation of the key drivers in this scenario and why prices might evolve that way.

Output format (MUST be valid JSON and nothing else):

{{
  "forecast_prices": [p_month_1, p_month_2, p_month_3],
  "explanation": "Your explanation here..."
}}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.35,
    )

    content = response.choices[0].message.content.strip()

    # Strip Markdown ```json fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    import json
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        raise RuntimeError(f"OpenAI returned non-JSON content: {content}")

    forecast_prices = parsed.get("forecast_prices", [])
    explanation = parsed.get("explanation", "").strip()

    if not isinstance(forecast_prices, list) or len(forecast_prices) != 3:
        raise RuntimeError(f"OpenAI returned unexpected forecast_prices: {forecast_prices}")

    try:
        forecast_prices = [float(p) for p in forecast_prices]
    except Exception as e:
        raise RuntimeError(f"Could not parse forecast prices as floats: {forecast_prices}") from e

    return {"forecast": forecast_prices, "explanation": explanation}


# ---------------------------------------------------------------------------
# Helper: basic company analysis (index.html feature)
# ---------------------------------------------------------------------------

def analyze_company_basic(company: str) -> str:
    """
    Simple company analysis using Yahoo Finance fundamentals + OpenAI summary.
    Used by /api/analyze-company for the index.html page.
    """
    if not company.strip():
        raise ValueError("Company is required.")

    symbol = resolve_symbol_from_company(company)
    print(f"[analyze_company_basic] company='{company}' resolved symbol='{symbol}'")

    ticker = yf.Ticker(symbol)
    info = {}
    try:
        info = ticker.info or {}
    except Exception as e:
        print(f"[analyze_company_basic] error fetching info for {symbol}: {e}")

    # Try to get some basic fields safely
    name = info.get("longName") or info.get("shortName") or company
    sector = info.get("sector") or "Unknown sector"
    industry = info.get("industry") or ""
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    day_change = None
    if "regularMarketPreviousClose" in info and current_price is not None:
        prev = info["regularMarketPreviousClose"]
        if prev not in (None, 0):
            day_change = (current_price - prev) / prev * 100.0

    fifty_two_wk_high = info.get("fiftyTwoWeekHigh")
    fifty_two_wk_low = info.get("fiftyTwoWeekLow")
    market_cap = info.get("marketCap")
    pe_ratio = info.get("trailingPE")

    bullet_lines = [
        f"Name: {name}",
        f"Ticker: {symbol}",
        f"Sector: {sector}",
    ]
    if industry:
        bullet_lines.append(f"Industry: {industry}")
    if current_price is not None:
        bullet_lines.append(f"Current price: {current_price}")
    if day_change is not None:
        bullet_lines.append(f"Change today: {day_change:.2f}%")
    if fifty_two_wk_low is not None and fifty_two_wk_high is not None:
        bullet_lines.append(f"52-week range: {fifty_two_wk_low} – {fifty_two_wk_high}")
    if market_cap is not None:
        bullet_lines.append(f"Market cap: {market_cap}")
    if pe_ratio is not None:
        bullet_lines.append(f"Trailing P/E: {pe_ratio}")

    fundamentals_text = "\n".join(bullet_lines)

    system_prompt = (
        "You are a financial analysis assistant. "
        "Given some basic fundamentals for a public company, write a concise, plain-English summary "
        "for a curious retail investor. Highlight what the company does, its general financial health, "
        "and one or two key risks or considerations. Avoid giving direct investment advice or "
        "telling the user explicitly to buy or sell."
    )

    user_prompt = f"""
Please analyze the following company and write a concise overview
(2–4 short paragraphs, no bullet lists unless truly helpful).

Raw fundamentals:
{fundamentals_text}
"""

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Helpers for text extraction (earnings & report analyzers)
# ---------------------------------------------------------------------------

def extract_text_from_filestorage(fs) -> str:
    """
    Extract text from an uploaded FileStorage object.
    Supports .txt natively and .pdf if PyPDF2 is installed.
    """
    filename = fs.filename or "uploaded_file"
    ext = os.path.splitext(filename)[1].lower()
    data = fs.read()

    if ext == ".txt":
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode("latin-1", errors="ignore")

    if ext == ".pdf" and PyPDF2 is not None:
        try:
            from io import BytesIO
            # Wrap raw bytes in a file-like object for PyPDF2
            pdf_stream = BytesIO(data)
            reader = PyPDF2.PdfReader(pdf_stream)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            print(f"[extract_text_from_filestorage] PDF parse error for {filename}: {e}")
            return f"(Could not extract text from PDF {filename}.)"

    if ext == ".pdf" and PyPDF2 is None:
        return f"(PyPDF2 is not installed, so PDF text for {filename} could not be extracted.)"

    # Fallback for unknown types
    return f"(Unsupported file type for {filename}.)"


# ---------------------------------------------------------------------------
# NEW helpers: feasibility, risk, growth projections (Investment Strategy Builder)
# ---------------------------------------------------------------------------

def compute_required_return(current: float, target: float, years: float) -> float | None:
    """
    Compute the annualized return needed to go from `current` to `target` in `years`.

    A = P * (1 + r)^n  ->  r = (A / P)^(1/n) - 1

    Returns r as a decimal (0.07 = 7%). Returns None if invalid input.
    """
    if current <= 0 or target <= 0 or years <= 0:
        return None
    if target <= current:
        # Already at or above target, required return is 0 or negative.
        return 0.0
    try:
        return (target / current) ** (1.0 / years) - 1.0
    except Exception:
        return None


def classify_feasibility(required_r: float | None) -> tuple[str, str]:
    """
    Map required annual return to a label + explanation.

    Rough example thresholds:
      <= 0%: "already_met"
      0–5%: "very_conservative"
      5–10%: "realistic"
      10–20%: "aggressive"
      >20%: "extremely_unlikely"
    """
    if required_r is None:
        return "unknown", "We could not compute a meaningful required annual return from the inputs."

    r_pct = required_r * 100.0

    if r_pct <= 0:
        return (
            "already_met",
            "Your current amount already meets or exceeds your target; you don't need growth to reach this goal."
        )
    if r_pct <= 5:
        return (
            "very_conservative",
            f"Your goal requires about {r_pct:.1f}% annual growth, which is quite conservative and historically achievable with low-risk assets."
        )
    if r_pct <= 10:
        return (
            "realistic",
            f"Your goal requires about {r_pct:.1f}% annual growth, which is realistic for a diversified portfolio over this time horizon."
        )
    if r_pct <= 20:
        return (
            "aggressive",
            f"Your goal requires about {r_pct:.1f}% annual growth, which is aggressive. It may be possible but would likely involve higher-risk assets and volatility."
        )
    return (
        "extremely_unlikely",
        f"Your goal requires about {r_pct:.1f}% annual growth, which is extremely high and historically rare. "
        "This level of return usually implies very high risk and a significant chance of not reaching the target."
    )


def make_growth_projection_chart(
    current: float,
    years: float,
    rates: List[float],
    required_rate: float | None = None
) -> str | None:
    """
    Build a simple growth projection chart with constant annual rates.
    `rates` and `required_rate` are decimals (0.07 = 7%).

    Returns a base64 PNG string, or None on failure.
    """
    if current <= 0 or years <= 0:
        return None

    try:
        horizon_years = max(1, int(math.ceil(years)))
        x = list(range(0, horizon_years + 1))

        fig, ax = plt.subplots(figsize=(8, 3))

        # Base scenarios
        for r in rates:
            y = [current * ((1 + r) ** t) for t in x]
            ax.plot(x, y, label=f"{r * 100:.0f}% per year")

        # Required rate scenario (if makes sense and not already in list)
        if required_rate is not None and required_rate > 0 and required_rate not in rates:
            y_req = [current * ((1 + required_rate) ** t) for t in x]
            ax.plot(x, y_req, linestyle="--", label=f"Required: {required_rate * 100:.1f}%")

        ax.set_title("Simple Growth Projections (Educational Only)")
        ax.set_xlabel("Years")
        ax.set_ylabel("Portfolio Value")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"[make_growth_projection_chart] Error: {e}")
        return None


def build_risk_warning(
    risk_level: str,
    feasibility_label: str,
    required_r: float | None
) -> str:
    """
    Build a user-facing risk warning string combining feasibility + self-declared risk.
    """
    r_pct = required_r * 100.0 if required_r is not None else None

    # Base by feasibility
    if feasibility_label == "extremely_unlikely":
        base = (
            "Your goal requires an extremely high annual return, which is historically rare and implies a high chance "
            "of not reaching the target even with very aggressive strategies."
        )
    elif feasibility_label == "aggressive":
        base = (
            "Your goal requires an aggressive annual return, which likely involves significant volatility and a real "
            "risk of meaningful losses along the way."
        )
    elif feasibility_label in ("realistic", "very_conservative"):
        base = (
            "Your required annual return is within a range that has historically been achievable with diversified "
            "portfolios over similar horizons, but returns are never guaranteed."
        )
    else:
        base = "We couldn't fully assess the feasibility of your goal from the provided numbers."

    # Adjust by user self-declared risk preference
    risk_level = (risk_level or "").lower()
    if risk_level in ("high", "aggressive"):
        addon = (
            " Because you selected a high risk preference, it's especially important to be prepared for large swings "
            "in value and to avoid investing money you cannot afford to lose."
        )
    elif risk_level in ("low", "safe"):
        addon = (
            " Because you prefer low risk, you may need to accept slower growth or consider adjusting your target amount "
            "or timeline if the required return is high."
        )
    else:
        addon = ""

    if r_pct is not None and r_pct > 0:
        prefix = f"In numeric terms, your goal implies about {r_pct:.1f}% annual growth. "
    else:
        prefix = ""

    return prefix + base + addon


# ---------------------------------------------------------------------------
# Routes: pages
# ---------------------------------------------------------------------------

@app.route("/")
def start_page():
    """Landing page where user chooses which feature to open."""
    return send_from_directory(BASE_DIR, "menu.html")


@app.route("/start")
def start_alias():
    return send_from_directory(BASE_DIR, "menu.html")


@app.route("/index")
def index_page():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/earnings")
def earnings_page():
    return send_from_directory(BASE_DIR, "earnings.html")


@app.route("/report")
def report_page():
    return send_from_directory(BASE_DIR, "report.html")


@app.route("/forecast")
def forecast_page():
    return send_from_directory(BASE_DIR, "forecast.html")


@app.route("/forecast.html")
def forecast_html_alias():
    return send_from_directory(BASE_DIR, "forecast.html")


@app.route("/index.html")
def index_html_alias():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/earnings.html")
def earnings_html_alias():
    return send_from_directory(BASE_DIR, "earnings.html")


@app.route("/report.html")
def report_html_alias():
    return send_from_directory(BASE_DIR, "report.html")


@app.route("/forecast.js")
def forecast_js():
    return send_from_directory(BASE_DIR, "forecast.js")


@app.route("/earnings.js")
def earnings_js():
    return send_from_directory(BASE_DIR, "earnings.js")


@app.route("/report.js")
def report_js():
    return send_from_directory(BASE_DIR, "report.js")


@app.route("/script.js")
def script_js():
    return send_from_directory(BASE_DIR, "script.js")


@app.route("/style.css")
def style_css():
    return send_from_directory(BASE_DIR, "style.css")


# New crypto page route (serves crypto.html from project root)
@app.route("/crypto")
def crypto_page():
    return send_from_directory(BASE_DIR, "crypto.html")


@app.route("/crypto.html")
def crypto_html_alias():
    return send_from_directory(BASE_DIR, "crypto.html")


# New metals page route (serves metals.html from project root)
@app.route("/metals")
def metals_page():
    return send_from_directory(BASE_DIR, "metals.html")


@app.route("/metals.html")
def metals_html_alias():
    return send_from_directory(BASE_DIR, "metals.html")


# NEW: Investment Strategy Builder page
@app.route("/strategy")
def strategy_page():
    return send_from_directory(BASE_DIR, "strategy.html")


@app.route("/strategy.html")
def strategy_html_alias():
    return send_from_directory(BASE_DIR, "strategy.html")

# Bonds page routes (NEW)
@app.route("/bonds")
def bonds_page():
    return send_from_directory(BASE_DIR, "bonds.html")


@app.route("/bonds.html")
def bonds_html_alias():
    return send_from_directory(BASE_DIR, "bonds.html")



# ---------------------------------------------------------------------------
# Routes: APIs
# ---------------------------------------------------------------------------

@app.route("/api/analyze-company", methods=["GET"])
def api_analyze_company():
    """
    Simple company overview for index.html
    Query param: ?company=MSFT
    Response JSON: { "summary": "..." }
    """
    company = (request.args.get("company") or "").strip()
    if not company:
        return jsonify({"error": "Missing 'company' query parameter."}), 400

    try:
        summary = analyze_company_basic(company)
        return jsonify({"summary": summary})
    except Exception as e:
        print(f"[api/analyze-company] Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/earnings-qa", methods=["POST"])
def api_earnings_qa():
    """
    Endpoint for earnings.html + earnings.js.

    Expects multipart/form-data with:
      - files: one or more transcript files (PDF or TXT)
      - questions: text area content (one or more questions)
    Returns JSON:
      { "answers": "..." } or { "error": "..." }
    """
    files = request.files.getlist("files")
    questions = (request.form.get("questions") or "").strip()

    if not files:
        return jsonify({"error": "Please upload at least one transcript file."}), 400
    if not questions:
        return jsonify({"error": "Please enter at least one question."}), 400

    transcripts_texts: List[str] = []
    for fs in files:
        if not fs:
            continue
        text = extract_text_from_filestorage(fs)
        transcripts_texts.append(f"File: {fs.filename}\n{text}")

    combined_text = "\n\n".join(transcripts_texts)
    # Light truncation for safety (if huge transcripts)
    max_chars = 15000
    if len(combined_text) > max_chars:
        combined_text = combined_text[:max_chars] + "\n\n[Transcript truncated for length in this preview.]"

    system_prompt = (
        "You are an expert financial analyst specializing in corporate earnings calls. "
        "Given one or more earnings call transcripts and a set of questions, provide clear, concise answers. "
        "If some questions cannot be answered from the transcript, say so explicitly. "
        "Do not fabricate specific figures that are not clearly present."
    )

    user_prompt = f"""
Below are earnings call transcripts (possibly truncated) followed by questions.

=== TRANSCRIPTS START ===
{combined_text}
=== TRANSCRIPTS END ===

=== QUESTIONS ===
{questions}

Please answer in a structured Q&A format. For each question, restate it briefly and then answer.
"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        answers = resp.choices[0].message.content.strip()
        return jsonify({"answers": answers})
    except Exception as e:
        print(f"[api/earnings-qa] Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze-report", methods=["POST"])
def api_analyze_report():
    """
    Endpoint for report.html + report.js.

    Expects multipart/form-data with:
      - files: one or more report files (e.g., annual reports, 10-Ks)
    Returns JSON:
      { "analysis": "..." } or { "error": "..." }
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Please upload at least one report file."}), 400

    report_texts: List[str] = []
    for fs in files:
        if not fs:
            continue
        text = extract_text_from_filestorage(fs)
        report_texts.append(f"File: {fs.filename}\n{text}")

    combined_text = "\n\n".join(report_texts)
    max_chars = 18000
    if len(combined_text) > max_chars:
        combined_text = combined_text[:max_chars] + "\n\n[Reports truncated for length in this preview.]"

    system_prompt = (
        "You are an equity research analyst. "
        "Given one or more company reports or filings (such as annual reports or 10-Ks), "
        "write a concise but insightful analysis for a long-term investor. "
        "Highlight the business model, recent performance, balance sheet health, key risks, "
        "and any notable opportunities or strategic initiatives. Avoid direct investment advice."
        "Key risks or concerns that stand out. Concrete recommendations for how the company can either sustain its success or improve in the next quarter Give a very detailed summary of muliple paragraphs of exact and precise suggestions with what the company can do for the next quarter and how they can achieve these suggestions"
    )

    user_prompt = f"""
        The user has uploaded a company report (10-K, 10-Q, or consolidated financial statements).

        Using the text below, write a detailed analysis in 4–7 well-structured paragraphs.
        Do NOT use bullet points or numbered lists — only paragraphs.

        Cover at least:
        - Overall financial health and performance for the period
        - Revenue and profitability trends
        - Notable segment or regional performance if visible
        - Balance sheet and cash flow strength or weakness
        - Key risks or concerns that stand out
        - Concrete recommendations for how the company can either sustain its success
        or improve in the next quarter
        - Give a very detailed summary of 2-3 paragraphs of exact and precise suggestions with what the company can do for the next quarter and how they can achieve these suggestions

        Write as if explaining to an informed but non-expert investor.

        Report text:
        \"\"\"{combined_text}\"\"\""""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.35,
        )
        analysis = resp.choices[0].message.content.strip()
        return jsonify({"analysis": analysis})
    except Exception as e:
        print(f"[api/analyze-report] Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/forecast_scenario", methods=["POST"])
def api_forecast_scenario():
    """
    Request JSON:
      {
        "company": "AAPL" or "Apple",
        "scenario": "cost of materials for the iPhone goes up 15%"
      }

    Response JSON:
      {
        "symbol": "AAPL",
        "company": "Apple",
        "historical": [ { "date": "YYYY-MM-DD", "close": 123.45 }, ... ],
        "forecast":  [ { "date": "YYYY-MM-DD", "price": 130.0 }, ... ],
        "explanation": "..."
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    company = (data.get("company") or "").strip()
    scenario = (data.get("scenario") or "").strip()

    if not company or not scenario:
        return jsonify({"error": "Both 'company' and 'scenario' are required."}), 400

    try:
        symbol = resolve_symbol_from_company(company)
        print(f"[api_forecast_scenario] Company='{company}' resolved symbol='{symbol}'")

        historical = get_historical_prices(symbol)
        if not historical:
            return jsonify({"error": f"No historical data found for symbol '{symbol}'."}), 404

        oa_result = call_openai_scenario_forecast(symbol, company, scenario, historical)
        forecast_prices = oa_result["forecast"]
        explanation = oa_result["explanation"]

        last_date = dt.date.fromisoformat(historical[-1]["date"])
        forecast_dates = [
            (last_date + relativedelta(months=+1)).isoformat(),
            (last_date + relativedelta(months=+2)).isoformat(),
            (last_date + relativedelta(months=+3)).isoformat(),
        ]
        forecast_points = [
            {"date": d, "price": p}
            for d, p in zip(forecast_dates, forecast_prices)
        ]

        response = {
            "symbol": symbol,
            "company": company,
            "historical": historical,
            "forecast": forecast_points,
            "explanation": explanation,
        }
        return jsonify(response)
    except Exception as e:
        print(f"[api_forecast_scenario] Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/crypto-info", methods=["POST"])
def api_crypto_info():
    """
    Endpoint for crypto.html (inline JS).

    Expects JSON:
      { "cryptos": ["bitcoin", "ethereum", ...] }

    Returns JSON:
      {
        "success": true,
        "cryptos": [ { ...info... }, ... ],
        "chart_image": "<base64 png or null>"
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    cryptos = data.get("cryptos") or []
    if not isinstance(cryptos, list) or not cryptos:
        return jsonify({"success": False, "error": "No cryptos provided."}), 400

    results: List[Dict[str, Any]] = []
    first_valid_symbol: str | None = None

    for raw_symbol in cryptos:
        symbol = (str(raw_symbol) or "").strip()
        if not symbol:
            continue
        try:
            info = CryptoUtils.get_crypto_info(symbol)
            results.append(info)
            if first_valid_symbol is None:
                first_valid_symbol = symbol
        except Exception as e:
            print(f"[api/crypto-info] Error fetching {symbol}: {e}")

    if not results:
        return jsonify({"success": False, "error": "No valid cryptos found."}), 400

    chart_b64: str | None = None
    if first_valid_symbol:
        try:
            df = CryptoUtils.get_crypto_history(first_valid_symbol, days=365)
            if not df.empty:
                fig, ax = plt.subplots(figsize=(8, 3))
                df["price"].plot(ax=ax)
                ax.set_title(f"{first_valid_symbol.capitalize()} — Last 12 Months")
                ax.set_ylabel("Price (USD)")
                ax.grid(True)
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                plt.close(fig)
                buf.seek(0)
                chart_b64 = base64.b64encode(buf.read()).decode("utf-8")
        except Exception as e:
            print(f"[api/crypto-info] Error generating chart for {first_valid_symbol}: {e}")
            chart_b64 = None

    return jsonify({"success": True, "cryptos": results, "chart_image": chart_b64})


@app.route("/api/metals-info", methods=["POST"])
def api_metals_info():
    """
    Endpoint for metals.html (inline JS).

    Expects JSON:
      { "metals": ["gold", "silver", ...] }

    Returns JSON:
      {
        "success": true,
        "metals": [ { ...info... }, ... ],
        "chart_image": "<base64 png or null>"
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    metals = data.get("metals") or []
    if not isinstance(metals, list) or not metals:
        return jsonify({"success": False, "error": "No metals provided."}), 400

    results: List[Dict[str, Any]] = []
    first_valid_input: str | None = None

    for m in metals:
        name = (str(m) or "").strip()
        if not name:
            continue
        try:
            info = MetalsUtils.get_metal_info(name)
            results.append(info)
            if first_valid_input is None:
                first_valid_input = name
        except Exception as e:
            print(f"[api/metals-info] Error fetching {name}: {e}")

    if not results:
        return jsonify({"success": False, "error": "No valid metals found."}), 400

    chart_b64: str | None = None
    if first_valid_input:
        try:
            df = MetalsUtils.get_metal_history(first_valid_input, period="1y")
            if not df.empty:
                fig, ax = plt.subplots(figsize=(8, 3))
                df["price"].plot(ax=ax)
                ax.set_title(f"{first_valid_input.title()} — Last 12 Months")
                ax.set_ylabel("Price")
                ax.grid(True)
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                plt.close(fig)
                buf.seek(0)
                chart_b64 = base64.b64encode(buf.read()).decode("utf-8")
        except Exception as e:
            print(f"[api/metals-info] Error generating chart for {first_valid_input}: {e}")
            chart_b64 = None

    return jsonify({"success": True, "metals": results, "chart_image": chart_b64})


# ---------------------------------------------------------------------------
# NEW: Investment Strategy Builder API
# ---------------------------------------------------------------------------

@app.route("/api/strategy-plan", methods=["POST"])
def api_strategy_plan():
    """
    Build an educational investment strategy plan, feasibility check,
    risk warning, and simple growth projections.

    Expects JSON:
      {
        "goal": "...",
        "current_amount": 50000,
        "target_amount": 100000,
        "time_horizon_years": 10,
        "risk_level": "safe" | "moderate" | "high",
        "preferences": ["stocks", "bonds", "metals", "crypto"]
      }

    Returns JSON:
      {
        "strategy_type": "...",
        "recommended_assets": { ... },
        "examples": [ ... ],
        "explanation": "...",
        "feasibility": {
          "required_annual_return": 0.073,
          "feasibility_label": "realistic",
          "feasibility_explanation": "..."
        },
        "risk_warning": "...",
        "growth_chart_image": "<base64 png or null>"
      }
    """
    data = request.get_json(force=True, silent=True) or {}

    goal_text = (data.get("goal") or "").strip()
    current_amount = float(data.get("current_amount") or 0)
    target_amount = float(data.get("target_amount") or 0)
    time_horizon_years = float(data.get("time_horizon_years") or 0)
    risk_level = (data.get("risk_level") or "").strip()
    preferences = data.get("preferences") or []

    # --- 1) Feasibility check ---
    req_return = compute_required_return(current_amount, target_amount, time_horizon_years)
    feasibility_label, feasibility_expl = classify_feasibility(req_return)

    # --- 2) Growth projection chart ---
    growth_chart_b64 = make_growth_projection_chart(
        current=current_amount,
        years=time_horizon_years,
        rates=[0.04, 0.07, 0.10],
        required_rate=req_return,
    )

    # --- 3) Risk warning text ---
    risk_warning = build_risk_warning(risk_level, feasibility_label, req_return)

    # --- 4) Call OpenAI to build strategy (passing feasibility info) ---
    system_prompt = """
    You are an educational investment strategy assistant inside an app called FinRobot.
    You must NOT give financial advice, only educational examples.

    The user provides a goal, capital, target, time horizon, risk level, and asset preferences.
    You will:
    - Suggest a general strategy_type (e.g., "long-term low-volatility compounding", "high-risk growth").
    - Suggest specific example assets (stocks/ETFs, bonds, metals, crypto) as tickers or names, matching their preferences.
    - Explain WHY this type of strategy is aligned with their goal and risk.
    - Provide concrete examples, but emphasize that these are educational and not recommendations.

    You are also given:
    - required_annual_return (decimal)
    - feasibility_label
    - feasibility_explanation

    Only use widely known, liquid examples (broad index ETFs, major bond ETFs, large-cap stocks, major metals, and large-cap crypto assets).
    Avoid obscure microcaps or illiquid assets.

    You MUST output valid JSON with keys:
      - strategy_type (string)
      - recommended_assets (object with keys like 'stocks', 'bonds', 'metals', 'crypto'; each a list of strings)
      - examples (list of { "asset": string, "why": string })
      - explanation (string)
    """

    user_prompt = f"""
User goal and profile:

- Goal: {goal_text}
- Current amount: {current_amount}
- Target amount: {target_amount}
- Time horizon (years): {time_horizon_years}
- Risk level: {risk_level}
- Preferences: {', '.join(preferences)}

Feasibility info:

- required_annual_return: {req_return}
- feasibility_label: {feasibility_label}
- feasibility_explanation: {feasibility_expl}

Return ONLY the JSON object described in the system prompt.
"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        content = resp.choices[0].message.content.strip()

        # Strip ```json fences if present
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        import json
        plan = json.loads(content)
    except Exception as e:
        print(f"[api_strategy_plan] OpenAI error: {e}")
        return jsonify({"error": "Could not generate strategy"}), 500

    return jsonify({
        "strategy_type": plan.get("strategy_type"),
        "recommended_assets": plan.get("recommended_assets", {}),
        "examples": plan.get("examples", []),
        "explanation": plan.get("explanation", ""),
        "feasibility": {
            "required_annual_return": req_return,
            "feasibility_label": feasibility_label,
            "feasibility_explanation": feasibility_expl,
        },
        "risk_warning": risk_warning,
        "growth_chart_image": growth_chart_b64,
    })

@app.route("/api/bonds-info", methods=["POST"])
def api_bonds_info():
    """
    Endpoint for bonds.html (inline JS).

    Expects JSON:
      { "bonds": ["10-year", "TLT", ...] }

    Returns JSON:
      {
        "success": true,
        "bonds": [ { ...info... }, ... ],
        "chart_image": "<base64 png or null>"
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    bonds = data.get("bonds") or []
    if not isinstance(bonds, list) or not bonds:
        return jsonify({"success": False, "error": "No bonds provided."}), 400

    results: List[Dict[str, Any]] = []
    first_valid_input: str | None = None

    for b in bonds:
        name = (str(b) or "").strip()
        if not name:
            continue
        try:
            info = BondsUtils.get_bond_info(name)
            results.append(info)
            if first_valid_input is None:
                first_valid_input = name
        except Exception as e:
            print(f"[api/bonds-info] Error fetching {name}: {e}")

    if not results:
        return jsonify({"success": False, "error": "No valid bonds found."}), 400

    chart_b64: str | None = None
    if first_valid_input:
        try:
            symbol = BondsUtils.resolve_symbol(first_valid_input)
            df = BondsUtils.get_bond_history(first_valid_input, period="1y")
            if not df.empty:
                fig, ax = plt.subplots(figsize=(8, 3))
                df["value"].plot(ax=ax)
                title = f"{first_valid_input.title()} — Last 12 Months"
                ax.set_title(title)
                ax.set_ylabel("Yield (%)" if symbol.startswith("^") else "Price")
                ax.grid(True)
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                plt.close(fig)
                buf.seek(0)
                chart_b64 = base64.b64encode(buf.read()).decode("utf-8")
        except Exception as e:
            print(f"[api/bonds-info] Error generating chart for {first_valid_input}: {e}")
            chart_b64 = None

    return jsonify({"success": True, "bonds": results, "chart_image": chart_b64})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)














