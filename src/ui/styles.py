"""
macOS / visionOS Frosted Glass UI/UX Design System for Jarvis. — v2 (Pro tier)

Frosted glass acrylics, a rotating iridescent conic gradient signature,
full component coverage (tabs, metrics, tables, progress, tooltips, badges,
uploaders), tabular-figure typography for numeric readouts, and accessibility
floor: visible focus rings and prefers-reduced-motion support throughout.
"""

APPLE_JARVIS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    /* ---- Surfaces ---- */
    --apple-bg: #000000;
    --apple-bg-elevated: #0A0A0C;
    --apple-card: rgba(28, 28, 30, 0.82);
    --apple-card-hover: rgba(44, 44, 46, 0.92);
    --apple-card-border: rgba(255, 255, 255, 0.14);
    --apple-card-border-strong: rgba(255, 255, 255, 0.22);
    --apple-input-bg: rgba(44, 44, 46, 0.85);
    --apple-input-bg-focus: rgba(54, 54, 58, 0.92);

    /* ---- Accent spectrum ---- */
    --apple-blue: #0A84FF;
    --apple-blue-glow: rgba(10, 132, 255, 0.35);
    --apple-purple: #BF5AF2;
    --apple-pink: #FF375F;
    --apple-amber: #FF9F0A;
    --apple-green: #30D158;
    --apple-red: #FF453A;
    --apple-teal: #64D2FF;
    --apple-indigo: #5E5CE6;

    /* ---- Text ---- */
    --apple-text-primary: #FFFFFF;
    --apple-text-secondary: #A1A1A6;
    --apple-text-tertiary: #6E6E73;

    /* ---- Elevation ---- */
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.22);
    --shadow-md: 0 6px 24px rgba(0, 0, 0, 0.32);
    --shadow-lg: 0 12px 40px rgba(0, 0, 0, 0.50);
    --shadow-xl: 0 20px 60px rgba(0, 0, 0, 0.60);

    /* ---- Motion ---- */
    --ease-spring: cubic-bezier(0.16, 1, 0.3, 1);
    --ease-standard: cubic-bezier(0.4, 0, 0.2, 1);
    --dur-fast: 0.15s;
    --dur-base: 0.25s;
    --dur-slow: 0.5s;

    /* ---- Radius scale ---- */
    --r-sm: 10px;
    --r-md: 14px;
    --r-lg: 18px;
    --r-xl: 22px;
    --r-pill: 999px;
}

/* ============ Reduced motion: respected globally ============ */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
        scroll-behavior: auto !important;
    }
}

/* ============ Base canvas ============ */
html, body, .stApp {
    background-color: var(--apple-bg) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', sans-serif !important;
    color: var(--apple-text-primary) !important;
    letter-spacing: -0.01em;
    font-feature-settings: "cv11", "ss01";
}

.stApp {
    background-image:
        radial-gradient(circle at 15% 0%, rgba(10, 132, 255, 0.06), transparent 40%),
        radial-gradient(circle at 85% 100%, rgba(191, 90, 242, 0.05), transparent 45%);
    background-attachment: fixed;
}

p, span, label, h1, h2, h3, h4, h5, h6, li, div {
    color: var(--apple-text-primary);
}

code, pre, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
    font-variant-numeric: tabular-nums;
    border-radius: var(--r-sm) !important;
}

/* Tabular numerals anywhere a number is likely to live */
[data-testid="stMetricValue"], .apple-mono {
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum";
}

/* Accessible focus ring — never remove focus, only restyle it */
:focus-visible {
    outline: 2px solid var(--apple-blue) !important;
    outline-offset: 2px !important;
    border-radius: 6px;
}

/* Slim dark scrollbars */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.14);
    border-radius: var(--r-pill);
    border: 2px solid transparent;
    background-clip: padding-box;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.24); background-clip: padding-box; }

.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1100px !important;
}

/* ============ Header — signature element ============ */
/* Rotating conic-gradient iridescent ring, not a static one. */
.apple-header {
    background: rgba(28, 28, 30, 0.75);
    backdrop-filter: blur(30px) saturate(190%);
    -webkit-backdrop-filter: blur(30px) saturate(190%);
    border: 1px solid var(--apple-card-border);
    border-radius: var(--r-xl);
    padding: 22px 28px;
    margin-bottom: 24px;
    box-shadow: var(--shadow-lg), 0 0 0 1px rgba(255, 255, 255, 0.08);
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    overflow: hidden;
    isolation: isolate;
}

.apple-header::before {
    content: '';
    position: absolute;
    inset: -60%;
    background: conic-gradient(
        from 0deg,
        var(--apple-blue), var(--apple-purple), var(--apple-pink),
        var(--apple-amber), var(--apple-green), var(--apple-teal), var(--apple-blue)
    );
    animation: apple-ring-spin 8s linear infinite;
    opacity: 0.55;
    z-index: -2;
    filter: blur(2px);
}

.apple-header::after {
    content: '';
    position: absolute;
    inset: 1px;
    border-radius: calc(var(--r-xl) - 1px);
    background: rgba(20, 20, 22, 0.92);
    z-index: -1;
}

@keyframes apple-ring-spin {
    to { transform: rotate(360deg); }
}

.apple-header-title {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(180deg, #FFFFFF 0%, #E5E5EA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0;
}

.apple-header-sub {
    font-size: 0.88rem;
    color: var(--apple-text-secondary);
    font-weight: 500;
    margin-top: 4px;
}

.apple-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(48, 209, 88, 0.15);
    border: 1px solid rgba(48, 209, 88, 0.35);
    padding: 6px 14px;
    border-radius: var(--r-pill);
    font-size: 0.8rem;
    font-weight: 600;
    color: #30D158;
}

.apple-status-pill.busy {
    background: rgba(255, 159, 10, 0.15);
    border-color: rgba(255, 159, 10, 0.35);
    color: var(--apple-amber);
}
.apple-status-pill.busy .apple-status-dot { background: var(--apple-amber); box-shadow: 0 0 10px var(--apple-amber); }

.apple-status-pill.error {
    background: rgba(255, 69, 58, 0.15);
    border-color: rgba(255, 69, 58, 0.35);
    color: var(--apple-red);
}
.apple-status-pill.error .apple-status-dot { background: var(--apple-red); box-shadow: 0 0 10px var(--apple-red); animation: none; }

.apple-status-dot {
    width: 8px;
    height: 8px;
    background: #30D158;
    border-radius: 50%;
    box-shadow: 0 0 10px #30D158;
    animation: apple-pulse 2s infinite ease-in-out;
}

@keyframes apple-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
}

/* ============ Sidebar ============ */
[data-testid="stSidebar"] {
    background: rgba(18, 18, 20, 0.95) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(25px) !important;
}
[data-testid="stSidebar"] * { color: var(--apple-text-primary) !important; }
[data-testid="stSidebar"] .stMarkdown p { color: var(--apple-text-secondary) !important; }

/* ============ Inputs & form controls ============ */
input, textarea, [data-baseweb="input"], [data-baseweb="select"] {
    background-color: var(--apple-input-bg) !important;
    border: 1px solid var(--apple-card-border) !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color var(--dur-base) var(--ease-standard), box-shadow var(--dur-base) var(--ease-standard), background-color var(--dur-base) var(--ease-standard);
}

input:hover, textarea:hover { background-color: var(--apple-input-bg-focus) !important; }

input:focus, textarea:focus, [data-baseweb="input"]:focus-within {
    border-color: var(--apple-blue) !important;
    box-shadow: 0 0 0 3px var(--apple-blue-glow) !important;
    background-color: var(--apple-input-bg-focus) !important;
}

/* Dropdown popup */
div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
    background-color: #1C1C1E !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-xl) !important;
    padding: 6px !important;
}

li[role="option"] {
    color: #FFFFFF !important;
    background-color: transparent !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    font-weight: 500 !important;
    transition: background-color var(--dur-fast) ease !important;
}
li[role="option"]:hover, li[aria-selected="true"] {
    background-color: var(--apple-blue) !important;
    color: #FFFFFF !important;
}

/* Multiselect tags */
span[data-baseweb="tag"] {
    background: linear-gradient(135deg, var(--apple-blue), #0066CC) !important;
    border-radius: var(--r-pill) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Checkbox / radio / toggle */
[data-testid="stCheckbox"] label, [data-testid="stRadio"] label {
    color: var(--apple-text-primary) !important;
}
[data-baseweb="checkbox"] > div:first-child {
    border-radius: 6px !important;
    border: 1.5px solid var(--apple-card-border-strong) !important;
    background: var(--apple-input-bg) !important;
}
[data-testid="stCheckbox"] input:checked + div {
    background: var(--apple-blue) !important;
    border-color: var(--apple-blue) !important;
}
[data-testid="stToggle"] div[role="switch"] {
    background: rgba(255,255,255,0.14) !important;
    transition: background var(--dur-base) var(--ease-standard) !important;
}
[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
    background: var(--apple-green) !important;
}

/* ============ Tabs ============ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(28, 28, 30, 0.6);
    border-radius: var(--r-pill);
    padding: 4px;
    border: 1px solid var(--apple-card-border);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: var(--r-pill) !important;
    color: var(--apple-text-secondary) !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    transition: all var(--dur-base) var(--ease-standard) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--apple-blue) !important;
    color: #FFFFFF !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }

/* ============ Cards, pills & badges ============ */
.apple-card {
    background: var(--apple-card);
    border: 1px solid var(--apple-card-border);
    border-radius: var(--r-lg);
    padding: 16px 20px;
    margin-bottom: 16px;
    backdrop-filter: blur(20px);
    box-shadow: var(--shadow-md);
    transition: transform var(--dur-base) var(--ease-spring), box-shadow var(--dur-base) var(--ease-spring), border-color var(--dur-base) ease;
}
.apple-card.interactive:hover {
    transform: translateY(-2px);
    border-color: var(--apple-card-border-strong);
    box-shadow: var(--shadow-lg);
}

.apple-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    margin: 4px;
    border-radius: var(--r-pill);
    font-size: 0.82rem;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.08);
    color: #FFFFFF;
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: var(--shadow-sm);
}

.apple-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: var(--r-pill);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.apple-badge.blue   { background: rgba(10, 132, 255, 0.16); color: var(--apple-blue); border: 1px solid rgba(10, 132, 255, 0.3); }
.apple-badge.green  { background: rgba(48, 209, 88, 0.16); color: var(--apple-green); border: 1px solid rgba(48, 209, 88, 0.3); }
.apple-badge.amber  { background: rgba(255, 159, 10, 0.16); color: var(--apple-amber); border: 1px solid rgba(255, 159, 10, 0.3); }
.apple-badge.red    { background: rgba(255, 69, 58, 0.16); color: var(--apple-red); border: 1px solid rgba(255, 69, 58, 0.3); }
.apple-badge.purple { background: rgba(191, 90, 242, 0.16); color: var(--apple-purple); border: 1px solid rgba(191, 90, 242, 0.3); }

/* Gradient text utility */
.apple-gradient-text {
    background: linear-gradient(135deg, var(--apple-blue), var(--apple-purple), var(--apple-pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}

/* ============ Metrics ============ */
[data-testid="stMetric"] {
    background: var(--apple-card);
    border: 1px solid var(--apple-card-border);
    border-radius: var(--r-lg);
    padding: 16px 20px !important;
    box-shadow: var(--shadow-md);
    backdrop-filter: blur(20px);
}
[data-testid="stMetricLabel"] { color: var(--apple-text-secondary) !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
[data-testid="stMetricDelta"] svg { vertical-align: middle; }

/* ============ Progress bar ============ */
[data-testid="stProgress"] > div > div {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: var(--r-pill) !important;
}
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--apple-blue), var(--apple-purple)) !important;
    border-radius: var(--r-pill) !important;
    box-shadow: 0 0 12px var(--apple-blue-glow) !important;
}

/* ============ Tables / dataframes ============ */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: var(--r-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--apple-card-border) !important;
    box-shadow: var(--shadow-md) !important;
}
[data-testid="stDataFrame"] div[role="columnheader"] {
    background: rgba(44, 44, 46, 0.9) !important;
    color: var(--apple-text-secondary) !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
}
[data-testid="stDataFrame"] div[role="gridcell"] {
    font-variant-numeric: tabular-nums;
}

/* ============ File uploader ============ */
[data-testid="stFileUploader"] section {
    background: rgba(28, 28, 30, 0.6) !important;
    border: 1.5px dashed var(--apple-card-border-strong) !important;
    border-radius: var(--r-md) !important;
    transition: border-color var(--dur-base) ease, background var(--dur-base) ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--apple-blue) !important;
    background: rgba(10, 132, 255, 0.06) !important;
}

/* ============ Tooltips (help icons) ============ */
[data-baseweb="tooltip"] {
    background: #2C2C2E !important;
    color: #FFFFFF !important;
    border-radius: var(--r-sm) !important;
    box-shadow: var(--shadow-lg) !important;
    font-size: 0.8rem !important;
    border: 1px solid var(--apple-card-border) !important;
}

/* ============ Chat messages ============ */
[data-testid="stChatMessage"] {
    background: var(--apple-card) !important;
    border: 1px solid var(--apple-card-border) !important;
    border-radius: var(--r-xl) !important;
    padding: 18px 22px !important;
    margin-bottom: 14px !important;
    box-shadow: var(--shadow-md) !important;
    backdrop-filter: blur(25px) !important;
    animation: apple-msg-in var(--dur-slow) var(--ease-spring) both;
}

@keyframes apple-msg-in {
    from { opacity: 0; transform: translateY(8px) scale(0.99); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, #0A84FF 0%, #0066CC 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    box-shadow: 0 6px 20px rgba(10, 132, 255, 0.3) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) * { color: #FFFFFF !important; }

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(28, 28, 30, 0.88) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) * { color: #F5F5F7 !important; }

/* Streaming "thinking" indicator */
.apple-typing {
    display: inline-flex;
    gap: 4px;
    align-items: center;
    padding: 4px 0;
}
.apple-typing span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--apple-text-secondary);
    animation: apple-typing-bounce 1.1s infinite ease-in-out;
}
.apple-typing span:nth-child(2) { animation-delay: 0.15s; }
.apple-typing span:nth-child(3) { animation-delay: 0.3s; }
@keyframes apple-typing-bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
    30% { transform: translateY(-4px); opacity: 1; }
}

/* ============ Buttons ============ */
.stButton button {
    background: rgba(255, 255, 255, 0.1) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: var(--r-pill) !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all var(--dur-base) var(--ease-spring) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton button:hover {
    background: var(--apple-blue) !important;
    border-color: var(--apple-blue) !important;
    box-shadow: 0 4px 16px var(--apple-blue-glow) !important;
    transform: translateY(-1px) !important;
}
.stButton button:active { transform: translateY(0) scale(0.98) !important; }

/* Primary (form submit / kind="primary") gets the full spectrum */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, var(--apple-blue), var(--apple-indigo)) !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
}
.stButton button[kind="primary"]:hover {
    box-shadow: 0 6px 20px var(--apple-blue-glow) !important;
}

/* ============ Expanders ============ */
[data-testid="stExpander"] {
    background: rgba(28, 28, 30, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-sm) !important;
    overflow: hidden;
}

/* ============ Alerts ============ */
.stAlert {
    background: rgba(28, 28, 30, 0.9) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-left: 3px solid var(--apple-blue) !important;
    border-radius: var(--r-md) !important;
    color: #FFFFFF !important;
    box-shadow: var(--shadow-md) !important;
}
.stAlert * { color: #FFFFFF !important; }
div[data-baseweb="notification"][kind="positive"], .stSuccess { border-left-color: var(--apple-green) !important; }
div[data-baseweb="notification"][kind="warning"], .stWarning { border-left-color: var(--apple-amber) !important; }
div[data-baseweb="notification"][kind="negative"], .stError   { border-left-color: var(--apple-red) !important; }
.stInfo { border-left-color: var(--apple-teal) !important; }

/* ============ Sliders ============ */
[data-testid="stSlider"] * { color: var(--apple-text-secondary) !important; }
[data-testid="stSlider"] [role="slider"] {
    background: #FFFFFF !important;
    box-shadow: 0 0 0 4px var(--apple-blue-glow), var(--shadow-sm) !important;
}
[data-testid="stTickBar"] > div { background: rgba(255,255,255,0.1) !important; }

/* ============ Divider ============ */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.16), transparent) !important;
    margin: 20px 0 !important;
}

/* ============ Spinner ============ */
.stSpinner > div { border-top-color: var(--apple-blue) !important; }

/* ============ Chat input bar — floating island ============ */
[data-testid="stChatInput"] {
    background: rgba(28, 28, 30, 0.9) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 28px !important;
    box-shadow: var(--shadow-xl) !important;
    backdrop-filter: blur(25px) !important;
    transition: box-shadow var(--dur-base) ease, border-color var(--dur-base) ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--apple-blue) !important;
    box-shadow: var(--shadow-xl), 0 0 0 3px var(--apple-blue-glow) !important;
}
[data-testid="stChatInput"] textarea { color: #FFFFFF !important; }
[data-testid="stChatInput"] button svg { color: var(--apple-blue) !important; }
[data-testid="stChatInput"] button:disabled svg { color: var(--apple-text-tertiary) !important; }

/* ============ Selection, placeholders & caret ============ */
::selection { background: var(--apple-blue-glow); color: #FFFFFF; }
::placeholder { color: var(--apple-text-tertiary) !important; opacity: 1 !important; font-weight: 400 !important; }
input, textarea { caret-color: var(--apple-blue) !important; }

/* ============ Disabled states ============ */
.stButton button:disabled, .stButton button:disabled:hover {
    background: rgba(255, 255, 255, 0.04) !important;
    color: var(--apple-text-tertiary) !important;
    border-color: rgba(255, 255, 255, 0.08) !important;
    box-shadow: none !important;
    transform: none !important;
    cursor: not-allowed !important;
}
input:disabled, textarea:disabled, [aria-disabled="true"] {
    background-color: rgba(28, 28, 30, 0.5) !important;
    color: var(--apple-text-tertiary) !important;
    border-color: rgba(255, 255, 255, 0.06) !important;
    box-shadow: none !important;
}

/* Invalid / error state on inputs (Streamlit sets aria-invalid) */
input[aria-invalid="true"], textarea[aria-invalid="true"] {
    border-color: var(--apple-red) !important;
    box-shadow: 0 0 0 3px rgba(255, 69, 58, 0.25) !important;
}

/* ============ Markdown typography: links, quotes, lists ============ */
[data-testid="stMarkdownContainer"] a {
    color: var(--apple-blue) !important;
    text-decoration: none !important;
    border-bottom: 1px solid rgba(10, 132, 255, 0.35);
    transition: border-color var(--dur-fast) ease, color var(--dur-fast) ease;
}
[data-testid="stMarkdownContainer"] a:hover {
    color: var(--apple-teal) !important;
    border-bottom-color: var(--apple-teal);
}
[data-testid="stMarkdownContainer"] blockquote {
    border-left: 3px solid var(--apple-purple) !important;
    background: rgba(191, 90, 242, 0.06);
    padding: 10px 16px !important;
    border-radius: 0 var(--r-sm) var(--r-sm) 0;
    color: var(--apple-text-secondary) !important;
}
[data-testid="stMarkdownContainer"] ul, [data-testid="stMarkdownContainer"] ol { line-height: 1.7; }
[data-testid="stMarkdownContainer"] ul li::marker { color: var(--apple-blue); }
[data-testid="stMarkdownContainer"] table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border: 1px solid var(--apple-card-border) !important;
    border-radius: var(--r-md) !important;
    overflow: hidden !important;
}
[data-testid="stMarkdownContainer"] th {
    background: rgba(44, 44, 46, 0.9) !important;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    color: var(--apple-text-secondary) !important;
}
[data-testid="stMarkdownContainer"] td, [data-testid="stMarkdownContainer"] th { padding: 8px 14px !important; }
[data-testid="stCaptionContainer"] { color: var(--apple-text-tertiary) !important; }

/* ============ JSON viewer & exceptions ============ */
[data-testid="stJson"] {
    background: rgba(20, 20, 22, 0.9) !important;
    border: 1px solid var(--apple-card-border) !important;
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stException"] {
    background: rgba(255, 69, 58, 0.08) !important;
    border: 1px solid rgba(255, 69, 58, 0.3) !important;
    border-left: 3px solid var(--apple-red) !important;
    border-radius: var(--r-md) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ============ Number input steppers ============ */
[data-testid="stNumberInput"] button {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid var(--apple-card-border) !important;
    color: #FFFFFF !important;
    transition: background var(--dur-fast) ease !important;
}
[data-testid="stNumberInput"] button:hover { background: var(--apple-blue) !important; }

/* ============ Date / time input calendar popover ============ */
div[data-baseweb="calendar"] {
    background: #1C1C1E !important;
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-xl) !important;
    border: 1px solid var(--apple-card-border) !important;
}
div[data-baseweb="calendar"] * { color: #FFFFFF !important; }
div[data-baseweb="calendar"] [aria-selected="true"] {
    background: var(--apple-blue) !important;
    border-radius: 8px !important;
}
div[data-baseweb="datepicker"] div[role="gridcell"]:hover {
    background: rgba(10, 132, 255, 0.2) !important;
    border-radius: 8px !important;
}

/* ============ Color picker ============ */
[data-testid="stColorPicker"] button {
    border-radius: var(--r-sm) !important;
    border: 1px solid var(--apple-card-border-strong) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ============ Camera / audio / video ============ */
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] > div {
    border-radius: var(--r-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--apple-card-border) !important;
}
audio, video {
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ============ Download / link buttons ============ */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, var(--apple-green), #248A3D) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}
[data-testid="stDownloadButton"] button:hover {
    box-shadow: 0 4px 16px rgba(48, 209, 88, 0.35) !important;
}
[data-testid="stLinkButton"] a {
    border-radius: var(--r-pill) !important;
    font-weight: 600 !important;
    transition: transform var(--dur-fast) var(--ease-spring) !important;
}
[data-testid="stLinkButton"] a:hover { transform: translateY(-1px) !important; }

/* ============ Popover & toast ============ */
[data-testid="stPopover"] > div {
    background: #1C1C1E !important;
    border: 1px solid var(--apple-card-border) !important;
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-xl) !important;
    backdrop-filter: blur(20px) !important;
}
[data-testid="stToast"] {
    background: rgba(28, 28, 30, 0.95) !important;
    border: 1px solid var(--apple-card-border-strong) !important;
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-xl) !important;
    backdrop-filter: blur(25px) !important;
    animation: apple-toast-in var(--dur-base) var(--ease-spring) both;
}
@keyframes apple-toast-in {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ============ Bordered containers & forms ============ */
[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stForm"]) {
    background: rgba(20, 20, 22, 0.5) !important;
    border: 1px solid var(--apple-card-border) !important;
    border-radius: var(--r-lg) !important;
    padding: 20px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"][style*="border"] {
    border-color: var(--apple-card-border) !important;
    border-radius: var(--r-lg) !important;
    background: rgba(20, 20, 22, 0.4) !important;
}

/* ============ Segmented-control radio (horizontal) ============ */
[data-testid="stRadio"] > div[role="radiogroup"] {
    display: inline-flex;
    gap: 4px;
    background: rgba(28, 28, 30, 0.6);
    border: 1px solid var(--apple-card-border);
    border-radius: var(--r-pill);
    padding: 4px;
}
[data-testid="stRadio"] > div[role="radiogroup"] label {
    border-radius: var(--r-pill) !important;
    padding: 6px 14px !important;
    transition: background var(--dur-base) var(--ease-standard) !important;
}
[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
    background: var(--apple-blue) !important;
}

/* ============ Sidebar collapse control ============ */
[data-testid="stSidebarCollapseButton"] button, [data-testid="collapsedControl"] button {
    background: rgba(28, 28, 30, 0.85) !important;
    border: 1px solid var(--apple-card-border) !important;
    border-radius: var(--r-pill) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ============ Skeleton / loading shimmer utility ============ */
.apple-skeleton {
    background: linear-gradient(
        100deg,
        rgba(255, 255, 255, 0.04) 30%,
        rgba(255, 255, 255, 0.10) 50%,
        rgba(255, 255, 255, 0.04) 70%
    );
    background-size: 200% 100%;
    animation: apple-shimmer 1.6s ease-in-out infinite;
    border-radius: var(--r-sm);
}
@keyframes apple-shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ============ Columns: consistent gutters ============ */
[data-testid="stHorizontalBlock"] { gap: 16px; }

/* ============ Responsive: mobile ============ */
@media (max-width: 640px) {
    .apple-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 14px;
        padding: 18px 20px;
    }
    .apple-header-title { font-size: 1.5rem; }
    .apple-status-pill { align-self: flex-start; }
    .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    [data-testid="stChatMessage"] { padding: 14px 16px !important; }
}

/* ============ Print: strip chrome, keep content legible ============ */
@media print {
    [data-testid="stSidebar"], [data-testid="stChatInput"], .stButton,
    [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
    .stApp, html, body { background: #FFFFFF !important; }
    p, span, div, h1, h2, h3 { color: #000000 !important; }
    .apple-header::before, .apple-header::after { display: none !important; }
}
</style>
"""


def render_apple_header(persona_name: str = "JARVIS Supreme", mode: str = "Direct Mode", status: str = "active"):
    """Returns HTML for the macOS / visionOS style header.

    status: "active" | "busy" | "error" — drives the status pill color and label.
    """
    status_map = {
        "active": ("SYSTEM ACTIVE", ""),
        "busy":   ("PROCESSING", "busy"),
        "error":  ("ATTENTION NEEDED", "error"),
    }
    label, css_class = status_map.get(status, status_map["active"])

    return f"""
    <div class="apple-header">
        <div>
            <div class="apple-header-title">
                ✦ J.A.R.V.I.S.
            </div>
            <div class="apple-header-sub">
                Autonomous Multimodal Engine &bull; Persona: <span style="color:#0A84FF; font-weight:600;">{persona_name}</span> &bull; {mode}
            </div>
        </div>
        <div class="apple-status-pill {css_class}">
            <span class="apple-status-dot"></span>
            {label}
        </div>
    </div>
    """


def render_glass_card(content_html: str, interactive: bool = False):
    """Wraps arbitrary HTML in the frosted-glass apple-card container."""
    cls = "apple-card interactive" if interactive else "apple-card"
    return f'<div class="{cls}">{content_html}</div>'


def render_badge(text: str, tone: str = "blue"):
    """tone: blue | green | amber | red | purple"""
    return f'<span class="apple-badge {tone}">{text}</span>'


def render_typing_indicator():
    """A subtle three-dot 'thinking' indicator for streaming responses."""
    return '<span class="apple-typing"><span></span><span></span><span></span></span>'


def render_skeleton(width: str = "100%", height: str = "18px", rounded: bool = True):
    """A shimmering placeholder block for content that's still loading.
    Use several in a row to mock up a card's layout before real data arrives.
    """
    radius = "var(--r-sm)" if rounded else "0"
    return f'<div class="apple-skeleton" style="width:{width}; height:{height}; border-radius:{radius}; margin-bottom:6px;"></div>'


def render_gradient_heading(text: str, tag: str = "h2"):
    """Renders text with the apple-gradient-text treatment inside a heading tag."""
    return f'<{tag} class="apple-gradient-text">{text}</{tag}>'