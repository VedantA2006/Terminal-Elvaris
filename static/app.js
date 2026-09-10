/**
 * TradingView Duplicate Web Terminal — JavaScript Application Logic.
 * 
 * Features:
 *  - Lightweight Charts (Candlesticks, Indicators, Trade Markers, Equity Curve)
 *  - Canvas Trade Position Boxes (TradingView Long / Short Position tools overlay)
 *  - Floating Monthly Performance Table (exact replica of TradingView strategy results)
 *  - Interactive Trade Navigator Bar (Step through trades one by one with centered chart)
 *  - Monaco Editor (Python Strategy Editor)
 *  - Dynamic Strategy Execution & AST Lookahead Bias Guard Feedback
 *  - TradingView Strategy Tester (Overview, Performance, Trade List)
 *  - AI Strategy Generator & Autonomous Continuous Optimizer
 */

// =============================================================================
// GLOBAL STATE
// =============================================================================
let mainChart, equityChart;
let candleSeries, volumeSeries, equitySeries;
let indicatorSeries = {};
let customIndicatorSeries = {};
let monacoEditor = null;

let currentOhlc = [];
let currentTrades = [];
let currentStats = {};
let currentCode = '';
let currentMarkers = [];
let isOptimizing = false;
let stopOptimizationRequested = false;

let showTradeBoxes = true;
let activeTradeIndex = 0;
let overlayCanvas = null;
let overlayCtx = null;

const TV_COLORS = {
    bg: '#131722',
    panelBg: '#1e222d',
    gridLines: '#1f2431',
    text: '#787b86',
    textBright: '#d1d4dc',
    green: '#089981',
    red: '#f23645',
    blue: '#2962ff',
    cyan: '#00bcd4',
    yellow: '#f6a623',
    purple: '#ab47bc',
};

// =============================================================================
// TOAST NOTIFICATIONS
// =============================================================================
function showToast(message, type = 'info') {
    const toast = document.getElementById('tvToast');
    if (!toast) return;
    toast.textContent = message;
    toast.className = `tv-toast show ${type}`;
    setTimeout(() => {
        toast.className = 'tv-toast';
    }, 4500);
}

// =============================================================================
// INITIALIZE LIGHTWEIGHT CHARTS
// =============================================================================
function initCharts() {
    const mount = document.getElementById('mainChartMount');
    mainChart = LightweightCharts.createChart(mount, {
        width: mount.clientWidth,
        height: mount.clientHeight,
        layout: {
            background: { type: 'solid', color: TV_COLORS.bg },
            textColor: TV_COLORS.text,
            fontFamily: "'Inter', sans-serif",
            fontSize: 11,
        },
        grid: {
            vertLines: { color: TV_COLORS.gridLines },
            horzLines: { color: TV_COLORS.gridLines },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: { color: '#ffffff30', width: 1, style: 3 },
            horzLine: { color: '#ffffff30', width: 1, style: 3 },
        },
        rightPriceScale: {
            borderColor: TV_COLORS.gridLines,
            scaleMargins: { top: 0.08, bottom: 0.15 },
        },
        timeScale: {
            borderColor: TV_COLORS.gridLines,
            timeVisible: true,
            secondsVisible: false,
        },
    });

    // Candlesticks with authentic TradingView colors
    candleSeries = mainChart.addCandlestickSeries({
        upColor: TV_COLORS.green,
        downColor: TV_COLORS.red,
        borderUpColor: TV_COLORS.green,
        borderDownColor: TV_COLORS.red,
        wickUpColor: TV_COLORS.green,
        wickDownColor: TV_COLORS.red,
    });

    // Volume histogram
    volumeSeries = mainChart.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
    });
    mainChart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
    });

    // Clean Chart: EMA, SMA, BB, and all indicator lines removed as requested.
    // The chart displays pure price action (candlesticks + volume) with trade position boxes.

    // Equity chart in Strategy Tester
    const eqMount = document.getElementById('equityChartMount');
    equityChart = LightweightCharts.createChart(eqMount, {
        width: eqMount.clientWidth,
        height: eqMount.clientHeight,
        layout: {
            background: { type: 'solid', color: TV_COLORS.panelBg },
            textColor: TV_COLORS.text,
            fontFamily: "'Inter', sans-serif",
            fontSize: 10,
        },
        grid: {
            vertLines: { color: TV_COLORS.gridLines },
            horzLines: { color: TV_COLORS.gridLines },
        },
        rightPriceScale: { borderColor: TV_COLORS.gridLines },
        timeScale: { borderColor: TV_COLORS.gridLines, timeVisible: true, visible: true },
        crosshair: {
            vertLine: { visible: true, color: '#ffffff22' },
            horzLine: { visible: true, color: '#ffffff22' },
        },
    });

    equitySeries = equityChart.addAreaSeries({
        topColor: 'rgba(41, 98, 255, 0.35)',
        bottomColor: 'rgba(41, 98, 255, 0.02)',
        lineColor: TV_COLORS.blue,
        lineWidth: 2,
    });

    // Setup Canvas Overlay for Trade Position Boxes
    overlayCanvas = document.getElementById('tradeOverlayCanvas');
    if (overlayCanvas) {
        overlayCtx = overlayCanvas.getContext('2d');
        syncCanvasSize();
    }

    // Subscribe to chart zoom/pan events to re-render Trade Boxes
    mainChart.timeScale().subscribeVisibleLogicalRangeChange(() => {
        requestAnimationFrame(drawTradeBoxes);
    });
    mainChart.subscribeCrosshairMove(param => {
        if (!param || !param.time || !param.seriesData.get(candleSeries)) {
            updateDefaultLegend();
            return;
        }
        const bar = param.seriesData.get(candleSeries);
        const change = bar.close - bar.open;
        const changePct = (change / bar.open) * 100;
        const sign = change >= 0 ? '+' : '';

        document.getElementById('legendOhlc').innerHTML = `
            O <span class="ohlc-v">${bar.open.toFixed(2)}</span>
            H <span class="ohlc-v">${bar.high.toFixed(2)}</span>
            L <span class="ohlc-v">${bar.low.toFixed(2)}</span>
            C <span class="ohlc-v">${bar.close.toFixed(2)}</span>
            <span class="${change >= 0 ? 'positive' : 'negative'}">${sign}${change.toFixed(2)} (${sign}${changePct.toFixed(2)}%)</span>
        `;
    });
}

function syncCanvasSize() {
    if (!overlayCanvas || !mainChart) return;
    const mount = document.getElementById('mainChartMount');
    overlayCanvas.width = mount.clientWidth;
    overlayCanvas.height = mount.clientHeight;
}

function updateDefaultLegend() {
    if (!currentOhlc || currentOhlc.length === 0) return;
    const last = currentOhlc[currentOhlc.length - 1];
    const prev = currentOhlc.length > 1 ? currentOhlc[currentOhlc.length - 2] : last;
    const change = last.close - prev.close;
    const changePct = (change / prev.close) * 100;
    const sign = change >= 0 ? '+' : '';

    document.getElementById('legendOhlc').innerHTML = `
        O <span class="ohlc-v">${last.open.toFixed(2)}</span>
        H <span class="ohlc-v">${last.high.toFixed(2)}</span>
        L <span class="ohlc-v">${last.low.toFixed(2)}</span>
        C <span class="ohlc-v">${last.close.toFixed(2)}</span>
        <span class="${change >= 0 ? 'positive' : 'negative'}">${sign}${change.toFixed(2)} (${sign}${changePct.toFixed(2)}%)</span>
    `;

    document.getElementById('quotePrice').textContent = `$${last.close.toFixed(2)}`;
    const qChg = document.getElementById('quoteChange');
    qChg.textContent = `${sign}${changePct.toFixed(2)}%`;
    qChg.className = `tv-quote-change ${change >= 0 ? 'positive' : 'negative'}`;

    document.getElementById('wlPriceXAU').textContent = `$${last.close.toFixed(2)}`;
    const wlChg = document.getElementById('wlChgXAU');
    wlChg.textContent = `${sign}${changePct.toFixed(2)}%`;
    wlChg.className = `wl-chg ${change >= 0 ? 'positive' : 'negative'}`;

    document.getElementById('cardPriceMain').textContent = `$${last.close.toFixed(2)}`;
    const highs = currentOhlc.map(b => b.high);
    const lows = currentOhlc.map(b => b.low);
    document.getElementById('csHigh').textContent = `$${Math.max(...highs).toFixed(2)}`;
    document.getElementById('csLow').textContent = `$${Math.min(...lows).toFixed(2)}`;
}

// =============================================================================
// PARSE & FORMAT TIMESTAMPS (Ensure exact consistency with Dukascopy bar times)
// =============================================================================
function parseTimestamp(ts) {
    if (!ts) return null;
    if (typeof ts === 'number') return ts;
    const s = String(ts).trim();
    if (/^\d+$/.test(s)) return parseInt(s, 10);
    const utcStr = s.endsWith('Z') || s.includes('+') || (s.lastIndexOf('-') > 10) ? s : s + 'Z';
    return Math.floor(new Date(utcStr).getTime() / 1000);
}

function formatTradeTime(isoStr) {
    if (!isoStr) return '—';
    try {
        const parts = String(isoStr).split('T');
        const datePart = parts[0];
        const timePart = parts[1] ? parts[1].slice(0, 5) : '';
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const [y, m, d] = datePart.split('-');
        const monthName = months[parseInt(m, 10) - 1] || m;
        return `${monthName} ${parseInt(d, 10)}, ${timePart}`;
    } catch (e) {
        return String(isoStr);
    }
}

// =============================================================================
// DRAW TRADINGVIEW LONG & SHORT POSITION BOXES ON CANVAS
// =============================================================================
function drawTradeBoxes() {
    if (!overlayCtx || !overlayCanvas || !mainChart || !candleSeries) return;
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (!showTradeBoxes || !currentTrades || currentTrades.length === 0) return;

    const timeScale = mainChart.timeScale();
    const visibleRange = timeScale.getVisibleRange();
    if (!visibleRange) return;

    const visibleFrom = visibleRange.from;
    const visibleTo = visibleRange.to;

    currentTrades.forEach((trade, idx) => {
        const entryTs = trade.entry_time_ts ? trade.entry_time_ts : parseTimestamp(trade.entry_time);
        const exitTs = trade.exit_time_ts ? trade.exit_time_ts : (trade.exit_time ? parseTimestamp(trade.exit_time) : (entryTs ? entryTs + (20 * 300) : null));

        if (!entryTs) return;

        // Fast skip if trade is completely outside visible time window
        if (exitTs && (exitTs < visibleFrom - 3600 || entryTs > visibleTo + 3600)) return;

        const x1 = timeScale.timeToCoordinate(entryTs);
        let x2 = exitTs ? timeScale.timeToCoordinate(exitTs) : null;
        if (x1 === null) return;
        if (x2 === null || x2 <= x1) x2 = x1 + 60;

        const boxWidth = Math.max(x2 - x1, 18);

        const yEntry = candleSeries.priceToCoordinate(trade.entry_price);
        const ySL = trade.sl ? candleSeries.priceToCoordinate(trade.sl) : null;
        const tpPrice = (trade.tps && trade.tps[0]) ? trade.tps[0] : (trade.exit_price || trade.entry_price + (trade.direction === 'long' ? 15 : -15));
        const yTP = candleSeries.priceToCoordinate(tpPrice);
        const yExit = trade.exit_price ? candleSeries.priceToCoordinate(trade.exit_price) : yEntry;

        if (yEntry === null || ySL === null || yTP === null) return;

        const isLong = trade.direction === 'long';
        const isWin = trade.pnl > 0;
        const isActive = (idx === activeTradeIndex);

        overlayCtx.save();

        if (isLong) {
            // ---- LONG POSITION BOX ----
            // 1. Profit Target Box (Green, above entry)
            const tpTop = Math.min(yEntry, yTP);
            const tpHeight = Math.abs(yEntry - yTP);
            overlayCtx.fillStyle = isActive ? 'rgba(8, 153, 129, 0.35)' : 'rgba(8, 153, 129, 0.18)';
            overlayCtx.fillRect(x1, tpTop, boxWidth, tpHeight);
            overlayCtx.strokeStyle = 'rgba(8, 153, 129, 0.7)';
            overlayCtx.lineWidth = isActive ? 1.5 : 1;
            overlayCtx.strokeRect(x1, tpTop, boxWidth, tpHeight);

            // 2. Stop Loss Box (Red, below entry)
            const slTop = yEntry;
            const slHeight = Math.abs(ySL - yEntry);
            overlayCtx.fillStyle = isActive ? 'rgba(242, 54, 69, 0.35)' : 'rgba(242, 54, 69, 0.18)';
            overlayCtx.fillRect(x1, slTop, boxWidth, slHeight);
            overlayCtx.strokeStyle = 'rgba(242, 54, 69, 0.7)';
            overlayCtx.lineWidth = isActive ? 1.5 : 1;
            overlayCtx.strokeRect(x1, slTop, boxWidth, slHeight);

            // 3. Entry Line (Cyan / Blue)
            overlayCtx.beginPath();
            overlayCtx.strokeStyle = '#00bcd4';
            overlayCtx.lineWidth = 1.5;
            overlayCtx.moveTo(x1, yEntry);
            overlayCtx.lineTo(x1 + boxWidth, yEntry);
            overlayCtx.stroke();

            // 4. TP Target Dashed Line
            overlayCtx.beginPath();
            overlayCtx.setLineDash([4, 4]);
            overlayCtx.strokeStyle = '#089981';
            overlayCtx.moveTo(x1, yTP);
            overlayCtx.lineTo(x1 + boxWidth, yTP);
            overlayCtx.stroke();

            // 5. SL Stop Dashed Line
            overlayCtx.beginPath();
            overlayCtx.setLineDash([4, 4]);
            overlayCtx.strokeStyle = '#f23645';
            overlayCtx.moveTo(x1, ySL);
            overlayCtx.lineTo(x1 + boxWidth, ySL);
            overlayCtx.stroke();
            overlayCtx.setLineDash([]);

            // Label: LONG Entry Pill
            drawPill(overlayCtx, x1, yEntry, `▲ LONG ${trade.entry_price.toFixed(2)}`, '#089981', '#ffffff');
            // Label: TP Target
            drawMiniLabel(overlayCtx, x1 + boxWidth, yTP, `TP ${tpPrice.toFixed(2)}`, '#089981');
            // Label: SL Stop
            drawMiniLabel(overlayCtx, x1 + boxWidth, ySL, `SL ${trade.sl.toFixed(2)}`, '#f23645');

        } else {
            // ---- SHORT POSITION BOX ----
            // 1. Stop Loss Box (Red, above entry)
            const slTop = Math.min(yEntry, ySL);
            const slHeight = Math.abs(ySL - yEntry);
            overlayCtx.fillStyle = isActive ? 'rgba(242, 54, 69, 0.35)' : 'rgba(242, 54, 69, 0.18)';
            overlayCtx.fillRect(x1, slTop, boxWidth, slHeight);
            overlayCtx.strokeStyle = 'rgba(242, 54, 69, 0.7)';
            overlayCtx.lineWidth = isActive ? 1.5 : 1;
            overlayCtx.strokeRect(x1, slTop, boxWidth, slHeight);

            // 2. Profit Target Box (Green, below entry)
            const tpTop = yEntry;
            const tpHeight = Math.abs(yTP - yEntry);
            overlayCtx.fillStyle = isActive ? 'rgba(8, 153, 129, 0.35)' : 'rgba(8, 153, 129, 0.18)';
            overlayCtx.fillRect(x1, tpTop, boxWidth, tpHeight);
            overlayCtx.strokeStyle = 'rgba(8, 153, 129, 0.7)';
            overlayCtx.lineWidth = isActive ? 1.5 : 1;
            overlayCtx.strokeRect(x1, tpTop, boxWidth, tpHeight);

            // 3. Entry Line
            overlayCtx.beginPath();
            overlayCtx.strokeStyle = '#ff9800';
            overlayCtx.lineWidth = 1.5;
            overlayCtx.moveTo(x1, yEntry);
            overlayCtx.lineTo(x1 + boxWidth, yEntry);
            overlayCtx.stroke();

            // 4. Dashed lines
            overlayCtx.beginPath();
            overlayCtx.setLineDash([4, 4]);
            overlayCtx.strokeStyle = '#089981';
            overlayCtx.moveTo(x1, yTP);
            overlayCtx.lineTo(x1 + boxWidth, yTP);
            overlayCtx.stroke();

            overlayCtx.beginPath();
            overlayCtx.setLineDash([4, 4]);
            overlayCtx.strokeStyle = '#f23645';
            overlayCtx.moveTo(x1, ySL);
            overlayCtx.lineTo(x1 + boxWidth, ySL);
            overlayCtx.stroke();
            overlayCtx.setLineDash([]);

            // Label: SHORT Entry Pill
            drawPill(overlayCtx, x1, yEntry, `▼ SHORT ${trade.entry_price.toFixed(2)}`, '#f23645', '#ffffff');
            drawMiniLabel(overlayCtx, x1 + boxWidth, yTP, `TP ${tpPrice.toFixed(2)}`, '#089981');
            drawMiniLabel(overlayCtx, x1 + boxWidth, ySL, `SL ${trade.sl.toFixed(2)}`, '#f23645');
        }

        // Result Badge at trade exit
        if (trade.exit_price && yExit !== null) {
            const pnlSign = isWin ? '+' : '';
            const resText = `${isWin ? '✓' : '✕'} ${pnlSign}$${trade.pnl.toFixed(0)} (${trade.exit_reason})`;
            drawResultBadge(overlayCtx, x1 + boxWidth, yExit, resText, isWin ? '#089981' : '#f23645');
        }

        overlayCtx.restore();
    });
}

function drawPill(ctx, x, y, text, bg, color) {
    ctx.font = 'bold 9.5px Inter, sans-serif';
    const paddingH = 6;
    const width = ctx.measureText(text).width + paddingH * 2;
    const height = 18;
    const radius = 3;

    ctx.fillStyle = bg;
    ctx.beginPath();
    ctx.roundRect(x - 2, y - height / 2, width, height, radius);
    ctx.fill();

    ctx.fillStyle = color;
    ctx.fillText(text, x + paddingH - 2, y + 3.5);
}

function drawMiniLabel(ctx, x, y, text, color) {
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillStyle = color;
    ctx.fillText(text, x + 4, y + 3);
}

function drawResultBadge(ctx, x, y, text, bg) {
    ctx.font = 'bold 9.5px Inter, sans-serif';
    const paddingH = 6;
    const width = ctx.measureText(text).width + paddingH * 2;
    const height = 17;
    const radius = 4;

    ctx.fillStyle = bg;
    ctx.beginPath();
    ctx.roundRect(x + 2, y - height / 2, width, height, radius);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, x + paddingH + 2, y + 3.5);
}

// Clean Chart Marker Display Manager: Only displays trades taken
function updateMarkersDisplay() {
    if (!candleSeries) return;
    if (showTradeBoxes) {
        // Trade Position Boxes canvas is active: clear candle markers so no duplicate text clutters the chart
        candleSeries.setMarkers([]);
    } else {
        // When Trade Position boxes are toggled off: show clean entry/exit markers for trades taken directly on candles
        candleSeries.setMarkers(currentMarkers || []);
    }
}

// =============================================================================
// FLOATING MONTHLY PERFORMANCE TABLE (TradingView Replica)
// =============================================================================
function renderFloatingMonthlyStats(trades, stats) {
    const tbody = document.getElementById('fswTableBody');
    if (!tbody || !trades || trades.length === 0) return;

    // Group trades by YYYY-MM
    const monthsMap = {};
    const avgRisk = stats.avg_loss > 0 ? stats.avg_loss : 1000;
    trades.forEach(t => {
        const mKey = t.entry_time.slice(0, 7);
        if (!monthsMap[mKey]) {
            monthsMap[mKey] = { trades: 0, wins: 0, pnl: 0, r: 0 };
        }
        monthsMap[mKey].trades++;
        if (t.pnl > 0) monthsMap[mKey].wins++;
        monthsMap[mKey].pnl += t.pnl;
        const rVal = (t.r_return !== undefined && t.r_return !== null) ? Number(t.r_return) : (t.pnl / avgRisk);
        monthsMap[mKey].r += rVal;
    });

    const months = Object.keys(monthsMap).sort().reverse();

    tbody.innerHTML = months.map(m => {
        const data = monthsMap[m];
        const winRate = ((data.wins / data.trades) * 100).toFixed(1);
        const rReturns = Math.round(data.r);
        const rSign = rReturns >= 0 ? '+' : '';
        const pnlCls = data.pnl >= 0 ? 'positive' : 'negative';

        return `
            <tr>
                <td>${m}</td>
                <td class="${pnlCls}">${rSign}${rReturns}R ($${data.pnl >= 0 ? '+' : ''}${Math.round(data.pnl).toLocaleString()})</td>
                <td class="${winRate >= 50 ? 'positive' : ''}">${winRate}%</td>
                <td>${data.trades}</td>
            </tr>
        `;
    }).join('');

    // Summary footer
    const maxDD = stats.max_drawdown ?? 0;
    const rDD = Math.round(maxDD / avgRisk);
    document.getElementById('fswMaxDD').textContent = `-$${maxDD.toLocaleString()} (-${rDD}R)`;
    document.getElementById('fswPF').textContent = stats.profit_factor >= 999 ? '∞' : (stats.profit_factor ?? 0);
    const netPnl = stats.total_pnl ?? 0;
    const netEl = document.getElementById('fswNetPnl');
    netEl.textContent = `${netPnl >= 0 ? '+' : ''}$${netPnl.toLocaleString()}`;
    netEl.className = netPnl >= 0 ? 'positive' : 'negative';
}

// =============================================================================
// TRADE NAVIGATOR (Step through trades one by one)
// =============================================================================
function updateTradeNavigator() {
    if (!currentTrades || currentTrades.length === 0) return;
    const trade = currentTrades[activeTradeIndex];
    if (!trade) return;

    const dirBadge = document.getElementById('tnavDir');
    if (dirBadge) {
        dirBadge.textContent = trade.direction.toUpperCase();
        dirBadge.className = `tnav-badge ${trade.direction}`;
    }

    const textEl = document.getElementById('tnavText');
    if (textEl) {
        const entryDt = formatTradeTime(trade.entry_time);
        textEl.textContent = `Trade #${trade.id} of ${currentTrades.length} · ${entryDt} ($${trade.entry_price.toFixed(2)} → $${(trade.exit_price || 0).toFixed(2)})`;
    }

    const pnlEl = document.getElementById('tnavPnl');
    if (pnlEl) {
        const isWin = trade.pnl > 0;
        pnlEl.textContent = `${isWin ? '+' : ''}$${trade.pnl.toFixed(2)} (${trade.exit_reason || ''})`;
        pnlEl.className = `tnav-pnl ${isWin ? 'positive' : 'negative'}`;
    }

    const listBtn = document.getElementById('tnavOpenListBtn');
    if (listBtn) {
        listBtn.textContent = `📋 List (${currentTrades.length})`;
    }
}

function jumpToTradeIndex(idx) {
    if (!currentTrades || currentTrades.length === 0) return;
    activeTradeIndex = Math.max(0, Math.min(idx, currentTrades.length - 1));
    updateTradeNavigator();

    const trade = currentTrades[activeTradeIndex];
    const ts = trade.entry_time_ts ? trade.entry_time_ts : parseTimestamp(trade.entry_time);
    if (!ts) return;
    // Center chart with ~100 bars context
    mainChart.timeScale().setVisibleRange({
        from: ts - (40 * 300),
        to: ts + (60 * 300),
    });

    requestAnimationFrame(drawTradeBoxes);
}

// =============================================================================
// INITIAL DATA LOADING (Baseline)
// =============================================================================
async function loadBaselineData() {
    try {
        const [ohlcRes, indRes, sigRes, tradesRes, eqRes, statsRes, codeRes] = await Promise.all([
            fetch('/api/ohlc'),
            fetch('/api/indicators'),
            fetch('/api/signals'),
            fetch('/api/trades'),
            fetch('/api/equity'),
            fetch('/api/stats'),
            fetch('/api/strategy/default'),
        ]);

        currentOhlc = await ohlcRes.json();
        const indicators = await indRes.json();
        const signals = await sigRes.json();
        currentTrades = await tradesRes.json();
        const equity = await eqRes.json();
        currentStats = await statsRes.json();
        const defaultCode = await codeRes.json();

        // 1. Candlesticks & Volume
        candleSeries.setData(currentOhlc);
        const vols = currentOhlc.map(b => ({
            time: b.time,
            value: b.volume || 100,
            color: b.close >= b.open ? '#08998133' : '#f2364533',
        }));
        volumeSeries.setData(vols);

        // 2. Clean Markers: Only for trades taken
        currentMarkers = signals || [];
        updateMarkersDisplay();

        // 3. Clean chart: No indicator lines plotted

function prepareEquityData(equityList) {
    if (!equityList || !equityList.length) return [];
    const eqMap = new Map();
    equityList.forEach(e => {
        let ts = typeof e.time_ts === 'number' && e.time_ts > 0
            ? e.time_ts
            : (e.time ? Math.floor(new Date(e.time).getTime() / 1000) : NaN);
        if (!isNaN(ts)) {
            eqMap.set(ts, e.equity);
        }
    });
    return Array.from(eqMap.entries())
        .sort((a, b) => a[0] - b[0])
        .map(([time, value]) => ({ time, value }));
}

        // 4. Equity Curve
        const eqData = prepareEquityData(equity);
        if (eqData.length) equitySeries.setData(eqData);

        // 5. Render Strategy Tester & Floating Stats & Navigator
        renderStrategyTester(currentStats, currentTrades);
        renderFloatingMonthlyStats(currentTrades, currentStats);
        updateTradeNavigator();
        runMonteCarloSimulation(currentTrades);
        fetchLeaderboard();

        // 6. Monaco Editor default code
        if (monacoEditor && defaultCode.code) {
            monacoEditor.setValue(defaultCode.code);
        }
        currentCode = defaultCode.code || '';

        // Auto-fit scales and jump to the most recent trades!
        mainChart.timeScale().fitContent();
        equityChart.timeScale().fitContent();
        updateDefaultLegend();

        // Jump to the latest trade so the user immediately sees the position boxes!
        if (currentTrades.length > 0) {
            jumpToTradeIndex(currentTrades.length - 1);
        }

        addConsoleLog(`Loaded ${currentOhlc.length} real Dukascopy 5m Gold candles up to today.`, 'success');
        addConsoleLog(`Strategy Engine ready: ${currentTrades.length} trades plotted on chart.`, 'info');

    } catch (err) {
        console.error('Failed to load baseline data:', err);
        showToast('Error loading backtest data', 'error');
    } finally {
        document.getElementById('loadingOverlay').classList.add('hidden');
    }
}

// =============================================================================
// STRATEGY TESTER RENDERING (Overview, Performance, Trades)
// =============================================================================
function renderStrategyTester(stats, trades) {
    renderOverview(stats);
    renderPerformanceSummary(stats, trades);
    renderTradeList(trades);
}

function renderOverview(stats) {
    const grid = document.getElementById('metricsGrid');
    const netPnl = stats.total_pnl ?? 0;
    const pnlSign = netPnl >= 0 ? '+' : '';

    const cards = [
        { label: 'Net Profit', val: `${pnlSign}$${netPnl.toLocaleString()}`, cls: netPnl >= 0 ? 'positive' : 'negative', highlight: true },
        { label: 'Profit Factor', val: stats.profit_factor >= 999 ? '∞' : stats.profit_factor ?? 0 },
        { label: 'Win Rate', val: `${stats.win_rate ?? 0}%`, cls: stats.win_rate >= 50 ? 'positive' : '' },
        { label: 'Total Closed Trades', val: stats.total_trades ?? 0 },
        { label: 'Max Drawdown', val: `$${stats.max_drawdown?.toLocaleString() ?? 0} (${stats.max_drawdown_pct ?? 0}%)`, cls: 'negative' },
        { label: 'Sharpe Ratio', val: stats.sharpe_ratio ?? 0, cls: stats.sharpe_ratio > 1 ? 'positive' : '' },
        { label: 'Sortino Ratio', val: stats.sortino_ratio ?? 0 },
        { label: 'Avg Trade PnL', val: `${stats.avg_pnl >= 0 ? '+' : ''}$${stats.avg_pnl?.toLocaleString() ?? 0}` },
        { label: 'Win / Loss Ratio', val: stats.ratio_win_loss ?? 0 },
    ];

    grid.innerHTML = cards.map(c => `
        <div class="metric-card ${c.highlight ? 'metric-highlight' : ''}">
            <div class="metric-label">${c.label}</div>
            <div class="metric-val ${c.cls || ''}">${c.val}</div>
        </div>
    `).join('');

    document.getElementById('equityFinal').textContent = `${pnlSign}$${netPnl.toLocaleString()}`;
    document.getElementById('equityFinal').className = `equity-final ${netPnl >= 0 ? 'positive' : 'negative'}`;
}

function renderPerformanceSummary(stats, trades) {
    const tbody = document.getElementById('perfSummaryBody');
    const longs = trades.filter(t => t.direction === 'long');
    const shorts = trades.filter(t => t.direction === 'short');

    const longPnls = longs.map(t => t.pnl);
    const shortPnls = shorts.map(t => t.pnl);
    const longWins = longPnls.filter(p => p > 0);
    const shortWins = shortPnls.filter(p => p > 0);

    const rows = [
        ['Total Closed Trades', stats.total_trades ?? 0, longs.length, shorts.length],
        ['Win Rate', `${stats.win_rate ?? 0}%`, `${longs.length ? ((longWins.length / longs.length) * 100).toFixed(1) : 0}%`, `${shorts.length ? ((shortWins.length / shorts.length) * 100).toFixed(1) : 0}%`],
        ['Net Profit ($)', `$${stats.total_pnl?.toLocaleString() ?? 0}`, `$${longPnls.reduce((a, b) => a + b, 0).toFixed(2)}`, `$${shortPnls.reduce((a, b) => a + b, 0).toFixed(2)}`],
        ['Gross Profit ($)', `$${stats.gross_profit?.toLocaleString() ?? 0}`, `$${longWins.reduce((a, b) => a + b, 0).toFixed(2)}`, `$${shortWins.reduce((a, b) => a + b, 0).toFixed(2)}`],
        ['Gross Loss ($)', `$${stats.gross_loss?.toLocaleString() ?? 0}`, `—`, `—`],
        ['Profit Factor', stats.profit_factor >= 999 ? '∞' : stats.profit_factor ?? 0, '—', '—'],
        ['Max Drawdown ($)', `$${stats.max_drawdown?.toLocaleString() ?? 0}`, '—', '—'],
        ['Average Trade ($)', `$${stats.avg_pnl ?? 0}`, '—', '—'],
        ['Average Win ($)', `$${stats.avg_win ?? 0}`, '—', '—'],
        ['Average Loss ($)', `$${stats.avg_loss ?? 0}`, '—', '—'],
        ['Largest Winning Trade ($)', `$${stats.best_trade ?? 0}`, '—', '—'],
        ['Largest Losing Trade ($)', `$${stats.worst_trade ?? 0}`, '—', '—'],
        ['Max Consecutive Wins', stats.max_consecutive_wins ?? 0, '—', '—'],
        ['Max Consecutive Losses', stats.max_consecutive_losses ?? 0, '—', '—'],
    ];

    tbody.innerHTML = rows.map(r => `
        <tr>
            <td style="font-weight: 600; color: #ffffff;">${r[0]}</td>
            <td>${r[1]}</td>
            <td>${r[2]}</td>
            <td>${r[3]}</td>
        </tr>
    `).join('');
}

function renderTradeList(trades, filter = 'all') {
    const tbody = document.getElementById('tradesTableBody');
    
    document.getElementById('countAll').textContent = trades.length;
    document.getElementById('countLong').textContent = trades.filter(t => t.direction === 'long').length;
    document.getElementById('countShort').textContent = trades.filter(t => t.direction === 'short').length;
    document.getElementById('countWin').textContent = trades.filter(t => t.pnl > 0).length;
    document.getElementById('countLoss').textContent = trades.filter(t => t.pnl <= 0).length;

    let filtered = trades;
    if (filter === 'long') filtered = trades.filter(t => t.direction === 'long');
    if (filter === 'short') filtered = trades.filter(t => t.direction === 'short');
    if (filter === 'win') filtered = trades.filter(t => t.pnl > 0);
    if (filter === 'loss') filtered = trades.filter(t => t.pnl <= 0);

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="11" style="text-align:center; padding: 24px; color: var(--tv-text-muted);">No trades match filter '${filter}'</td></tr>`;
        return;
    }

    tbody.innerHTML = filtered.map(t => {
        const isWin = t.pnl > 0;
        const pnlCls = isWin ? 'positive' : 'negative';
        const pnlSign = isWin ? '+' : '';
        const entryDt = formatTradeTime(t.entry_time);
        const exitDt = formatTradeTime(t.exit_time);
        const badgeCls = t.direction === 'long' ? 'long' : 'short';
        const origIndex = currentTrades.findIndex(tr => tr.id === t.id);

        return `
            <tr onclick="jumpToTradeIndex(${origIndex})" title="Click to view Long/Short box on chart">
                <td style="color: var(--tv-text-secondary); font-weight: 700;">#${t.id}</td>
                <td><span class="trade-badge ${badgeCls}">${t.direction.toUpperCase()}</span></td>
                <td>${entryDt}</td>
                <td>$${t.entry_price.toFixed(2)}</td>
                <td>${exitDt}</td>
                <td>$${(t.exit_price || 0).toFixed(2)}</td>
                <td><span style="color: ${isWin ? 'var(--tv-green)' : 'var(--tv-red)'}">${t.exit_reason || '—'}</span></td>
                <td>${t.initial_size} oz</td>
                <td class="${pnlCls}">${pnlSign}$${t.pnl.toFixed(2)}</td>
                <td class="${pnlCls}">${pnlSign}${t.pnl_percent || 0}%</td>
                <td>$${t.cumulative_pnl?.toLocaleString() ?? 0}</td>
            </tr>
        `;
    }).join('');
}

// =============================================================================
// RUN STRATEGY & DYNAMIC BACKTEST
// =============================================================================
async function runActiveStrategy() {
    if (!monacoEditor) return;
    const code = monacoEditor.getValue();
    if (!code.trim()) {
        showToast('Strategy code cannot be empty!', 'error');
        return;
    }

    showLoading('Executing Python strategy with Anti-Lookahead AST check...');
    addConsoleLog('[Compiler] Validating AST syntax and causality rules...', 'info');

    try {
        const res = await fetch('/api/strategy/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                code: code,
                capital: 100000.0,
                lot_size: 100.0,
                spread: 0.20,
                slippage: 0.05,
            }),
        });

        const data = await res.json();

        if (!data.success) {
            if (data.error_type === 'LOOKAHEAD_BIAS') {
                showToast('🚨 STRATEGY REJECTED: Lookahead Bias Detected!', 'error');
                addConsoleLog(`[LOOKAHEAD GUARD BLOCKED]: ${data.message}`, 'error');
                alert(`🚨 STRATEGY REJECTED BY ANTI-LOOKAHEAD GUARD:\n\n${data.message}\n\nStrategies are forbidden from accessing future candles (e.g. shift(-1), center=True, bfill).`);
            } else {
                showToast(`Execution Error: ${data.message}`, 'error');
                addConsoleLog(`[ERROR]: ${data.message}\n${data.traceback || ''}`, 'error');
            }
            return;
        }

        // Update state
        currentStats = data.stats;
        currentTrades = data.trades;
        currentCode = code;

        // 1. Update Chart Markers: Only for trades taken
        currentMarkers = data.markers || [];
        updateMarkersDisplay();

        // 2. Update Equity Curve
        if (data.equity) {
            const eqData = prepareEquityData(data.equity);
            if (eqData.length) {
                equitySeries.setData(eqData);
                equityChart.timeScale().fitContent();
            }
        }

        // 3. Update Custom Indicators if present
        if (data.indicators) {
            renderCustomIndicators(data.indicators);
        }

        // 4. Update Strategy Tester & Floating Tables
        renderStrategyTester(currentStats, currentTrades);
        renderFloatingMonthlyStats(currentTrades, currentStats);
        updateTradeNavigator();
        runMonteCarloSimulation(currentTrades, true);

        // 5. Jump to latest trade and redraw boxes
        if (currentTrades.length > 0) {
            jumpToTradeIndex(currentTrades.length - 1);
        }

        // Console output
        if (data.logs) {
            addConsoleLog(`[Strategy stdout]:\n${data.logs}`, 'info');
        }
        addConsoleLog(`[Success] Backtested ${data.trades.length} trades | Profit Factor: ${data.stats.profit_factor} | Net PnL: $${data.stats.total_pnl?.toLocaleString()}`, 'success');
        showToast(`Strategy Backtest Complete: ${data.trades.length} trades, PnL: $${data.stats.total_pnl?.toLocaleString()}`, 'success');

        // Switch to Strategy Tester tab to show results
        switchDockTab('strategyTester');

    } catch (err) {
        console.error('Run strategy error:', err);
        showToast('Network / Server Error executing strategy', 'error');
    } finally {
        hideLoading();
    }
}

function renderCustomIndicators(indicatorsMap) {
    // Clean chart: remove any indicator lines so only trades and candles are shown
    Object.values(customIndicatorSeries).forEach(s => mainChart.removeSeries(s));
    customIndicatorSeries = {};
}

// =============================================================================
// AI STRATEGY GENERATION & CONTINUOUS OPTIMIZATION
// =============================================================================
async function generateAiStrategy() {
    const prompt = document.getElementById('aiPromptInput').value.trim();
    if (!prompt) {
        showToast('Please enter a strategy concept or prompt!', 'error');
        return;
    }

    const cfg = getAiConfig();
    if (!cfg.apiKey && cfg.provider !== 'omniroute') {
        openAiConfigModal();
        showToast('Please configure your AI Provider API Key first!', 'error');
        return;
    }

    showLoading('AI is designing your strategy with Hardcoded Anti-Lookahead rules...');
    addAiLog('System', `Requesting strategy from ${cfg.provider.toUpperCase()} (${cfg.model || 'default'}) via ${cfg.endpoint}...`, 'info');

    try {
        const res = await fetch('/api/ai/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: cfg.provider,
                api_key: cfg.apiKey,
                model: cfg.model,
                endpoint: cfg.endpoint,
                prompt: prompt,
            }),
        });

        const data = await res.json();
        if (!data.success) {
            showToast(`AI Generation failed: ${data.message}`, 'error');
            addAiLog('Error', data.message, 'error');
            return;
        }

        addAiLog('Guard', 'Passed AST Anti-Lookahead static validation cleanly!', 'success');
        addAiLog('Code', 'New strategy code received and loaded into Monaco Editor.', 'info');

        if (monacoEditor) {
            monacoEditor.setValue(data.code);
        }

        await runActiveStrategy();

    } catch (err) {
        console.error('AI Generate error:', err);
        showToast('Failed to call AI service', 'error');
    } finally {
        hideLoading();
    }
}

async function startContinuousOptimization() {
    const cfg = getAiConfig();
    if (!cfg.apiKey && cfg.provider !== 'omniroute') {
        openAiConfigModal();
        showToast('Please configure your AI API Key first!', 'error');
        return;
    }

    const maxIterations = parseInt(document.getElementById('optIterations').value) || 5;
    isOptimizing = true;
    stopOptimizationRequested = false;

    document.getElementById('optStatusPill').textContent = 'Optimizing...';
    document.getElementById('optStatusPill').className = 'ai-status-pill running';
    document.getElementById('btnStartOptimize').style.display = 'none';
    document.getElementById('btnStopOptimize').style.display = 'block';

    addAiLog('Loop', `Starting autonomous continuous optimization (Max ${maxIterations} iterations)...`, 'info');

    let bestScore = -999999;
    let bestStats = currentStats;
    let currentCodeForOpt = monacoEditor ? monacoEditor.getValue() : currentCode;

    for (let iter = 1; iter <= maxIterations; iter++) {
        if (stopOptimizationRequested) {
            addAiLog('Loop', 'Optimization stopped by user.', 'warn');
            break;
        }

        addAiLog(`Iteration #${iter}`, `Proposing parameter / filter mutation based on Profit Factor: ${bestStats.profit_factor}...`, 'info');

        try {
            const res = await fetch('/api/ai/optimize_step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider: cfg.provider,
                    api_key: cfg.apiKey,
                    model: cfg.model,
                    endpoint: cfg.endpoint,
                    current_code: currentCodeForOpt,
                    previous_stats: bestStats,
                    iteration: iter,
                }),
            });

            const data = await res.json();
            if (!data.success) {
                addAiLog(`Iter #${iter}`, `Mutation rejected: ${data.message}`, 'warn');
                continue;
            }

            const newScore = data.score;
            const newStats = data.stats;
            addAiLog(`Iter #${iter}`, `Tested: ${data.trades_count} trades | Win Rate: ${newStats.win_rate}% | PF: ${newStats.profit_factor} | Score: ${newScore}`, newScore > bestScore ? 'success' : 'info');

            if (newScore > bestScore) {
                bestScore = newScore;
                bestStats = newStats;
                currentCodeForOpt = data.code;
                document.getElementById('bestScoreBadge').textContent = `Best Score: ${bestScore.toFixed(1)} (PF: ${bestStats.profit_factor})`;

                if (monacoEditor) {
                    monacoEditor.setValue(data.code);
                }

                if (data.exec_result) {
                    currentStats = data.exec_result.stats;
                    currentTrades = data.exec_result.trades;
                    currentMarkers = data.exec_result.markers || [];
                    updateMarkersDisplay();
                    if (data.exec_result.equity) {
                        const eqData = prepareEquityData(data.exec_result.equity);
                        if (eqData.length) equitySeries.setData(eqData);
                    }
                    renderStrategyTester(currentStats, currentTrades);
                    renderFloatingMonthlyStats(currentTrades, currentStats);
                    updateTradeNavigator();
                    requestAnimationFrame(drawTradeBoxes);
                }
            }

        } catch (err) {
            console.error('Optimization iteration error:', err);
            addAiLog(`Iter #${iter}`, 'API communication error during optimization step.', 'error');
        }
    }

    isOptimizing = false;
    document.getElementById('optStatusPill').textContent = 'Finished';
    document.getElementById('optStatusPill').className = 'ai-status-pill';
    document.getElementById('btnStartOptimize').style.display = 'block';
    document.getElementById('btnStopOptimize').style.display = 'none';
    showToast('Autonomous Optimization Run Complete!', 'success');
}

function stopContinuousOptimization() {
    stopOptimizationRequested = true;
    addAiLog('Loop', 'Stopping optimization after current iteration...', 'warn');
}

function addAiLog(sender, message, type = 'info') {
    const stream = document.getElementById('aiLogStream');
    if (!stream) return;
    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    entry.innerHTML = `<span class="log-ts">[${sender}]</span><span class="log-msg">${message}</span>`;
    stream.appendChild(entry);
    stream.scrollTop = stream.scrollHeight;
}

function addConsoleLog(message, type = 'info') {
    const box = document.getElementById('consoleOutput');
    if (!box) return;
    const line = document.createElement('div');
    line.className = `console-line ${type}`;
    line.textContent = message;
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
}

function getAiConfig() {
    return {
        provider: localStorage.getItem('tv_ai_provider') || 'omniroute',
        apiKey: localStorage.getItem('tv_ai_apikey') || 'sk-e9b30155d949b791-9b5481-fe8fbacd',
        model: localStorage.getItem('tv_ai_model') || 'auto/best-coding',
        endpoint: localStorage.getItem('tv_ai_endpoint') || 'http://localhost:20128/v1',
    };
}

function saveAiConfig(provider, apiKey, model, endpoint) {
    localStorage.setItem('tv_ai_provider', provider);
    localStorage.setItem('tv_ai_apikey', apiKey);
    localStorage.setItem('tv_ai_model', model);
    if (endpoint !== undefined) {
        localStorage.setItem('tv_ai_endpoint', endpoint);
    }
}

function openAiConfigModal() {
    const cfg = getAiConfig();
    document.getElementById('cfgProvider').value = cfg.provider;
    document.getElementById('cfgApiKey').value = cfg.apiKey;
    document.getElementById('cfgModel').value = cfg.model;
    const endpInput = document.getElementById('cfgEndpoint');
    if (endpInput) {
        endpInput.value = cfg.endpoint;
    }
    document.getElementById('aiConfigModal').classList.add('open');
}

function closeAiConfigModal() {
    document.getElementById('aiConfigModal').classList.remove('open');
}

// =============================================================================
// MONACO EDITOR INTEGRATION
// =============================================================================
function initMonacoEditor() {
    require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs' } });
    require(['vs/editor/editor.main'], function () {
        monacoEditor = monaco.editor.create(document.getElementById('monacoEditorContainer'), {
            value: currentCode || '# Loading Elvaris River Strategy...',
            language: 'python',
            theme: 'vs-dark',
            automaticLayout: true,
            fontSize: 13,
            fontFamily: "'JetBrains Mono', Consolas, monospace",
            minimap: { enabled: false },
            lineNumbers: 'on',
            roundedSelection: true,
            scrollBeyondLastLine: false,
            padding: { top: 8, bottom: 8 },
        });

        monacoEditor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, function () {
            runActiveStrategy();
        });
    });
}

// =============================================================================
// UI NAVIGATION & CONTROLS SETUP
// =============================================================================
function setupNavigation() {
    // Dock main tabs
    document.querySelectorAll('.tv-tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            switchDockTab(tab);
        });
    });

    // Strategy Tester sub-tabs
    document.querySelectorAll('.subtab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.subtab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tester-subtab-pane').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const targetId = 'subtab' + btn.dataset.subtab.charAt(0).toUpperCase() + btn.dataset.subtab.slice(1);
            const pane = document.getElementById(targetId);
            if (pane) pane.classList.add('active');
            if (btn.dataset.subtab === 'overview' && equityChart) {
                setTimeout(() => equityChart.timeScale().fitContent(), 50);
            }
            if (btn.dataset.subtab === 'monteCarlo') {
                if (lastMonteCarloCorridors) {
                    setTimeout(() => renderMonteCarloChart(lastMonteCarloCorridors), 50);
                } else {
                    runMonteCarloSimulation(currentTrades);
                }
            }
        });
    });

    // Trade filter buttons
    document.querySelectorAll('.tr-filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tr-filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderTradeList(currentTrades, btn.dataset.trfilter);
        });
    });

    // Toggle Trade Boxes Button on Header
    const toggleBoxBtn = document.getElementById('toggleTradeBoxesBtn');
    if (toggleBoxBtn) {
        toggleBoxBtn.addEventListener('click', () => {
            showTradeBoxes = !showTradeBoxes;
            toggleBoxBtn.classList.toggle('active', showTradeBoxes);
            drawTradeBoxes();
            updateMarkersDisplay();
            showToast(`Trade Position Boxes: ${showTradeBoxes ? 'ON' : 'OFF'}`, 'info');
        });
    }

    // Trade Navigator Prev / Next Buttons
    const prevBtn = document.getElementById('tnavPrevBtn');
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            jumpToTradeIndex(activeTradeIndex - 1);
        });
    }

    const nextBtn = document.getElementById('tnavNextBtn');
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            jumpToTradeIndex(activeTradeIndex + 1);
        });
    }

    const openListBtn = document.getElementById('tnavOpenListBtn');
    if (openListBtn) {
        openListBtn.addEventListener('click', () => {
            switchDockTab('strategyTester');
            const tradesBtn = document.querySelector('.subtab-btn[data-subtab="trades"]');
            if (tradesBtn) tradesBtn.click();
        });
    }

    // Floating Stats Widget minimize/expand/close
    const fswToggle = document.getElementById('fswToggleBtn');
    if (fswToggle) {
        fswToggle.addEventListener('click', () => {
            const w = document.getElementById('tvFloatingStatsTable');
            w.classList.toggle('minimized');
            fswToggle.textContent = w.classList.contains('minimized') ? '+' : '−';
        });
    }

    const fswClose = document.getElementById('fswCloseBtn');
    if (fswClose) {
        fswClose.addEventListener('click', () => {
            const w = document.getElementById('tvFloatingStatsTable');
            if (w) w.style.display = 'none';
        });
    }

    // Split dragger resizing
    const dragger = document.getElementById('tvSplitDragger');
    const dock = document.getElementById('tvBottomDock');
    let isDragging = false;
    let startY, startHeight;

    if (dragger && dock) {
        dragger.addEventListener('mousedown', e => {
            isDragging = true;
            startY = e.clientY;
            startHeight = dock.offsetHeight;
            dragger.classList.add('dragging');
            document.body.style.cursor = 'row-resize';
            e.preventDefault();
        });

        window.addEventListener('mousemove', e => {
            if (!isDragging) return;
            const delta = startY - e.clientY;
            const newH = Math.min(Math.max(startHeight + delta, 40), window.innerHeight * 0.85);
            dock.style.height = `${newH}px`;
            const mount = document.getElementById('mainChartMount');
            if (mainChart && mount) mainChart.resize(mount.clientWidth, mount.clientHeight);
            if (equityChart && equityChartMount) equityChart.resize(equityChartMount.clientWidth, equityChartMount.clientHeight);
            syncCanvasSize();
            drawTradeBoxes();
            if (monacoEditor) monacoEditor.layout();
        });

        window.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                dragger.classList.remove('dragging');
                document.body.style.cursor = '';
            }
        });
    }

    // Collapse / Expand dock
    const collapseBtn = document.getElementById('dockCollapseBtn');
    if (collapseBtn && dock) {
        collapseBtn.addEventListener('click', () => {
            dock.classList.toggle('collapsed');
            const mount = document.getElementById('mainChartMount');
            if (mainChart && mount) mainChart.resize(mount.clientWidth, mount.clientHeight);
            syncCanvasSize();
            drawTradeBoxes();
            if (monacoEditor) monacoEditor.layout();
        });
    }

    // Reset strategy button
    const resetBtn = document.getElementById('resetStrategyBtn');
    if (resetBtn) {
        resetBtn.addEventListener('click', async () => {
            const res = await fetch('/api/strategy/default');
            const data = await res.json();
            if (data.code && monacoEditor) {
                monacoEditor.setValue(data.code);
                showToast('Reset code to Elvaris v2 baseline template', 'info');
            }
        });
    }

    // Run strategy button
    const runBtn = document.getElementById('runStrategyBtn');
    if (runBtn) {
        runBtn.addEventListener('click', runActiveStrategy);
    }

    // AI Buttons
    document.getElementById('btnAiGenerate')?.addEventListener('click', generateAiStrategy);
    document.getElementById('btnStartOptimize')?.addEventListener('click', startContinuousOptimization);
    document.getElementById('btnStopOptimize')?.addEventListener('click', stopContinuousOptimization);
    document.getElementById('openAiConfigBtn')?.addEventListener('click', openAiConfigModal);
    document.getElementById('aiConfigModalLink')?.addEventListener('click', openAiConfigModal);
    document.getElementById('closeAiModalBtn')?.addEventListener('click', closeAiConfigModal);
    document.getElementById('cancelAiModalBtn')?.addEventListener('click', closeAiConfigModal);

    document.getElementById('saveAiModalBtn')?.addEventListener('click', () => {
        saveAiConfig(
            document.getElementById('cfgProvider').value,
            document.getElementById('cfgApiKey').value.trim(),
            document.getElementById('cfgModel').value.trim(),
            document.getElementById('cfgEndpoint')?.value.trim() || 'http://localhost:20128/v1'
        );
        closeAiConfigModal();
        showToast('AI Provider Configuration Saved!', 'success');
    });

    // Indicators Modal
    document.getElementById('indicatorsBtn')?.addEventListener('click', () => {
        document.getElementById('indicatorsModal')?.classList.add('open');
    });
    document.getElementById('closeIndModalBtn')?.addEventListener('click', () => {
        document.getElementById('indicatorsModal')?.classList.remove('open');
    });
    document.getElementById('applyIndBtn')?.addEventListener('click', () => {
        const bbVis = document.getElementById('chkBB')?.checked ?? true;
        const s33Vis = document.getElementById('chkSMMA33')?.checked ?? true;
        const s144Vis = document.getElementById('chkSMMA144')?.checked ?? true;
        const rfVis = document.getElementById('chkRF')?.checked ?? true;

        if (indicatorSeries.bb_upper) indicatorSeries.bb_upper.applyOptions({ visible: bbVis });
        if (indicatorSeries.bb_lower) indicatorSeries.bb_lower.applyOptions({ visible: bbVis });
        if (indicatorSeries.bb_basis) indicatorSeries.bb_basis.applyOptions({ visible: bbVis });
        if (indicatorSeries.smma_33_high) indicatorSeries.smma_33_high.applyOptions({ visible: s33Vis });
        if (indicatorSeries.smma_33_low) indicatorSeries.smma_33_low.applyOptions({ visible: s33Vis });
        if (indicatorSeries.smma_144_high) indicatorSeries.smma_144_high.applyOptions({ visible: s144Vis });
        if (indicatorSeries.smma_144_low) indicatorSeries.smma_144_low.applyOptions({ visible: s144Vis });
        if (indicatorSeries.range_filter) indicatorSeries.range_filter.applyOptions({ visible: rfVis });

        document.getElementById('indicatorsModal')?.classList.remove('open');
        showToast('Indicator overlays updated', 'info');
    });

    // Window Resize Observer
    window.addEventListener('resize', () => {
        const mc = document.getElementById('mainChartMount');
        const ec = document.getElementById('equityChartMount');
        if (mainChart && mc) mainChart.resize(mc.clientWidth, mc.clientHeight);
        if (equityChart && ec) equityChart.resize(ec.clientWidth, ec.clientHeight);
        syncCanvasSize();
        drawTradeBoxes();
        if (monacoEditor) monacoEditor.layout();
        if (lastMonteCarloCorridors) {
            renderMonteCarloChart(lastMonteCarloCorridors);
        }
    });

    // Monte Carlo & Leaderboard action buttons
    document.getElementById('btnRerunMonteCarlo')?.addEventListener('click', () => {
        runMonteCarloSimulation(currentTrades, true);
    });
    document.getElementById('btnRefreshLeaderboard')?.addEventListener('click', fetchLeaderboard);
    document.getElementById('btnStartGenerationToLeaderboard')?.addEventListener('click', startAutonomousGeneration);

    // Multi-Agent Quant Lab listeners
    document.getElementById('btnStartResearch')?.addEventListener('click', startMultiAgentResearch);
    document.getElementById('btnStopResearch')?.addEventListener('click', stopMultiAgentResearch);
    document.getElementById('btnClearStream')?.addEventListener('click', () => {
        const stream = document.getElementById('qlStreamBody');
        if (stream) stream.innerHTML = '';
    });
}

function switchDockTab(tabId) {
    document.querySelectorAll('.tv-tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tv-tab-content').forEach(c => c.classList.remove('active'));

    const btn = document.querySelector(`.tv-tab-btn[data-tab="${tabId}"]`);
    if (btn) btn.classList.add('active');

    const targetId = 'tab' + tabId.charAt(0).toUpperCase() + tabId.slice(1);
    const content = document.getElementById(targetId);
    if (content) content.classList.add('active');

    const editorActions = document.getElementById('editorActions');
    if (tabId === 'pythonEditor') {
        if (editorActions) editorActions.style.display = 'flex';
        if (monacoEditor) {
            monacoEditor.layout();
            setTimeout(() => monacoEditor.layout(), 60);
        }
    } else {
        if (editorActions) editorActions.style.display = 'none';
    }

    if (tabId === 'strategyTester' && equityChart) {
        setTimeout(() => equityChart.timeScale().fitContent(), 50);
    }

    if (tabId === 'strategyLeaderboard') {
        fetchLeaderboard();
    }

    if (tabId === 'quantLab') {
        pollResearchStatus();
    }
}

// =============================================================================
// MONTE CARLO STRESS TEST ENGINE (UI VISUALIZER)
// =============================================================================
let lastMonteCarloCorridors = null;
let lastMcTradesCount = -1;

function renderMonteCarloChart(corridors) {
    const canvas = document.getElementById('mcCanvas');
    if (!canvas) return;
    const parent = canvas.parentElement;
    const width = parent ? parent.clientWidth : 900;
    const height = parent ? Math.max(parent.clientHeight, 260) : 260;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    if (!corridors || !corridors.p50 || corridors.p50.length === 0) {
        ctx.fillStyle = TV_COLORS.text;
        ctx.font = '13px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No Monte Carlo simulation data available', width / 2, height / 2);
        return;
    }

    const padLeft = 60;
    const padRight = 30;
    const padTop = 25;
    const padBottom = 30;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    const allVals = [
        ...corridors.p5,
        ...corridors.p25,
        ...corridors.p50,
        ...corridors.p75,
        ...corridors.p95
    ];
    let minVal = Math.min(...allVals, 0);
    let maxVal = Math.max(...allVals, 10);
    const valRange = (maxVal - minVal) || 1;
    minVal -= valRange * 0.08;
    maxVal += valRange * 0.08;

    const nPoints = corridors.p50.length;
    const getX = idx => padLeft + (idx / (nPoints - 1)) * plotW;
    const getY = val => padTop + (1 - (val - minVal) / (maxVal - minVal)) * plotH;

    // Background horizontal grid
    ctx.strokeStyle = '#1f2431';
    ctx.lineWidth = 1;
    const gridRows = 5;
    ctx.fillStyle = '#787b86';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'right';

    for (let i = 0; i <= gridRows; i++) {
        const y = padTop + (i / gridRows) * plotH;
        const v = maxVal - (i / gridRows) * (maxVal - minVal);
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(width - padRight, y);
        ctx.stroke();
        ctx.fillText(`${v >= 0 ? '+' : ''}${v.toFixed(1)} R`, padLeft - 8, y + 3);
    }

    // Zero R reference line
    if (minVal <= 0 && maxVal >= 0) {
        const yZero = getY(0);
        ctx.strokeStyle = '#ffffff30';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(padLeft, yZero);
        ctx.lineTo(width - padRight, yZero);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    function fillBand(topSeries, botSeries, fillColor) {
        ctx.fillStyle = fillColor;
        ctx.beginPath();
        for (let i = 0; i < nPoints; i++) {
            const x = getX(i);
            const y = getY(topSeries[i]);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        for (let i = nPoints - 1; i >= 0; i--) {
            const x = getX(i);
            const y = getY(botSeries[i]);
            ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fill();
    }

    function strokeCurve(series, strokeColor, lineWidth = 1.5) {
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        for (let i = 0; i < nPoints; i++) {
            const x = getX(i);
            const y = getY(series[i]);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // Shaded confidence corridors
    fillBand(corridors.p95, corridors.p75, 'rgba(8, 153, 129, 0.08)');
    fillBand(corridors.p75, corridors.p25, 'rgba(41, 98, 255, 0.08)');
    fillBand(corridors.p25, corridors.p5, 'rgba(242, 54, 69, 0.08)');

    // Stroke percentile fan lines
    strokeCurve(corridors.p95, '#089981', 1.5);
    strokeCurve(corridors.p75, '#00bcd4', 1.2);
    strokeCurve(corridors.p50, '#2962ff', 2.5); // 50th median path
    strokeCurve(corridors.p25, '#f6a623', 1.2);
    strokeCurve(corridors.p5, '#f23645', 1.5);

    // Axis label
    ctx.fillStyle = '#787b86';
    ctx.textAlign = 'center';
    ctx.fillText('Bootstrap Trade Progression (1,000 Reshuffled Sequences)', padLeft + plotW / 2, height - 8);
}

async function runMonteCarloSimulation(trades, force = false) {
    if (!trades || trades.length < 2) return;
    if (!force && trades.length === lastMcTradesCount) return;

    try {
        const res = await fetch('/api/monte_carlo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                trades: trades,
                simulations: 1000
            })
        });
        const data = await res.json();
        if (!data.success || !data.monte_carlo) return;

        lastMcTradesCount = trades.length;
        const mc = data.monte_carlo;
        lastMonteCarloCorridors = mc.corridors;

        const varEl = document.getElementById('mcVar95');
        if (varEl) varEl.textContent = `-${mc.var_95_max_dd_r} R`;

        const varUsdEl = document.getElementById('mcVar95Usd');
        if (varUsdEl) varUsdEl.textContent = `-$${mc.var_95_max_dd_usd?.toLocaleString()} (95% VaR)`;

        const expDdEl = document.getElementById('mcExpectedDd');
        if (expDdEl) expDdEl.textContent = `-${mc.expected_max_dd_r} R`;

        const expDdUsdEl = document.getElementById('mcExpectedDdUsd');
        if (expDdUsdEl) expDdUsdEl.textContent = `-$${mc.expected_max_dd_usd?.toLocaleString()} (mean)`;

        const r10El = document.getElementById('mcRisk10r');
        if (r10El) r10El.textContent = `${mc.risk_of_ruin_10r}%`;

        const r20El = document.getElementById('mcRisk20pct');
        if (r20El) r20El.textContent = `${mc.risk_of_ruin_20pct}%`;

        if (mc.corridors) {
            renderMonteCarloChart(mc.corridors);
        }
    } catch (err) {
        console.error('Monte Carlo simulation failed:', err);
    }
}

// =============================================================================
// STRATEGY LEADERBOARD & PERSISTENCE
// =============================================================================
async function fetchLeaderboard() {
    try {
        const res = await fetch('/api/leaderboard');
        const data = await res.json();
        if (!data.success || !data.leaderboard) return;
        renderLeaderboardTable(data.leaderboard);
    } catch (err) {
        console.error('Failed to fetch leaderboard:', err);
    }
}

function renderLeaderboardTable(items) {
    const tbody = document.getElementById('leaderboardTableBody');
    if (!tbody) return;

    if (!items || items.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 30px; color: var(--tv-text-muted);">No strategies ranked yet. Click 'Start Autonomous Strategy Generation' to generate!</td></tr>`;
        return;
    }

    tbody.innerHTML = items.map((item, idx) => {
        const rank = idx + 1;
        let rankBadge = `<span class="lb-rank-normal">#${rank}</span>`;
        if (rank === 1) rankBadge = `<span class="lb-rank-badge rank-1">🥇 1</span>`;
        else if (rank === 2) rankBadge = `<span class="lb-rank-badge rank-2">🥈 2</span>`;
        else if (rank === 3) rankBadge = `<span class="lb-rank-badge rank-3">🥉 3</span>`;

        const totalR = item.total_r ?? 0;
        const totalRClass = totalR >= 0 ? 'positive' : 'negative';
        const totalRSign = totalR >= 0 ? '+' : '';

        const monthly = item.monthly_r || {};
        const monthsCount10 = item.months_ge_10r || 0;
        let monthlyHtml = '';
        if (monthsCount10 > 0) {
            monthlyHtml += `<div class="lb-months-headline">⭐ <strong>${monthsCount10} Months ≥ +10R</strong></div>`;
        }
        monthlyHtml += `<div class="lb-monthly-pills">`;
        for (const [mName, mVal] of Object.entries(monthly)) {
            const isHigh = mVal >= 10.0;
            const pClass = isHigh ? 'high-yield' : (mVal > 0 ? 'positive' : 'negative');
            const sign = mVal > 0 ? '+' : '';
            monthlyHtml += `<span class="monthly-pill ${pClass}" title="${mName}: ${sign}${mVal}R">${mName.slice(0, 3)}: ${sign}${mVal}R</span>`;
        }
        monthlyHtml += `</div>`;

        const maxDdR = item.max_drawdown_r ?? 0;
        const maxDdPct = item.max_drawdown_pct ?? 0;

        const mcVar = item.monte_carlo?.var_95_max_dd_r ? `-${item.monte_carlo.var_95_max_dd_r} R` : '—';
        const mcProb = item.monte_carlo?.probability_of_profit ? `(${item.monte_carlo.probability_of_profit}% prob)` : '';

        return `
            <tr>
                <td style="text-align: center;">${rankBadge}</td>
                <td>
                    <div class="lb-strat-name">${item.name}</div>
                    <div class="lb-strat-concept">${item.concept || ''}</div>
                    <div class="lb-strat-badges">
                        <span class="lb-tag tag-guard">AST Anti-Lookahead</span>
                        <span class="lb-tag tag-friction">Dukascopy Friction</span>
                        ${rank === 1 && totalR > 0 ? `<span class="lb-tag tag-champ">CHAMPION (${totalRSign}${totalR}R)</span>` : ''}
                    </div>
                </td>
                <td style="text-align: right;">
                    <span class="lb-total-r ${totalRClass}">${totalRSign}${totalR}R</span>
                </td>
                <td>
                    ${monthlyHtml}
                </td>
                <td style="text-align: right;">
                    <span class="negative font-semibold">-${maxDdR} R</span>
                    <div class="lb-sub-val">(${maxDdPct.toFixed(1)}%)</div>
                </td>
                <td style="text-align: right;">
                    <span class="${item.win_rate >= 50 ? 'positive' : ''}">${item.win_rate}%</span>
                    <div class="lb-sub-val">${item.winning_trades || 0}W / ${item.losing_trades || 0}L</div>
                </td>
                <td style="text-align: right;">
                    <span class="${item.profit_factor >= 1.2 ? 'positive' : ''}">${item.profit_factor >= 999 ? '∞' : item.profit_factor}</span>
                </td>
                <td style="text-align: right;">
                    <span class="negative">${mcVar}</span>
                    <div class="lb-sub-val">${mcProb}</div>
                </td>
                <td style="text-align: center;">
                    <button class="lb-btn-load" onclick="loadLeaderboardStrategy('${item.id}')">
                        ⚡ Load Strategy
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

async function loadLeaderboardStrategy(stratId) {
    showLoading('Loading strategy from Leaderboard into Chart & Python Editor...');
    try {
        const res = await fetch('/api/leaderboard/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: stratId })
        });
        const data = await res.json();
        if (!data.success) {
            showToast(`Failed to load strategy: ${data.message}`, 'error');
            return;
        }

        if (monacoEditor && data.code) {
            monacoEditor.setValue(data.code);
        }
        currentCode = data.code || '';

        if (data.exec_result && data.exec_result.success) {
            currentStats = data.exec_result.stats;
            currentTrades = data.exec_result.trades;
            currentMarkers = data.exec_result.markers || [];
            updateMarkersDisplay();

            if (data.exec_result.equity) {
                const eqData = prepareEquityData(data.exec_result.equity);
                if (eqData.length) {
                    equitySeries.setData(eqData);
                    equityChart.timeScale().fitContent();
                }
            }

            renderStrategyTester(currentStats, currentTrades);
            renderFloatingMonthlyStats(currentTrades, currentStats);
            updateTradeNavigator();

            if (currentTrades.length > 0) {
                jumpToTradeIndex(currentTrades.length - 1);
            }
            requestAnimationFrame(drawTradeBoxes);
            runMonteCarloSimulation(currentTrades, true);
        }

        showToast(`Loaded "${data.strategy?.name}" into Chart!`, 'success');
        addConsoleLog(`[Leaderboard] Loaded strategy: "${data.strategy?.name}". Total Return: +${data.strategy?.total_r}R`, 'success');

        switchDockTab('strategyTester');

    } catch (err) {
        console.error('Error loading strategy:', err);
        showToast('Error loading strategy from leaderboard', 'error');
    } finally {
        hideLoading();
    }
}
window.loadLeaderboardStrategy = loadLeaderboardStrategy;

async function startAutonomousGeneration() {
    const cfg = getAiConfig();
    if (!cfg.apiKey && cfg.provider !== 'omniroute') {
        openAiConfigModal();
        showToast('Please configure your AI Provider API Key first!', 'error');
        return;
    }

    showLoading('Autonomous AI is researching and generating a new institutional quantitative strategy...');
    addConsoleLog('[Autonomous Loop] Initiating strategy generation via OmniRoute with zero-lookahead AST guard & Monte Carlo stress test...', 'info');

    const promptList = [
        "Create an institutional Fair Value Gap (FVG) and Liquidity Sweep Strategy on 5m Gold with Dynamic ATR Stop Loss and strict R:R target.",
        "Create an Order Block & Break of Structure (BOS) Trend Continuation Strategy on 5m Gold with multi-timeframe session volume filters.",
        "Create a Volume Imbalance Mean Reversion Strategy on 5m Gold with ATR volatility bands and time-decay expiration."
    ];
    const prompt = promptList[Math.floor(Math.random() * promptList.length)];

    try {
        const res = await fetch('/api/ai/start_generation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: cfg.provider,
                api_key: cfg.apiKey,
                model: cfg.model,
                endpoint: cfg.endpoint,
                prompt: prompt
            })
        });
        const data = await res.json().catch(() => ({ success: false, message: `HTTP ${res.status}: ${res.statusText}` }));
        if (!res.ok || !data.success) {
            const msg = data.message || `Server Error (${res.status})`;
            showToast(`Generation failed: ${msg}`, 'error');
            addConsoleLog(`[Generation Error]: ${msg}`, 'error');
            return;
        }

        const strat = data.strategy;
        showToast(`🎉 New Strategy Generated: "${strat.name}" (+${strat.total_r}R, Rank #${strat.rank})!`, 'success');
        addConsoleLog(`[Leaderboard Added] Strategy "${strat.name}" evaluated: +${strat.total_r}R, ${strat.months_ge_10r} months >= 10R, Rank #${strat.rank}`, 'success');

        await fetchLeaderboard();
        switchDockTab('strategyLeaderboard');

    } catch (err) {
        console.error('Autonomous generation error:', err);
        showToast('Error calling autonomous generation service', 'error');
    } finally {
        hideLoading();
    }
}

function showLoading(text) {
    const el = document.getElementById('loadingText');
    if (el) el.textContent = text;
    document.getElementById('loadingOverlay')?.classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingOverlay')?.classList.add('hidden');
}

// =============================================================================
// MULTI-AGENT QUANT LAB & AUTONOMOUS RESEARCH ENGINE
// =============================================================================
let researchPollTimer = null;

async function startMultiAgentResearch() {
    const roundsInput = document.getElementById('qlRoundsInput');
    const rounds = roundsInput ? parseInt(roundsInput.value, 10) || 100 : 100;

    const btnStart = document.getElementById('btnStartResearch');
    const btnStop = document.getElementById('btnStopResearch');
    if (btnStart) btnStart.disabled = true;
    if (btnStop) btnStop.disabled = false;

    showToast(`🧬 Starting Multi-Agent Quant Lab (${rounds} Rounds)...`, 'info');

    try {
        const res = await fetch('/api/research/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rounds: rounds,
                provider: 'omniroute',
                endpoint: 'http://localhost:20128/v1'
            })
        });
        const data = await res.json().catch(() => ({ success: false }));
        if (!res.ok || !data.success) {
            showToast(data.message || 'Failed to start research loop', 'error');
            if (btnStart) btnStart.disabled = false;
            if (btnStop) btnStop.disabled = true;
            return;
        }

        // Start active polling
        if (!researchPollTimer) {
            researchPollTimer = setInterval(pollResearchStatus, 1500);
        }
        pollResearchStatus();

    } catch (err) {
        console.error('Failed to start research loop:', err);
        showToast('Network error starting research loop', 'error');
        if (btnStart) btnStart.disabled = false;
        if (btnStop) btnStop.disabled = true;
    }
}

async function stopMultiAgentResearch() {
    const btnStop = document.getElementById('btnStopResearch');
    if (btnStop) btnStop.disabled = true;

    try {
        await fetch('/api/research/stop', { method: 'POST' });
        showToast('Stop signal sent to Multi-Agent Lab', 'warning');
        pollResearchStatus();
    } catch (err) {
        console.error('Failed to stop research loop:', err);
    }
}

async function pollResearchStatus() {
    try {
        const res = await fetch('/api/research/status');
        if (!res.ok) return;
        const data = await res.json();

        // 1. Status Pill & Buttons
        const pill = document.getElementById('qlStatusPill');
        const badge = document.getElementById('labRunningBadge');
        const btnStart = document.getElementById('btnStartResearch');
        const btnStop = document.getElementById('btnStopResearch');

        if (pill) {
            pill.className = `ql-status-pill ${data.status}`;
            pill.textContent = data.status.toUpperCase();
        }

        if (badge) {
            badge.style.display = (data.status === 'running') ? 'inline-block' : 'none';
        }

        if (data.status === 'running') {
            if (btnStart) btnStart.disabled = true;
            if (btnStop) btnStop.disabled = false;
        } else if (data.status === 'stopping') {
            if (btnStart) btnStart.disabled = true;
            if (btnStop) btnStop.disabled = true;
        } else {
            if (btnStart) btnStart.disabled = false;
            if (btnStop) btnStop.disabled = true;
            if (researchPollTimer) {
                clearInterval(researchPollTimer);
                researchPollTimer = null;
            }
        }

        // 2. Telemetry
        const hypEl = document.getElementById('qlHypothesisVal');
        if (hypEl) hypEl.textContent = data.current_hypothesis || (data.status === 'running' ? 'Mining combinatorial alpha...' : 'Ready to explore');

        const progEl = document.getElementById('qlProgressVal');
        if (progEl) progEl.textContent = `Round ${data.current_round} of ${data.max_rounds}`;

        const candEl = document.getElementById('qlCandidatesVal');
        if (candEl) candEl.textContent = data.total_candidates;

        const bestEl = document.getElementById('qlBestVal');
        if (bestEl) {
            if (data.best_candidate) {
                const b = data.best_candidate;
                const sign = b.total_r > 0 ? '+' : '';
                bestEl.textContent = `${b.name} (${sign}${b.total_r}R, ${b.months_ge_10r || 0} mos >= 10R)`;
            } else {
                bestEl.textContent = '—';
            }
        }

        // 3. Agent Cards Highlight
        const active = data.active_agent || 'Idle';
        document.querySelectorAll('.agent-card').forEach(c => c.classList.remove('active'));

        const cardMap = {
            'Idea Generator': 'agentCardIdea',
            'Risk Officer': 'agentCardRisk',
            'Backtester': 'agentCardBacktest',
            'Critic Post-Mortem': 'agentCardCritic',
            'Optimizer': 'agentCardOptimizer'
        };

        if (cardMap[active]) {
            document.getElementById(cardMap[active])?.classList.add('active');
        }

        // Update detail strings
        const ideaDet = document.getElementById('agentIdeaDetail');
        if (ideaDet && active === 'Idea Generator') {
            ideaDet.textContent = 'Mining alpha blueprint...';
        }
        const riskDet = document.getElementById('agentRiskDetail');
        if (riskDet && active === 'Risk Officer') {
            riskDet.textContent = 'Auditing AST & SL/TP...';
        }
        const btDet = document.getElementById('agentBacktestDetail');
        if (btDet && active === 'Backtester') {
            btDet.textContent = 'Simulating 34,744 bars...';
        }
        const critDet = document.getElementById('agentCriticDetail');
        if (critDet && active === 'Critic Post-Mortem') {
            critDet.textContent = 'Clustering worst losses...';
        }
        const optDet = document.getElementById('agentOptimizerDetail');
        if (optDet && active === 'Optimizer') {
            optDet.textContent = 'Targeting >= 10R yield...';
        }

        // 4. Render Logs
        const stream = document.getElementById('qlStreamBody');
        if (stream && data.recent_logs && data.recent_logs.length > 0) {
            stream.innerHTML = data.recent_logs.map(l => {
                const lvlClass = l.level || 'info';
                return `
                    <div class="ql-log-row ${lvlClass}">
                        <span class="ql-log-time">[${l.timestamp}]</span>
                        <span class="ql-log-agent">${l.agent}:</span>
                        <span class="ql-log-msg">${l.message}</span>
                    </div>
                `;
            }).join('');
            stream.scrollTop = stream.scrollHeight;
        }

    } catch (err) {
        console.error('Error polling research status:', err);
    }
}

// =============================================================================
// BOOTSTRAP
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
    try { initMonacoEditor(); } catch (e) { console.error('Monaco boot error:', e); }
    try { initCharts(); } catch (e) { console.error('Charts boot error:', e); }
    try { setupNavigation(); } catch (e) { console.error('Navigation boot error:', e); }
    try { loadBaselineData(); } catch (e) { console.error('Data boot error:', e); }
});

