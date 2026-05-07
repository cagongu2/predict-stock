import { useEffect, useState } from 'react';
import axios from 'axios';
import { RefreshCw, TrendingUp, Activity, BarChart3, ChevronRight, Sparkles, X, TrendingDown, Minus, Zap } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import './App.css';

const API_URL = 'http://127.0.0.1:8000/api';

interface StockData {
  metrics?: { mse: number; mape: number };
  predictions?: { time: string; y_true: number; y_pred: number }[];
  future?: { time: string; predicted_close: number }[];
}

function App() {
  const [stocks, setStocks] = useState<string[]>([]);
  const [selectedStock, setSelectedStock] = useState<string | null>(null);
  const [data, setData] = useState<StockData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [crawling, setCrawling] = useState<boolean>(false);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [showFutureModal, setShowFutureModal] = useState<boolean>(false);

  useEffect(() => {
    fetchStocks();
  }, []);

  useEffect(() => {
    if (selectedStock) {
      fetchStockData(selectedStock);
      setShowFutureModal(false); // close modal on stock change
    }
  }, [selectedStock]);

  const fetchStocks = async () => {
    try {
      const res = await axios.get(`${API_URL}/stocks`);
      setStocks(res.data);
      if (res.data.length > 0 && !selectedStock) {
        setSelectedStock(res.data[0]);
      }
    } catch (error) {
      console.error('Error fetching stocks:', error);
    }
  };

  const fetchStockData = async (code: string) => {
    setLoading(true);
    setData(null);
    try {
      const res = await axios.get(`${API_URL}/stocks/${code}?t=${new Date().getTime()}`);
      setData(res.data);
    } catch (error) {
      console.error('Error fetching stock data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!selectedStock) return;
    setPredicting(true);
    try {
      const res = await axios.post(`${API_URL}/stocks/${selectedStock}/predict`);
      // Update data with new future predictions from response
      setData(prev => prev ? { ...prev, future: res.data.future } : prev);
      setShowFutureModal(true); // Auto-open modal to show results
    } catch (error) {
      console.error('Error predicting:', error);
      alert('Có lỗi xảy ra khi chạy dự báo.');
    } finally {
      setPredicting(false);
    }
  };

  const handleCrawl = async () => {
    setCrawling(true);
    try {
      await axios.post(`${API_URL}/crawl`);
      alert('Cập nhật dữ liệu thành công!');
      fetchStocks();
      if (selectedStock) {
        fetchStockData(selectedStock);
      }
    } catch (error) {
      console.error('Error crawling data:', error);
      alert('Có lỗi xảy ra khi cập nhật dữ liệu.');
    } finally {
      setCrawling(false);
    }
  };

  // Prepare chart data (only test predictions, no future in chart)
  let chartData: any[] = [];
  let lastActualDate = '';

  if (data) {
    const preds = data.predictions || [];

    chartData = preds.map(p => ({
      time: p.time.split(' ')[0],
      'Thực tế': p.y_true,
      'Dự đoán (Test)': p.y_pred,
    })).slice(-60);

    if (chartData.length > 0) {
      lastActualDate = chartData[chartData.length - 1].time;
    }
  }

  const formatPercent = (val: number) => `${(val * 100).toFixed(2)}%`;
  const formatNumber = (val: number) => val.toFixed(2);

  // Future modal data
  const futureData = data?.future || [];
  const hasFuture = futureData.length > 0;

  // Calculate trend for each future row
  const getFutureTrend = (idx: number) => {
    if (idx === 0) {
      // Compare first future with last actual
      const lastActual = data?.predictions?.slice(-1)[0]?.y_pred;
      if (!lastActual) return 'neutral';
      return futureData[0].predicted_close > lastActual ? 'up' : futureData[0].predicted_close < lastActual ? 'down' : 'neutral';
    }
    const prev = futureData[idx - 1].predicted_close;
    const cur = futureData[idx].predicted_close;
    return cur > prev ? 'up' : cur < prev ? 'down' : 'neutral';
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <TrendingUp color="#60a5fa" size={28} />
          <h1>VN30 Predictor</h1>
        </div>
        <div className="stock-list">
          {stocks.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: 20 }}>
              Chưa có dữ liệu
            </div>
          ) : (
            stocks.map(code => (
              <button
                key={code}
                className={`stock-btn ${selectedStock === code ? 'active' : ''}`}
                onClick={() => setSelectedStock(code)}
              >
                {code}
                {selectedStock === code && <ChevronRight size={18} />}
              </button>
            ))
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="topbar">
          <h2>Tổng quan: {selectedStock || '...'}</h2>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            {data && hasFuture && (
              <button
                className="btn-future"
                onClick={() => setShowFutureModal(true)}
                id="btn-show-future"
              >
                <Sparkles size={18} />
                Dự báo Tương lai
              </button>
            )}
            <button className="btn-predict" onClick={handlePredict} disabled={predicting || !selectedStock} id="btn-run-predict">
              <Zap size={18} className={predicting ? 'spin' : ''} />
              {predicting ? 'Đang dự báo...' : hasFuture ? 'Cập nhật Dự báo' : 'Tạo Dự báo'}
            </button>
            <button className="btn-sync" onClick={handleCrawl} disabled={crawling}>
              <RefreshCw size={18} className={crawling ? 'spin' : ''} />
              {crawling ? 'Đang cập nhật...' : 'Cập nhật dữ liệu (Crawl)'}
            </button>
          </div>
        </header>

        {loading ? (
          <div className="loader-container">
            <RefreshCw size={40} className="spin" color="#3b82f6" />
          </div>
        ) : !data ? (
          <div className="empty-state">
            <BarChart3 size={64} opacity={0.5} />
            <p>Chọn một mã cổ phiếu để xem dữ liệu</p>
          </div>
        ) : (
          <div className="dashboard">
            {/* Metrics */}
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Mean Squared Error (MSE)</div>
                <div className="metric-value">
                  {data.metrics ? formatNumber(data.metrics.mse) : 'N/A'}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-label">MAPE</div>
                <div className="metric-value" style={{ color: 'var(--success)' }}>
                  {data.metrics ? formatPercent(data.metrics.mape) : 'N/A'}
                </div>
              </div>
              <div className="metric-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <div className="metric-label">Trạng thái</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-main)', fontWeight: 500 }}>
                  <Activity size={20} color="var(--success)" />
                  Mô hình đã huấn luyện
                </div>
              </div>
              {hasFuture && (
                <div className="metric-card future-card" style={{ cursor: 'pointer' }} onClick={() => setShowFutureModal(true)}>
                  <div className="metric-label">Dự báo Tương lai</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#fb923c', fontWeight: 600, fontSize: '1.1rem' }}>
                    <Sparkles size={20} color="#fb923c" />
                    {futureData.length} ngày sắp tới
                  </div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: 6 }}>
                    {futureData[0]?.time.split(' ')[0]} → {futureData[futureData.length - 1]?.time.split(' ')[0]}
                  </div>
                </div>
              )}
            </div>

            {/* Chart */}
            <div className="chart-container">
              <div className="chart-title">
                Biểu đồ giá đóng cửa & Dự báo (60 ngày gần nhất)
              </div>
              <div style={{ flex: 1, width: '100%', minHeight: 0 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart key={selectedStock || 'none'} data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis
                      dataKey="time"
                      stroke="#94a3b8"
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                      tickMargin={10}
                    />
                    <YAxis
                      stroke="#94a3b8"
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                      domain={['auto', 'auto']}
                      tickFormatter={(val) => val.toFixed(1)}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                      itemStyle={{ color: '#f8fafc' }}
                      labelStyle={{ color: '#94a3b8', marginBottom: '8px' }}
                    />
                    <Legend wrapperStyle={{ paddingTop: '20px' }} />
                    <Line
                      type="monotone"
                      dataKey="Thực tế"
                      stroke="#60a5fa"
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 6 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="Dự đoán (Test)"
                      stroke="#a78bfa"
                      strokeWidth={2}
                      dot={false}
                    />
                    {lastActualDate && (
                      <ReferenceLine x={lastActualDate} stroke="#64748b" strokeDasharray="3 3" />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Data Table */}
            <div className="table-container">
              <div className="table-title">Dữ liệu chi tiết</div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Ngày</th>
                    <th>Giá Thực tế</th>
                    <th>Dự đoán (Test)</th>
                  </tr>
                </thead>
                <tbody>
                  {[...chartData].reverse().map((row, idx) => (
                    <tr key={idx}>
                      <td>{row.time}</td>
                      <td>{row['Thực tế'] ? formatNumber(row['Thực tế']) : '-'}</td>
                      <td>{row['Dự đoán (Test)'] ? formatNumber(row['Dự đoán (Test)']) : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>

      {/* Future Prediction Modal */}
      {showFutureModal && (
        <div className="modal-overlay" onClick={() => setShowFutureModal(false)}>
          <div className="modal-card" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <Sparkles size={24} color="#fb923c" />
                <div>
                  <div className="modal-title">Dự báo Tương lai</div>
                  <div className="modal-subtitle">{selectedStock} — {futureData.length} ngày làm việc tiếp theo</div>
                </div>
              </div>
              <button className="modal-close" onClick={() => setShowFutureModal(false)}>
                <X size={20} />
              </button>
            </div>

            <div className="future-table-wrapper">
              <table className="future-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Ngày</th>
                    <th>Giá Dự báo (VND nghìn)</th>
                    <th>Xu hướng</th>
                  </tr>
                </thead>
                <tbody>
                  {futureData.map((row, idx) => {
                    const trend = getFutureTrend(idx);
                    const prev = idx === 0
                      ? data?.predictions?.slice(-1)[0]?.y_pred
                      : futureData[idx - 1].predicted_close;
                    const change = prev ? ((row.predicted_close - prev) / prev * 100) : 0;
                    return (
                      <tr key={idx} className={`future-row future-row-${trend}`}>
                        <td style={{ color: 'var(--text-muted)', fontWeight: 500 }}>{idx + 1}</td>
                        <td style={{ fontWeight: 600 }}>{row.time.split(' ')[0]}</td>
                        <td style={{ fontWeight: 700, fontSize: '1.1rem' }}>
                          {formatNumber(row.predicted_close)}
                        </td>
                        <td>
                          <div className="trend-badge" data-trend={trend}>
                            {trend === 'up' && <><TrendingUp size={14} /> +{change.toFixed(2)}%</>}
                            {trend === 'down' && <><TrendingDown size={14} /> {change.toFixed(2)}%</>}
                            {trend === 'neutral' && <><Minus size={14} /> 0.00%</>}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="modal-footer">
              <span>⚠️ Dự báo mang tính tham khảo, không phải khuyến nghị đầu tư</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
