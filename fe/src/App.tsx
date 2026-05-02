import { useEffect, useState } from 'react';
import axios from 'axios';
import { RefreshCw, TrendingUp, Activity, BarChart3, ChevronRight } from 'lucide-react';
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

  useEffect(() => {
    fetchStocks();
  }, []);

  useEffect(() => {
    if (selectedStock) {
      fetchStockData(selectedStock);
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
    try {
      const res = await axios.get(`${API_URL}/stocks/${code}`);
      setData(res.data);
    } catch (error) {
      console.error('Error fetching stock data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCrawl = async () => {
    setCrawling(true);
    try {
      await axios.post(`${API_URL}/crawl`);
      alert('Cập nhật dữ liệu thành công!');
      fetchStocks(); // refresh the list
    } catch (error) {
      console.error('Error crawling data:', error);
      alert('Có lỗi xảy ra khi cập nhật dữ liệu.');
    } finally {
      setCrawling(false);
    }
  };

  // Prepare chart data
  let chartData: any[] = [];
  let lastActualDate = '';

  if (data) {
    const preds = data.predictions || [];
    const futures = data.future || [];

    // Map predictions
    const mappedPreds = preds.map(p => ({
      time: p.time.split(' ')[0], // just in case there's time
      'Thực tế': p.y_true,
      'Dự đoán (Test)': p.y_pred,
      'Dự đoán (Tương lai)': null
    }));

    if (mappedPreds.length > 0) {
      lastActualDate = mappedPreds[mappedPreds.length - 1].time;
    }

    // Map futures
    const mappedFutures = futures.map(f => ({
      time: f.time.split(' ')[0],
      'Thực tế': null,
      'Dự đoán (Test)': null,
    }));

    chartData = [...mappedPreds.slice(-60), ...mappedFutures]; // Show last 60 days
  }

  const formatPercent = (val: number) => `${(val * 100).toFixed(2)}%`;
  const formatNumber = (val: number) => val.toFixed(4);

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
          <button className="btn-sync" onClick={handleCrawl} disabled={crawling}>
            <RefreshCw size={18} className={crawling ? 'spin' : ''} />
            {crawling ? 'Đang cập nhật...' : 'Cập nhật dữ liệu (Crawl)'}
          </button>
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
            </div>

            {/* Chart */}
            <div className="chart-container">
              <div className="chart-title">
                Biểu đồ giá đóng cửa & Dự báo (60 ngày gần nhất)
              </div>
              <div style={{ flex: 1, width: '100%', minHeight: 0 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
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
                    <Line
                      type="monotone"
                      dataKey="Dự đoán (Tương lai)"
                      stroke="#fb923c"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={{ r: 4 }}
                      activeDot={{ r: 6 }}
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
    </div>
  );
}

export default App;
