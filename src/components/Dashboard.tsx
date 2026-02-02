import React, { useState, useMemo, useEffect, useRef } from 'react';
import {
    TrendingUp,
    BarChart3,
    Download,
    Play,
    Settings,
    Info,
    AlertCircle,
    Calculator,
    Target,
    ChevronDown,
    ChevronUp,
    FileText
} from 'lucide-react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    AreaChart,
    Area,
    ReferenceLine,
    Label
} from 'recharts';
import { runSimulation, formatBRL, parseBRL, calculateFaixas } from '../utils/simulation';
import type { SimulationResults } from '../utils/simulation';
import * as XLSX from 'xlsx';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

const Dashboard: React.FC = () => {
    const [itemName, setItemName] = useState('Negociação Alpha');
    const [pisoText, setPisoText] = useState('2.000.000,00');
    const [provavelText, setProvavelText] = useState('2.500.000,00');
    const [tetoText, setTetoText] = useState('3.500.000,00');
    const [iterations, setIterations] = useState(200000);
    const [manualLimits, setManualLimits] = useState('');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [results, setResults] = useState<SimulationResults | null>(null);
    const [isExporting, setIsExporting] = useState(false);

    const dashboardRef = useRef<HTMLDivElement>(null);

    const stats = useMemo(() => {
        if (!results) return null;

        const { values, mean, median, p95 } = results;
        const min = values[0];
        const max = values[values.length - 1];

        // Histograma
        const numBins = 40;
        const binSize = (max - min) / numBins;
        const bins = Array.from({ length: numBins }, (_, i) => ({
            range: min + i * binSize,
            count: 0
        }));

        for (let i = 0; i < values.length; i++) {
            const v = values[i];
            const idx = Math.min(Math.floor((v - min) / binSize), numBins - 1);
            if (idx >= 0) bins[idx].count++;
        }

        // CDF - Amostragem inteligente para curva S real
        const cdfData = [];
        const pointsCount = 100;
        for (let i = 0; i <= pointsCount; i++) {
            const pct = i / pointsCount;
            const idx = Math.min(Math.floor(pct * (values.length - 1)), values.length - 1);
            cdfData.push({
                value: values[idx],
                probability: pct
            });
        }

        // Faixas
        let limits: number[] = [];
        if (manualLimits.trim()) {
            limits = manualLimits.split(',')
                .map(s => parseBRL(s.trim()))
                .filter(n => n > 0);
        }

        if (limits.length === 0) {
            // Automático: 8 faixas
            limits = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875].map(p =>
                min + p * (max - min)
            );
        }

        const faixas = calculateFaixas(values, limits);

        return { bins, cdfData, faixas, mean, median, p95, duration: results.duration };
    }, [results, manualLimits]);

    const handleSimulate = () => {
        const p = parseBRL(pisoText);
        const m = parseBRL(provavelText);
        const t = parseBRL(tetoText);

        if (p <= m && m <= t) {
            const res = runSimulation(p, m, t, iterations);
            setResults(res);
        }
    };

    useEffect(() => {
        handleSimulate();
    }, []);

    const exportExcel = () => {
        if (!results) return;
        const ws = XLSX.utils.json_to_sheet(results.values.map(v => ({ Valor: v })));
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, "Simulações");
        XLSX.writeFile(wb, `exxata_monte_carlo_${new Date().getTime()}.xlsx`);
    };

    const exportPDF = async () => {
        if (!dashboardRef.current || isExporting) return;

        setIsExporting(true);
        try {
            await new Promise(resolve => setTimeout(resolve, 500));

            const canvas = await html2canvas(dashboardRef.current, {
                scale: 1, // Escala 1 para máxima compatibilidade local e remota
                useCORS: true,
                backgroundColor: '#020617',
                onclone: (clonedDoc) => {
                    // Limpeza radical de CSS moderno que o html2canvas não suporta
                    const style = clonedDoc.createElement('style');
                    style.innerHTML = `
                        * { 
                            box-shadow: none !important; 
                            text-shadow: none !important; 
                            backdrop-filter: none !important; 
                            -webkit-backdrop-filter: none !important;
                            transition: none !important;
                            animation: none !important;
                            /* Forçar fallback de cores oklch para HEX */
                            color: #f8fafc !important; 
                        }
                        .glass-card { 
                            background: #1e293b !important; 
                            border: 1px solid #334155 !important;
                        }
                        h2, h3, h4, .text-white { color: #ffffff !important; }
                        .text-slate-400 { color: #94a3b8 !important; }
                        .text-exxata-blue { color: #4284D7 !important; }
                        .bg-exxata-red { background-color: #D51D07 !important; }
                    `;
                    clonedDoc.head.appendChild(style);
                }
            });

            const imgData = canvas.toDataURL('image/jpeg', 0.9);
            const pdf = new jsPDF('p', 'mm', 'a4');
            const pageWidth = pdf.internal.pageSize.getWidth();
            const imgHeight = (canvas.height * pageWidth) / canvas.width;

            pdf.addImage(imgData, 'JPEG', 0, 0, pageWidth, imgHeight, undefined, 'FAST');
            pdf.save(`exxata_relatorio_${new Date().getTime()}.pdf`);
        } catch (error: any) {
            console.error('PDF Error:', error);
            alert('Erro ao gerar PDF: O navegador está tendo dificuldade em processar os gráficos. Dica: Tente usar o botão de imprimir do sistema (Ctrl+P) e escolha "Salvar como PDF".');
        } finally {
            setIsExporting(false);
        }
    };

    return (
        <div className="flex flex-col md:flex-row h-screen bg-[#020617] overflow-hidden text-white font-sans selection:bg-exxata-blue/30">
            {/* Sidebar */}
            <aside className="w-full md:w-80 bg-[#1E293B] border-r border-white/5 p-6 flex flex-col gap-6 overflow-y-auto">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-exxata-red rounded-xl flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-exxata-red/20 uppercase">E</div>
                    <div>
                        <h1 className="font-manrope font-extrabold text-exxata-red leading-tight text-lg">EXXATA</h1>
                        <p className="text-[10px] font-bold text-exxata-blue tracking-widest uppercase">Monte Carlo</p>
                    </div>
                </div>

                <div className="space-y-6 flex-1">
                    <div className="space-y-4 pt-4 border-t border-white/5">
                        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                            <Settings size={14} /> Configurações
                        </h3>
                        <div className="space-y-3">
                            <div className="space-y-1">
                                <label className="text-xs font-bold text-slate-400">Item / Negociação</label>
                                <input
                                    value={itemName}
                                    onChange={e => setItemName(e.target.value)}
                                    className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-exxata-blue/20 outline-none transition-all"
                                />
                            </div>
                            <div className="grid grid-cols-1 gap-3">
                                {[
                                    { label: 'A — Piso (Mínimo)', val: pisoText, set: setPisoText },
                                    { label: 'B — Provável (Moda)', val: provavelText, set: setProvavelText },
                                    { label: 'C — Teto (Máximo)', val: tetoText, set: setTetoText },
                                ].map((input, idx) => (
                                    <div key={idx} className="space-y-1">
                                        <label className="text-xs font-bold text-slate-400">{input.label}</label>
                                        <div className="relative">
                                            <span className="absolute left-3 top-2 text-slate-500 text-xs font-bold">R$</span>
                                            <input
                                                value={input.val}
                                                onChange={e => input.set(e.target.value)}
                                                className="w-full bg-black/20 border border-white/10 rounded-lg pl-8 pr-3 py-2 text-sm focus:ring-2 focus:ring-exxata-blue/20 outline-none"
                                            />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    <div className="space-y-4 pt-2">
                        <button
                            onClick={() => setShowAdvanced(!showAdvanced)}
                            className="w-full flex items-center justify-between text-xs font-bold text-slate-400 uppercase tracking-wider hover:text-white transition-colors"
                        >
                            <span className="flex items-center gap-2"><Target size={14} /> Avançado (opcional)</span>
                            {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                        </button>

                        {showAdvanced && (
                            <div className="space-y-3 animate-in fade-in slide-in-from-top-2">
                                <div className="space-y-1">
                                    <label className="text-[10px] font-bold text-slate-500">Limites R$ (máx. 8, sep. por vírgula)</label>
                                    <textarea
                                        value={manualLimits}
                                        onChange={e => setManualLimits(e.target.value)}
                                        placeholder="Ex: 2100000, 2400000, 2800000"
                                        className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-xs h-20 outline-none focus:ring-1 focus:ring-exxata-blue/30"
                                    />
                                </div>
                                <div className="space-y-1">
                                    <label className="text-[10px] font-bold text-slate-500">Precisão: {(iterations / 1000).toFixed(0)}k iterações</label>
                                    <input
                                        type="range" min="10000" max="500000" step="10000" value={iterations}
                                        onChange={e => setIterations(parseInt(e.target.value))}
                                        className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-exxata-blue"
                                    />
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                <button
                    onClick={handleSimulate}
                    className="btn-primary w-full flex items-center justify-center gap-2 mt-auto"
                >
                    <Play size={18} fill="currentColor" /> Rodar Simulação
                </button>

                {/* Hidden Signature */}
                <div className="mt-4 pt-4 border-t border-white/5 flex justify-center">
                    <a
                        href="https://github.com/chamonefelipe11-spec"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[10px] font-medium text-slate-600 hover:text-exxata-blue transition-colors duration-500 opacity-30 hover:opacity-100 italic"
                    >
                        (Cha)² — chamonefelipe11-spec
                    </a>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 overflow-y-auto w-full">
                <div ref={dashboardRef} className="p-4 md:p-8 space-y-8 bg-[#020617]">
                    <header className="flex flex-col md:flex-row justify-between items-start gap-4 mb-8">
                        <div>
                            <h2 className="text-2xl font-manrope font-extrabold text-white">{itemName}</h2>
                            <p className="text-slate-400 text-sm flex items-center gap-1.5">
                                <Info size={14} className="text-exxata-blue" /> Simulação triangular estocástica baseada em riscos comerciais.
                            </p>
                        </div>
                        <div className="flex gap-2">
                            <button
                                onClick={exportPDF}
                                disabled={isExporting}
                                className={`flex items-center gap-2 bg-white/5 border border-white/10 text-white px-4 py-2 rounded-xl text-xs font-bold transition-all shadow-sm ${isExporting ? 'opacity-50 cursor-not-allowed' : 'hover:bg-white/10'}`}
                            >
                                <FileText size={16} />
                                {isExporting ? 'Gerando PDF...' : 'Baixar Relatório PDF'}
                            </button>
                            <button onClick={exportExcel} className="flex items-center gap-2 bg-white/5 border border-white/10 text-white px-4 py-2 rounded-xl text-xs font-bold hover:bg-white/10 transition-all shadow-sm">
                                <Download size={16} /> Exportar XLSX
                            </button>
                        </div>
                    </header>

                    {stats ? (
                        <div className="space-y-6">
                            {/* KPI Grid */}
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                                {[
                                    { label: 'E.V. (Média)', val: formatBRL(stats.mean), color: 'text-exxata-blue', bg: 'bg-exxata-blue/5', icon: Target },
                                    { label: 'P50 (Mediana)', val: formatBRL(stats.median), color: 'text-slate-300', bg: 'bg-white/5', icon: TrendingUp },
                                    { label: 'P95 (Cenário Alto)', val: formatBRL(stats.p95), color: 'text-exxata-red', bg: 'bg-exxata-red/5', icon: BarChart3 },
                                    { label: 'Performance', val: `${stats.duration.toFixed(0)} ms`, color: 'text-slate-500', bg: 'bg-white/5', icon: Calculator },
                                ].map((kpi, i) => (
                                    <div key={i} className={`glass-card p-5 rounded-2xl ${kpi.bg} border-white/5`}>
                                        <div className="flex justify-between items-start mb-1">
                                            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest block">{kpi.label}</span>
                                            <kpi.icon size={14} className={kpi.color} />
                                        </div>
                                        <div className={`text-xl font-manrope font-extrabold truncate ${kpi.color}`}>{kpi.val}</div>
                                    </div>
                                ))}
                            </div>

                            {/* Charts Area */}
                            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                                <div className="glass-card p-6 rounded-3xl min-h-[400px] bg-white/5 border-white/5">
                                    <h4 className="font-bold text-slate-300 mb-6 flex items-center gap-2 text-sm uppercase tracking-tighter">
                                        <BarChart3 size={16} className="text-exxata-red" /> Distribuição de Frequência
                                    </h4>
                                    <div className="h-72">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={stats.bins}>
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                                                <XAxis dataKey="range" hide />
                                                <YAxis hide />
                                                <Tooltip
                                                    contentStyle={{ backgroundColor: '#1E293B', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', color: '#fff' }}
                                                    labelFormatter={(v) => formatBRL(v)}
                                                    formatter={(v: any) => [v, 'Ocorrências']}
                                                />
                                                <Bar dataKey="count" fill="#D51D07" radius={[4, 4, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>

                                <div className="glass-card p-6 rounded-3xl min-h-[400px] bg-white/5 border-white/5">
                                    <h4 className="font-bold text-slate-300 mb-6 flex items-center gap-2 text-sm uppercase tracking-tighter">
                                        <TrendingUp size={16} className="text-exxata-blue" /> Curva de Probabilidade Acumulada (S-Curve)
                                    </h4>
                                    <div className="h-72">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={stats.cdfData}>
                                                <defs>
                                                    <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="5%" stopColor="#4284D7" stopOpacity={0.3} />
                                                        <stop offset="95%" stopColor="#4284D7" stopOpacity={0} />
                                                    </linearGradient>
                                                </defs>
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                                                <XAxis
                                                    dataKey="value"
                                                    type="number"
                                                    domain={['dataMin', 'dataMax']}
                                                    hide
                                                />
                                                <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} stroke="#64748B" fontSize={10} />
                                                <Tooltip
                                                    contentStyle={{ backgroundColor: '#1E293B', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', color: '#fff' }}
                                                    labelFormatter={(v) => formatBRL(v)}
                                                    formatter={(v: any) => [`${(v * 100).toFixed(1)}%`, 'P-Value']}
                                                />
                                                <Area type="monotone" dataKey="probability" stroke="#4284D7" strokeWidth={3} fill="url(#colorProb)" connectNulls />

                                                {/* Linhas de Referência Fiscais */}
                                                <ReferenceLine x={stats.median} stroke="#94A3B8" strokeDasharray="3 3" strokeWidth={1.5}>
                                                    <Label value={`P50`} position="top" fill="#94A3B8" fontSize={10} />
                                                </ReferenceLine>
                                                <ReferenceLine x={stats.p95} stroke="#D51D07" strokeDasharray="3 3" strokeWidth={1.5}>
                                                    <Label value={`P95`} position="top" fill="#D51D07" fontSize={10} />
                                                </ReferenceLine>
                                                <ReferenceLine x={stats.mean} stroke="#4284D7" strokeDasharray="5 5" strokeWidth={2}>
                                                    <Label value={`EV`} position="bottom" fill="#4284D7" fontSize={10} />
                                                </ReferenceLine>
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </div>

                            {/* Probability Bands */}
                            <div className="space-y-4">
                                <h4 className="font-bold text-slate-300 text-sm uppercase tracking-tighter flex items-center gap-2">
                                    <Target size={16} className="text-exxata-blue" /> Probabilidade por Faixa de Acordo {manualLimits.trim() ? '(Manual)' : '(Automática)'}
                                </h4>
                                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                                    {stats.faixas.map((fx, i) => (
                                        <div key={i} className="bg-white/5 border border-white/10 p-4 rounded-2xl flex flex-col gap-2">
                                            <div className="flex justify-between items-start">
                                                <span className="text-[9px] font-bold text-slate-400 leading-tight uppercase block max-w-[70%]">{fx.label}</span>
                                                <span className="text-sm font-black text-exxata-blue shrink-0">{(fx.pct * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="progress-container h-1.5 bg-white/5">
                                                <div className="progress-bar" style={{ width: `${Math.max(fx.pct * 100, 1)}%` }}></div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            <div className="bg-exxata-blue/10 border-l-4 border-exxata-blue p-4 rounded-r-2xl flex items-start gap-3 mt-8">
                                <AlertCircle size={18} className="text-exxata-blue shrink-0 mt-0.5" />
                                <p className="text-[11px] text-slate-400 leading-normal font-medium">
                                    <strong>Atenção Estratégica:</strong> Estes dados são gerados por um modelo matemático de simulação triangular.
                                    Antes de qualquer movimentação comercial, valide estes cenários com a diretoria técnica e financeira da Exxata.
                                </p>
                            </div>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center h-64 text-slate-700">
                            <Calculator size={48} className="mb-4 animate-pulse opacity-20" />
                            <p className="text-sm font-bold uppercase tracking-widest opacity-20">Processando Inteligência...</p>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
};

export default Dashboard;
