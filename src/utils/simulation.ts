export interface SimulationResults {
    values: number[];
    mean: number;
    median: number;
    p95: number;
    duration: number;
}

/**
 * Gera uma simulação de Monte Carlo usando Distribuição Triangular (A, B, C)
 */
export function runSimulation(
    min: number,
    mode: number,
    max: number,
    iterations: number = 200000
): SimulationResults {
    const startTime = performance.now();
    const values = new Float64Array(iterations);

    const F = (mode - min) / (max - min);

    for (let i = 0; i < iterations; i++) {
        const U = Math.random();
        if (U < F) {
            values[i] = min + Math.sqrt(U * (max - min) * (mode - min));
        } else {
            values[i] = max - Math.sqrt((1 - U) * (max - min) * (max - mode));
        }
    }

    // Ordenar para cálculos de quantis
    values.sort();

    const sum = values.reduce((a, b) => a + b, 0);
    const mean = sum / iterations;
    const median = values[Math.floor(iterations * 0.5)];
    const p95 = values[Math.floor(iterations * 0.95)];
    const duration = performance.now() - startTime;

    return {
        values: Array.from(values),
        mean,
        median,
        p95,
        duration
    };
}

export function formatBRL(val: number): string {
    return new Intl.NumberFormat('pt-BR', {
        style: 'currency',
        currency: 'BRL',
    }).format(val);
}

export function parseBRL(txt: string): number {
    const cleaned = txt.replace(/[^\d,]/g, '').replace(',', '.');
    const result = parseFloat(cleaned);
    return isNaN(result) ? 0 : result;
}

/**
 * Calcula faixas de probabilidade para limites fixos
 */
export function calculateFaixas(values: number[], limits: number[]): { label: string, pct: number }[] {
    const faixas = [];
    const sortedLimits = [...limits].sort((a, b) => a - b);
    const edges = [-Infinity, ...sortedLimits, Infinity];

    for (let i = 0; i < edges.length - 1; i++) {
        const lo = edges[i];
        const hi = edges[i + 1];
        let label = '';

        // Contagem otimizada usando os valores já ordenados
        let count = 0;
        for (let j = 0; j < values.length; j++) {
            if (values[j] >= lo && values[j] < hi) {
                count++;
            } else if (values[j] >= hi) {
                break; // Aproveita que os valores estão ordenados
            }
        }

        const pct = count / values.length;

        if (i === 0) label = `Abaixo de ${formatBRL(hi)}`;
        else if (i === edges.length - 2) label = `Acima de ${formatBRL(edges[i])}`;
        else label = `Entre ${formatBRL(lo)} e ${formatBRL(hi)}`;

        faixas.push({ label, pct });
    }

    return faixas;
}
