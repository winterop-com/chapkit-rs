import type { MetricId } from '../api/types'

/** The scores a backtest reports, by chap-core's names, in the order they are shown. */
export const METRICS: readonly { id: MetricId; name: string; share?: boolean; info: string }[] = [
    { id: 'crps', name: 'CRPS', info: 'Continuous ranked probability score: how far the forecast distribution is from what happened. Lower is better.' },
    { id: 'mae', name: 'MAE', info: 'Mean absolute error of the median forecast. Lower is better.' },
    { id: 'rmse', name: 'RMSE', info: 'Root mean squared error of the median forecast. Lower is better.' },
    { id: 'coverage_10_90', name: 'Coverage 10-90', share: true, info: 'Share of observations inside the 10-90 interval; 80% is calibrated.' },
    { id: 'coverage_25_75', name: 'Coverage 25-75', share: true, info: 'Share of observations inside the 25-75 interval; 50% is calibrated.' },
    { id: 'crps_log1p', name: 'CRPS (log1p)', info: 'CRPS on log(1 + cases), so large counts weigh less.' },
    { id: 'crps_norm', name: 'CRPS Normalized', info: 'CRPS divided by the range of the observations.' },
    { id: 'mape', name: 'MAPE', info: 'Mean absolute percentage error of the median forecast.' },
    { id: 'winkler_score_10_90', name: 'Winkler Score 10-90', info: 'Width of the 10-90 interval plus a penalty for observations outside it.' },
    { id: 'winkler_score_25_75', name: 'Winkler Score 25-75', info: 'Width of the 25-75 interval plus a penalty for observations outside it.' },
    { id: 'ratio_above_truth', name: 'Ratio Above Truth', share: true, info: 'Share of forecast samples above the observation; 50% is unbiased.' },
]

export const metricName = (id: MetricId) => METRICS.find((metric) => metric.id === id)?.name ?? id

/** A score written for a reader: shares as percentages, other figures to three significant digits. */
export function score(value: number | null | undefined, share = false): string {
    if (value === null || value === undefined || Number.isNaN(value)) return '-'
    if (share) return `${(value * 100).toFixed(1)}%`
    if (Math.abs(value) >= 1000) return Math.round(value).toLocaleString('en')
    const text = value.toPrecision(3)
    // Only zeros after a decimal point are padding: 140 must stay 140, not 14.
    return text.includes('.') && !text.includes('e') ? text.replace(/\.?0+$/, '') : text
}

/** A count with thousands grouped. */
export const count = (value: number | null | undefined) => (value === null || value === undefined ? '-' : value.toLocaleString('en'))

/** A timestamp from the service as a date and time, in the reader's zone. */
export function when(stamp: string | null | undefined): string {
    if (stamp === null || stamp === undefined) return '-'
    const date = new Date(/[zZ]|[+-]\d\d:\d\d$/.test(stamp) ? stamp : `${stamp}Z`)
    if (Number.isNaN(date.getTime())) return stamp
    return date.toLocaleString('en-GB', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

/** A timestamp from the service as a Date. */
export const dateOf = (stamp: string) => new Date(/[zZ]|[+-]\d\d:\d\d$/.test(stamp) ? stamp : `${stamp}Z`)

/** How long between two timestamps, in words. */
export function took(from: string | null | undefined, to: string | null | undefined): string {
    if (from === null || from === undefined || to === null || to === undefined) return '-'
    const ms = dateOf(to).getTime() - dateOf(from).getTime()
    if (ms < 1000) return `${Math.max(0, Math.round(ms))} ms`
    if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`
    return `${Math.floor(ms / 60_000)} min ${Math.round((ms % 60_000) / 1000)} s`
}

/** Bytes for a reader. */
export function bytes(size: number | null | undefined): string {
    if (size === null || size === undefined) return '-'
    if (size < 1024) return `${size} B`
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} kB`
    return `${(size / 1024 / 1024).toFixed(1)} MB`
}

/** A period type for a reader. */
export const periodWord = (type: string | null | undefined) =>
    ({ month: 'Monthly', week: 'Weekly', day: 'Daily', year: 'Yearly', any: 'Any' })[type ?? ''] ?? type ?? '-'

/** A short id: the last characters of a ULID, which differ between ids made close together. */
export const shortId = (id: string) => id.slice(-8)

/**
 * A period as the service writes it after chap-core's preparation, from the way a file wrote it: a
 * week given as its dates (`2019-01-07/2019-01-13`) is its ISO week (`2019W02`), a month given as
 * `201901` is `2019-01`. Others are kept as they are.
 */
export function canonicalPeriod(period: string): string {
    const week = /^(\d{4})-(\d{2})-(\d{2})\/\d{4}-\d{2}-\d{2}$/.exec(period)
    if (week !== null) {
        const day = new Date(Date.UTC(Number(week[1]), Number(week[2]) - 1, Number(week[3])))
        // The ISO week's Thursday decides its year.
        const thursday = new Date(day.getTime() + (3 - ((day.getUTCDay() + 6) % 7)) * 86_400_000)
        const year = thursday.getUTCFullYear()
        const first = Date.UTC(year, 0, 1)
        const number = Math.floor((thursday.getTime() - first) / 86_400_000 / 7) + 1
        return `${year}W${String(number).padStart(2, '0')}`
    }
    const month = /^(\d{4})(\d{2})$/.exec(period)
    if (month !== null) return `${month[1]}-${month[2]}`
    return period
}
