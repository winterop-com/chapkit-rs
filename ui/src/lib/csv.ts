import { saveBlob } from '../api/client'

/** Reads CSV text into its header and rows of cells: quoted fields, doubled quotes, CRLF. */
export function parseCsv(text: string): { header: string[]; rows: string[][] } {
    const rows: string[][] = []
    let row: string[] = []
    let cell = ''
    let quoted = false
    for (let at = 0; at < text.length; at++) {
        const char = text[at]
        if (quoted) {
            if (char === '"' && text[at + 1] === '"') {
                cell += '"'
                at++
            } else if (char === '"') quoted = false
            else cell += char
        } else if (char === '"') quoted = true
        else if (char === ',') {
            row.push(cell)
            cell = ''
        } else if (char === '\n' || char === '\r') {
            if (char === '\r' && text[at + 1] === '\n') at++
            row.push(cell)
            if (row.some((value) => value !== '')) rows.push(row)
            row = []
            cell = ''
        } else cell += char
    }
    if (cell !== '' || row.length > 0) {
        row.push(cell)
        rows.push(row)
    }
    const [header = [], ...rest] = rows
    return { header, rows: rest }
}

/**
 * A cell as a spreadsheet must not read it as a formula: text starting with `=`, `+`, `-`, `@`, a
 * tab or a carriage return gets a leading `'` (OWASP's CSV injection advice). Numbers stay numbers.
 */
export function inert(value: unknown): string {
    if (value === null || value === undefined) return ''
    const text = String(value)
    return typeof value === 'string' && /^[=+\-@\t\r]/.test(text) ? `'${text}` : text
}

const quote = (value: unknown) => {
    const text = inert(value)
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

/** Writes rows of records as CSV for a spreadsheet, the columns in the order given; see `inert`. */
export function toCsv<T extends object>(rows: readonly T[], columns: readonly (keyof T & string)[]): string {
    const lines = [columns.map(quote).join(','), ...rows.map((row) => columns.map((column) => quote(row[column])).join(','))]
    return `${lines.join('\n')}\n`
}

/** Hands the reader a file made in the page. */
export const saveFile = (name: string, text: string, type = 'text/csv'): void => saveBlob(name, new Blob([text], { type }))
