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

const quote = (value: unknown) => {
    const text = value === null || value === undefined ? '' : String(value)
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

/** Writes rows of records as CSV, the columns in the order given. */
export function toCsv<T extends object>(rows: readonly T[], columns: readonly (keyof T & string)[]): string {
    const lines = [columns.join(','), ...rows.map((row) => columns.map((column) => quote(row[column])).join(','))]
    return `${lines.join('\n')}\n`
}

/** Hands the reader a file made in the page. */
export function saveFile(name: string, text: string, type = 'text/csv'): void {
    const url = URL.createObjectURL(new Blob([text], { type }))
    const link = document.createElement('a')
    link.href = url
    link.download = name
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
}
