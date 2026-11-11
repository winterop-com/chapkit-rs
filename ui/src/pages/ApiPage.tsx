import { DataTable, type DataColumn } from '@winterop-com/sokkel-ui/DataTable'
import { PageHeader } from '@winterop-com/sokkel-ui/PageHeader'
import { Badge } from '@winterop-com/sokkel-ui/primitives/badge'
import { Button } from '@winterop-com/sokkel-ui/primitives/button'
import { ExternalLink, FileJson } from 'lucide-react'
import { useMemo } from 'react'

import { api } from '../api/client'
import { useResource } from '../lib/resource'
import { ReadState } from '../parts'

interface Route {
    key: string
    method: string
    path: string
    tag: string
    summary: string
}

const COLUMNS: DataColumn<Route>[] = [
    {
        id: 'method',
        header: 'Method',
        width: '6rem',
        cell: (row) => (
            <Badge variant="outline" className="font-mono">
                {row.method}
            </Badge>
        ),
    },
    { id: 'path', header: 'Path', width: 'minmax(16rem,1.5fr)', cell: (row) => <span className="font-mono text-xs">{row.path}</span> },
    { id: 'summary', header: 'What it does', width: 'minmax(16rem,2fr)', cell: (row) => <span className="truncate text-muted-foreground">{row.summary}</span> },
]

/** API: the service's routes from its OpenAPI document, and the ways to its interactive docs. */
export function ApiPage() {
    const spec = useResource(api.openapi, [])
    const routes = useMemo(
        () =>
            Object.entries(spec.data?.paths ?? {}).flatMap(([path, methods]) =>
                Object.entries(methods).map(([method, operation]) => ({
                    key: `${method} ${path}`,
                    method: method.toUpperCase(),
                    path,
                    tag: operation.tags?.[0] ?? 'Other',
                    summary: ((operation.summary ?? '').split('\n')[0] ?? '').replace(/`/g, ''),
                })),
            ),
        [spec.data],
    )
    return (
        <>
            <PageHeader
                title="API"
                count={routes.length || undefined}
                description="The HTTP API this service offers Chap, compatible with Python chapkit."
                actions={
                    <>
                        <Button size="sm" variant="outline" nativeButton={false} render={<a href="/openapi.json" target="_blank" rel="noreferrer" />}>
                            <FileJson aria-hidden />
                            openapi.json
                        </Button>
                        <Button size="sm" nativeButton={false} render={<a href="/docs" target="_blank" rel="noreferrer" />}>
                            <ExternalLink aria-hidden />
                            Interactive docs
                        </Button>
                    </>
                }
            />
            {spec.error ? (
                <ReadState what="the OpenAPI document" resource={spec} primary />
            ) : (
                <DataTable
                    columns={COLUMNS}
                    rows={routes}
                    rowKey={(row) => row.key}
                    loading={spec.loading}
                    groupBy="tag"
                    rowHeight="fixed"
                    minWidth="40rem"
                    bar={false}
                />
            )}
        </>
    )
}
