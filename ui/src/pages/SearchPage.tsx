import { PageHeader } from '@winterop-com/sokkel-ui/PageHeader'
import { SearchResults, type SearchChosen, type SearchFacet } from '@winterop-com/sokkel-ui/SearchResults'
import { KIND_FACET } from '@winterop-com/sokkel-ui/lib/search'
import { toast } from '@winterop-com/sokkel-ui/primitives/toast'
import { useEffect, useMemo, useState } from 'react'

import { asApiError } from '../api/client'
import { useResource } from '../lib/resource'
import { go, hashParam } from '../lib/router'
import { findPages, foundGroups, joinFound, nextPages, resultRoute, RESULTS_SIZE, SEARCH_KINDS, searchRoute, type Found, type SearchKind } from '../lib/search'

/**
 * Every dataset, run and config a search finds, a page of each kind at a time, narrowed by kind:
 * where the top bar's search goes on from its few of each. The query lives in the address, so going
 * back from a result comes back to it.
 */
export function SearchPage() {
    const [query, setQuery] = useState(() => hashParam('q') ?? '')
    const [asked, setAsked] = useState(query.trim())
    const [chosen, setChosen] = useState<SearchChosen>({})
    // A query given from outside, such as the top bar's See all while this page is open, takes the field's place.
    useEffect(() => {
        const given = () => setQuery(hashParam('q') ?? '')
        window.addEventListener('hashchange', given)
        return () => window.removeEventListener('hashchange', given)
    }, [])
    // The query is asked after a pause in typing, and kept in the address without a history entry each key.
    useEffect(() => {
        const trimmed = query.trim()
        history.replaceState(history.state, '', `#${searchRoute(trimmed)}`)
        const timer = setTimeout(() => setAsked(trimmed), 250)
        return () => clearTimeout(timer)
    }, [query])

    const first = useResource(asked === '' ? null : (signal) => findPages(asked, { datasets: 1, evaluations: 1, forecasts: 1, configs: 1 }, RESULTS_SIZE, signal), [asked])
    // The pages read on with More, for the query they were read for.
    const [more, setMore] = useState<{ asked: string; found: Partial<Found> }>()
    const [loadingMore, setLoadingMore] = useState(false)
    const found = useMemo(() => {
        if (first.data === undefined) return undefined
        return more?.asked === asked ? joinFound(first.data, more.found) : first.data
    }, [first.data, more, asked])

    const groups = useMemo(() => (found === undefined ? [] : foundGroups(found)), [found])
    const kinds = (chosen[KIND_FACET] ?? []) as SearchKind[]
    const shown = groups.filter((group) => kinds.length === 0 || kinds.includes(group.kind as SearchKind))
    // The service counts each kind, so the facet says how many match, not how many are listed.
    const facets: SearchFacet[] = [{ id: KIND_FACET, title: 'Kind', values: groups.map((group) => ({ value: group.kind, label: group.title, count: group.total ?? group.results.length })) }]
    const next = found === undefined ? {} : nextPages(found, RESULTS_SIZE, kinds.length === 0 ? SEARCH_KINDS : kinds)

    const readMore = () => {
        if (found === undefined || loadingMore) return
        const forQuery = asked
        setLoadingMore(true)
        findPages(forQuery, next, RESULTS_SIZE, new AbortController().signal)
            .then((page) => setMore((before) => ({ asked: forQuery, found: joinFound(before?.asked === forQuery ? before.found : {}, page) })))
            .catch((failure: unknown) => toast.add({ type: 'error', title: 'Could not read more results', description: asApiError(failure).message }))
            .finally(() => setLoadingMore(false))
    }

    return (
        <>
            <PageHeader title="Search" description="Datasets and configs by name, evaluations and forecasts by the name of their dataset, newest runs first." />
            <SearchResults
                query={query}
                onQueryChange={setQuery}
                groups={shown}
                facets={groups.length > 0 ? facets : []}
                chosen={chosen}
                onChosenChange={setChosen}
                total={shown.reduce((sum, group) => sum + (group.total ?? group.results.length), 0)}
                onOpen={(result, group) => go(resultRoute(group.kind, result.id))}
                onMore={Object.keys(next).length > 0 ? readMore : undefined}
                loadingMore={loadingMore}
                state={asked === '' ? 'done' : first.error !== undefined ? 'failed' : first.loading || query.trim() !== asked ? 'loading' : 'done'}
                words={{
                    placeholder: 'Search datasets, runs and configs',
                    empty: (typed) => (typed === '' ? 'Type a name to search datasets, runs and configs.' : `Nothing matches “${typed}”.`),
                    ...(first.error !== undefined ? { failed: `The search did not answer: ${first.error.detail ?? first.error.message}` } : {}),
                }}
            />
        </>
    )
}
