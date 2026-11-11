import { AccountMenu } from '@winterop-com/sokkel-ui/AccountMenu'
import { Banner } from '@winterop-com/sokkel-ui/Banner'
import type { PaletteGroup } from '@winterop-com/sokkel-ui/CommandPalette'
import { useAppearance } from '@winterop-com/sokkel-ui/hooks/use-appearance'
import { THEMES } from '@winterop-com/sokkel-ui/lib/appearance'
import { appearanceCommands, placeCommands } from '@winterop-com/sokkel-ui/lib/commands'
import { Rail, type RailGroup } from '@winterop-com/sokkel-ui/Rail'
import { SchemeToggle } from '@winterop-com/sokkel-ui/SchemeToggle'
import { Page, Shell } from '@winterop-com/sokkel-ui/Shell'
import { StatusItem, StatusLine } from '@winterop-com/sokkel-ui/StatusLine'
import { TopBar, type Crumb } from '@winterop-com/sokkel-ui/TopBar'
import { Badge } from '@winterop-com/sokkel-ui/primitives/badge'
import { Button } from '@winterop-com/sokkel-ui/primitives/button'
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from '@winterop-com/sokkel-ui/primitives/empty'
import { Boxes, Braces, Database, Gauge, LayoutDashboard, ListChecks, Server, SlidersHorizontal, TrendingUp } from 'lucide-react'
import { createContext, lazy, Suspense, useContext, useMemo, type ReactNode } from 'react'

import { api } from './api/client'
import type { Health, Info } from './api/types'
import { useResource } from './lib/resource'
import { go, useRoute } from './lib/router'

// Each page loads the first time it is opened.
const ApiPage = lazy(() => import('./pages/ApiPage').then((module) => ({ default: module.ApiPage })))
const Artifacts = lazy(() => import('./pages/Artifacts').then((module) => ({ default: module.Artifacts })))
const Configs = lazy(() => import('./pages/Configs').then((module) => ({ default: module.Configs })))
const DatasetPage = lazy(() => import('./pages/DatasetPage').then((module) => ({ default: module.DatasetPage })))
const Datasets = lazy(() => import('./pages/Datasets').then((module) => ({ default: module.Datasets })))
const Evaluate = lazy(() => import('./pages/Evaluate').then((module) => ({ default: module.Evaluate })))
const Forecast = lazy(() => import('./pages/Forecast').then((module) => ({ default: module.Forecast })))
const Jobs = lazy(() => import('./pages/Jobs').then((module) => ({ default: module.Jobs })))
const Overview = lazy(() => import('./pages/Overview').then((module) => ({ default: module.Overview })))
const System = lazy(() => import('./pages/System').then((module) => ({ default: module.System })))

/** The model the app serves, read once from the service. */
const ModelContext = createContext<Info | undefined>(undefined)

/** The model's info; undefined until the service has answered. */
export const useModel = () => useContext(ModelContext)

const PLACES = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'datasets', label: 'Datasets', icon: Database },
    { id: 'evaluate', label: 'Evaluate', icon: Gauge },
    { id: 'forecast', label: 'Forecast', icon: TrendingUp },
    { id: 'configs', label: 'Configs', icon: SlidersHorizontal },
    { id: 'jobs', label: 'Jobs', icon: ListChecks },
    { id: 'artifacts', label: 'Artifacts', icon: Boxes },
    { id: 'system', label: 'System', icon: Server },
    { id: 'api', label: 'API', icon: Braces },
] as const

/** The app's mark: a rising forecast line over a band. */
function Mark() {
    return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden className="size-4">
            <path d="M3 17l5-5 4 3 8-8" />
            <path d="M3 21h18" opacity="0.5" />
        </svg>
    )
}

function healthNote(health: Health | undefined, failed: boolean): { note: string; tone?: 'warn' | 'error' } {
    if (failed) return { note: 'Service not answering', tone: 'error' }
    if (health === undefined) return { note: 'Connecting' }
    if (health.status === 'healthy') return { note: 'Service healthy' }
    return { note: `Service ${health.status}`, tone: 'warn' }
}

/** The Modeling app for one model: the rail of its places, the palette, and the page the route names. */
export function App() {
    const { settings, setSettings } = useAppearance()
    const route = useRoute()
    const [first = 'overview', second] = route
    const page = PLACES.some((place) => place.id === first) ? first : 'missing'
    const info = useResource(api.info, [])
    const health = useResource(api.health, [], 15_000)
    const jobs = useResource(api.jobs, [], 5_000)
    const running = (jobs.data ?? []).filter((job) => job.status === 'running' || job.status === 'pending').length

    const groups: RailGroup[] = useMemo(() => {
        const entry = (id: (typeof PLACES)[number]['id']) => {
            const place = PLACES.find((each) => each.id === id)!
            return {
                id,
                label: place.label,
                icon: place.icon,
                href: id === 'overview' ? '#/' : `#/${id}`,
                active: page === id,
                ...(id === 'jobs' && running > 0 ? { count: running } : {}),
            }
        }
        return [
            { id: 'home', entries: [entry('overview')] },
            { id: 'model', title: 'Model', entries: [entry('datasets'), entry('evaluate'), entry('forecast'), entry('configs')] },
            { id: 'service', title: 'Service', entries: [entry('jobs'), entry('artifacts'), entry('system'), entry('api')] },
        ]
    }, [page, running])

    const palette: PaletteGroup[] = [
        ...placeCommands(groups, (href) => go(href.slice(1))),
        ...appearanceCommands(settings, setSettings, { themes: THEMES.filter((theme) => theme.value === 'tidevann') }).map((group) => ({
            ...group,
            rank: 20,
        })),
    ]

    const model = info.data
    const label = PLACES.find((place) => place.id === page)?.label ?? 'Not found'
    const trail: Crumb[] = [{ label: model?.display_name ?? 'Model', href: '#/' }]
    if (page !== 'overview') trail.push(second === undefined ? { label } : { label, href: `#/${page}` })
    if (second !== undefined) trail.push({ label: second.length > 12 ? second.slice(-8) : second, code: true })
    const state = healthNote(health.data, health.error !== undefined)

    let content: ReactNode
    if (page === 'overview') content = <Overview />
    else if (page === 'datasets') content = second === undefined ? <Datasets /> : <DatasetPage id={second} />
    else if (page === 'evaluate') content = <Evaluate opened={second} />
    else if (page === 'forecast') content = <Forecast opened={second} />
    else if (page === 'configs') content = <Configs />
    else if (page === 'jobs') content = <Jobs />
    else if (page === 'artifacts') content = <Artifacts opened={second} />
    else if (page === 'system') content = <System />
    else if (page === 'api') content = <ApiPage />
    else
        content = (
            <Empty>
                <EmptyHeader>
                    <EmptyTitle>Nothing at this address</EmptyTitle>
                    <EmptyDescription>There is no page at #/{route.join('/')}.</EmptyDescription>
                </EmptyHeader>
                <Button size="sm" variant="outline" onClick={() => go('/')}>
                    Go to the overview
                </Button>
            </Empty>
        )

    return (
        <ModelContext.Provider value={model}>
            <Shell
                palette={palette}
                route={route.join('/')}
                banner={
                    info.error !== undefined ? (
                        <Banner tone="critical" actions={<Button size="xs" variant="outline" onClick={info.reload}>Try again</Button>}>
                            The model service does not answer: {info.error.message}.
                        </Banner>
                    ) : undefined
                }
                rail={
                    <Rail
                        mark={<Mark />}
                        markHref="#/"
                        name="Modeling"
                        groups={groups}
                        foot={
                            <AccountMenu
                                name="Local"
                                local
                                where={model?.id}
                                scheme={settings.scheme}
                                onSchemeChange={(scheme) => setSettings({ ...settings, scheme })}
                            />
                        }
                    />
                }
                top={
                    <TopBar
                        trail={trail}
                        actions={
                            <>
                                {model !== undefined && (
                                    <Badge variant="outline" className="font-mono">
                                        v{model.version}
                                    </Badge>
                                )}
                                <SchemeToggle />
                            </>
                        }
                    />
                }
                foot={
                    <StatusLine note={state.note} tone={state.tone} live={running > 0}>
                        {running > 0 && <StatusItem label="running">{running}</StatusItem>}
                        {model !== undefined && (
                            <StatusItem label="model" code>
                                {model.id}
                            </StatusItem>
                        )}
                        {model?.chapkit_rs_version !== undefined && <StatusItem label="chapkit-rs">{model.chapkit_rs_version}</StatusItem>}
                    </StatusLine>
                }
            >
                <Page>
                    <Suspense fallback={null}>{content}</Suspense>
                </Page>
            </Shell>
        </ModelContext.Provider>
    )
}
